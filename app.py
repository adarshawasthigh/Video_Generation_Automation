import os
import cv2
import requests
import subprocess
import operator
import sys
from typing import List, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI
import fal_client

# Optional: Load environment variables from a .env file if running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FAL_KEY = os.getenv("FAL_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Fail  if keys are missing, rather than crashing deep in the code
missing_keys = [k for k, v in {"GEMINI_API_KEY": GEMINI_API_KEY, "FAL_KEY": FAL_KEY, "SUPABASE_URL": SUPABASE_URL, "SUPABASE_KEY": SUPABASE_KEY}.items() if not v]
if missing_keys:
    print(f"ERROR: Missing Environment Variables: {', '.join(missing_keys)}")
    print("Please set these in your environment or a .env file before running.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# STATE ,SCHEMAS
class VideoState(TypedDict):
    input_type: Literal["description", "script"]
    input_text: str
    scenes: List[dict]
    video_clips: Annotated[List[str], operator.add]
    current_start_frame: str
    index: int
    session_id: str

class Scene(BaseModel):
    prompt: str = Field(description="The highly detailed, visual prompt for the video generator. Focus on camera angles, lighting, and action.")

class Storyboard(BaseModel):
    scenes: List[Scene] = Field(description="The sequential list of scenes for the video.")

# UTILITY FUNCTIONS
def process_and_upload(local_video_path: str, session_id: str, idx: int) -> tuple[str, str]:
    """Extracts frame and forcefully upserts to Supabase to prevent crashes."""
    img_path = f"last_frame_{idx}.jpg"
    
    cap = cv2.VideoCapture(local_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    ret, frame = cap.read()
    if ret: 
        cv2.imwrite(img_path, frame)
    cap.release()
    
    frame_storage_path = f"{session_id}/frames/frame_{idx}.jpg"
    with open(img_path, 'rb') as f:
        supabase.storage.from_("video_assets").upload(frame_storage_path, f, file_options={"x-upsert": "true", "content-type": "image/jpeg"})
    frame_url = supabase.storage.from_("video_assets").get_public_url(frame_storage_path)

    video_storage_path = f"{session_id}/clips/clip_{idx}.mp4"
    with open(local_video_path, 'rb') as f:
        supabase.storage.from_("video_assets").upload(video_storage_path, f, file_options={"x-upsert": "true", "content-type": "video/mp4"})
    video_url = supabase.storage.from_("video_assets").get_public_url(video_storage_path)
    
    return frame_url, video_url

def stitch_and_upload(video_urls: List[str], session_id: str) -> str:
    """Downloads segments, stitches via FFmpeg, and uploads to Supabase."""
    print("\n🎬 STITCHING PROTOCOL INITIATED...")
    
    # Handle missing FFmpeg installation
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("\n SYSTEM ERROR: FFmpeg is not installed or not in your system PATH.")
        print("   The individual clips are in Supabase, but automated stitching is aborted.")
        print("   Install FFmpeg (e.g., 'brew install ffmpeg' on Mac) to enable this feature.")
        return "Stitching Aborted (Missing Dependency)"

    local_files = []
    for i, url in enumerate(video_urls):
        filename = f"stitch_temp_{i}.mp4"
        print(f" Downloading segment {i+1}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            local_files.append(os.path.abspath(filename))
        else:
            print(f"Failed to download {url}")
            return "Error: Could not download all segments."
        
    with open("concat_list.txt", "w") as f:
        for file in local_files:
            f.write(f"file '{file}'\n")
            
    print(" Running lossless FFmpeg merge...")
    final_filename = "final_master_stitched.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat_list.txt", "-c", "copy", final_filename],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n FFmpeg Stitching Failed! Error details:\n{e.stderr}")
        return "Error: FFmpeg Failed."
    
    print("Uploading final stitched master to Supabase...")
    final_storage_path = f"{session_id}/final_master.mp4"
    
    try:
        with open(final_filename, 'rb') as f:
            supabase.storage.from_("video_assets").upload(
                final_storage_path, 
                f, 
                file_options={"x-upsert": "true", "content-type": "video/mp4"}
            )
        final_url = supabase.storage.from_("video_assets").get_public_url(final_storage_path)
        print(" Final video successfully persisted to database!")
        return final_url
    except Exception as e:
        print(f"\n Supabase Upload Failed: {e}")
        return "Error: Database Upload Failed."

#NODES
def script_writer_node(state: VideoState):
    print("Gemini LLM for Storyboarding...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
    structured_llm = llm.with_structured_output(Storyboard)
    
    if state["input_type"] == "description":
        print("   -> Creating 40-second visual storyboard from product description...")
        sys_prompt = f"You are an expert AI Video Director. Write a highly visual, exactly 8-scene video storyboard for this product. Each scene represents 5 seconds of action, flowing continuously to create a seamless 40-second video. Description: {state['input_text']}"
    else:
        print("   -> Parsing provided script into exactly 8 visual scenes...")
        sys_prompt = f"You are an expert AI Video Director. Break this script down into exactly 8 sequential, highly visual scene prompts to build a continuous 40-second video. Script: {state['input_text']}"
        
    result = structured_llm.invoke(sys_prompt)
    formatted_scenes = [{"p": scene.prompt} for scene in result.scenes[:8]]
    
    print(f"\n Gemini successfully created {len(formatted_scenes)} distinct scenes!")
    print("\n 40-SECOND STORYBOARD ")
    for i, scene in enumerate(formatted_scenes):
        start_time = i * 5
        end_time = (i + 1) * 5
        print(f"Scene {i+1} [00:{start_time:02d} - 00:{end_time:02d}]: {scene['p']}\n")
    print("----------------------------------\n")
    
    # HITL APPROVAL
    approval = input(" HUMAN REVIEW: Do you approve this storyboard for generation? Type 'y' to proceed (Estimated API Cost: ~$3.20) or 'n' to abort: ").strip().lower()
    if approval != 'y':
        print("\n Generation aborted by user to save compute credits.")
        return {"scenes": [], "index": 0} 
    
    return {"scenes": formatted_scenes, "index": 0}

def generation_node(state: VideoState):
    idx = state["index"]
    scene = state["scenes"][idx]
    start_frame = state["current_start_frame"]
    
    print(f"\n GENERATING Segment {idx+1}/{len(state['scenes'])}")
    print(f"   -> Prompt: {scene['p'][:60]}...")
    
    print(" Calling Fal.ai Cloud GPU...")
    video_result = fal_client.subscribe(
        "fal-ai/wan-i2v",
        arguments={
            "prompt": scene["p"],
            "image_url": start_frame,
            "aspect_ratio": "16:9"
        }
    )
    fal_video_url = video_result['video']['url']
    print(" Fal generation complete! Downloading for continuity extraction...")
    
    local_video_path = f"generated_clip_{idx}.mp4"
    with open(local_video_path, 'wb') as f:
        f.write(requests.get(fal_video_url).content)
        
    last_frame_url, clip_url = process_and_upload(local_video_path, state["session_id"], idx)
    print(f" Segment {idx+1} synced to Supabase!\n")
    
    return {
        "video_clips": [clip_url],
        "current_start_frame": last_frame_url,
        "index": idx + 1
    }

# GRAPH ORCHESTRATION
def should_continue(state: VideoState):
    if state["index"] < len(state["scenes"]): return "generate"
    return END

workflow = StateGraph(VideoState)
workflow.add_node("script_writer", script_writer_node)
workflow.add_node("generate", generation_node)
workflow.set_entry_point("script_writer")
workflow.add_edge("script_writer", "generate")
workflow.add_conditional_edges("generate", should_continue)

app = workflow.compile(checkpointer=MemorySaver())

# INTERACTIVE EXECUTION BLOCK
if __name__ == "__main__":
    print(" AUTONOMOUS VIDEO ORCHESTRATION AGENT ")
  
    user_choice = input("Do you have a 'script' or a 'description'? (Type one): ").strip().lower()
    if user_choice not in ["script", "description"]:
        print(" Invalid input. Defaulting to 'description'.")
        user_choice = "description"
        
    user_text = input(f"\nPlease enter your {user_choice} here:\n> ").strip()
    session_name = input("\nEnter a unique session name for Supabase (e.g., ad_run_01):\n> ").strip()
    
    print("\nProvide a starting image URL for the AI to anchor the first scene on.")
    start_image = input("(Press Enter to use a default dark cinematic background): ").strip()
    if not start_image:
        start_image = "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=1280&h=720&auto=format&fit=crop"

    initial_state = {
        "input_type": user_choice, 
        "input_text": user_text,
        "session_id": session_name,
        "current_start_frame": start_image, 
        "video_clips": []
    }
    
    config = {"configurable": {"thread_id": "1"}}
    
    print("\n Starting Live Agent Pipeline...")
    final_state = None
    for event in app.stream(initial_state, config=config):
        pass
            
    final_state = app.get_state(config)
    video_urls = final_state.values.get("video_clips", [])
    
    # run stitching if we actually approved the generation and clips exist
    if video_urls:
        print("\n Graph Execution Complete! Individual Video URLs:")
        for i, clip in enumerate(video_urls):
            print(f"Segment {i+1}: {clip}")
            
        final_master_url = stitch_and_upload(video_urls, final_state.values["session_id"])
        
        print("\n=======================================================")
        print(" FINAL SEAMLESS VIDEO AD IS READY:")
        print(final_master_url)
        print("=======================================================")
    else:
        print("\n Pipeline gracefully terminated.")
