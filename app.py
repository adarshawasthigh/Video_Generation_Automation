import os
import re
import cv2
import requests
import subprocess
import operator
import sys
import tempfile
import shutil
from typing import List, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI
import fal_client

# Load environment variables from a .env file if running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FAL_KEY = os.getenv("FAL_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Raise RuntimeError instead of sys.exit(1) so importing this module
# in tests or a larger app doesn't kill the entire process.
missing_keys = [k for k, v in {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "FAL_KEY": FAL_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY
}.items() if not v]
if missing_keys:
    raise RuntimeError(
        f"Missing Environment Variables: {', '.join(missing_keys)}. "
        "Please set these in your environment or a .env file before running."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# STATE & SCHEMAS
class VideoState(TypedDict):
    input_type: Literal["description", "script"]
    input_text: str
    scenes: List[dict]
    video_clips: Annotated[List[str], operator.add]
    current_start_frame: str
    index: int
    session_id: str


class Scene(BaseModel):
    prompt: str = Field(
        description="The highly detailed, visual prompt for the video generator. "
                    "Focus on camera angles, lighting, and action."
    )


class Storyboard(BaseModel):
    scenes: List[Scene] = Field(description="The sequential list of scenes for the video.")


# UTILITY FUNCTIONS
def sanitize_session_name(name: str) -> str:
    """FIX #4: Sanitize session name to prevent path traversal in Supabase storage paths."""
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    if not sanitized:
        sanitized = "default_session"
    return sanitized


def process_and_upload(local_video_path: str, session_id: str, idx: int) -> tuple[str, str]:
    """Extracts last frame and forcefully upserts both frame and clip to Supabase."""
    #  Use a temp directory for intermediate files so they're cleaned up automatically.
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, f"last_frame_{idx}.jpg")

        cap = cv2.VideoCapture(local_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(img_path, frame)
        cap.release()

        frame_storage_path = f"{session_id}/frames/frame_{idx}.jpg"
        with open(img_path, 'rb') as f:
            supabase.storage.from_("video_assets").upload(
                frame_storage_path, f,
                file_options={"x-upsert": "true", "content-type": "image/jpeg"}
            )
        frame_url = supabase.storage.from_("video_assets").get_public_url(frame_storage_path)

        video_storage_path = f"{session_id}/clips/clip_{idx}.mp4"
        with open(local_video_path, 'rb') as f:
            supabase.storage.from_("video_assets").upload(
                video_storage_path, f,
                file_options={"x-upsert": "true", "content-type": "video/mp4"}
            )
        video_url = supabase.storage.from_("video_assets").get_public_url(video_storage_path)

    return frame_url, video_url


def stitch_and_upload(video_urls: List[str], session_id: str) -> str:
    """Downloads segments, stitches via FFmpeg, and uploads final video to Supabase."""
    print("\n STITCHING PROTOCOL INITIATED...")

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n SYSTEM ERROR: FFmpeg is not installed or not in your system PATH.")
        print("   The individual clips are in Supabase, but automated stitching is aborted.")
        print("   Install FFmpeg (e.g., 'brew install ffmpeg' on Mac) to enable this feature.")
        return "Stitching Aborted (Missing Dependency)"

    # Use a temp directory for all stitching intermediates — auto-cleaned on exit.
    with tempfile.TemporaryDirectory() as tmpdir:
        local_files = []
        for i, url in enumerate(video_urls):
            filename = os.path.join(tmpdir, f"stitch_temp_{i}.mp4")
            print(f" Downloading segment {i+1}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                local_files.append(filename)
            else:
                print(f"Failed to download {url}")
                return "Error: Could not download all segments."

        concat_list_path = os.path.join(tmpdir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for file in local_files:
                f.write(f"file '{file}'\n")

        final_filename = os.path.join(tmpdir, "final_master_stitched.mp4")
        print(" Running lossless FFmpeg merge...")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list_path, "-c", "copy", final_filename],
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
                    final_storage_path, f,
                    file_options={"x-upsert": "true", "content-type": "video/mp4"}
                )
            final_url = supabase.storage.from_("video_assets").get_public_url(final_storage_path)
            print(" Final video successfully persisted to database!")
            return final_url
        except Exception as e:
            print(f"\n Supabase Upload Failed: {e}")
            return "Error: Database Upload Failed."


# NODES
def script_writer_node(state: VideoState):
    print("Gemini LLM for Storyboarding...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)
    structured_llm = llm.with_structured_output(Storyboard)

    if state["input_type"] == "description":
        print("   -> Creating 40-second visual storyboard from product description...")
        sys_prompt = (
            "You are an expert AI Video Director. Write a highly visual, exactly 8-scene "
            "video storyboard for this product. Each scene represents 5 seconds of action, "
            f"flowing continuously to create a seamless 40-second video. Description: {state['input_text']}"
        )
    else:
        print("   -> Parsing provided script into exactly 8 visual scenes...")
        sys_prompt = (
            "You are an expert AI Video Director. Break this script down into exactly 8 sequential, "
            "highly visual scene prompts to build a continuous 40-second video. "
            f"Script: {state['input_text']}"
        )

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
    approval = input(
        " HUMAN REVIEW: Do you approve this storyboard for generation? "
        "Type 'y' to proceed (Estimated API Cost: ~$3.20) or 'n' to abort: "
    ).strip().lower()

    if approval != 'y':
        print("\n Generation aborted by user to save compute credits.")
        # Return empty scenes so the conditional edge after this node
        # routes to END instead of crashing in generation_node.
        return {"scenes": [], "index": 0}

    return {"scenes": formatted_scenes, "index": 0}


def generation_node(state: VideoState):
    idx = state["index"]
    scene = state["scenes"][idx]
    start_frame = state["current_start_frame"]

    print(f"\n GENERATING Segment {idx+1}/{len(state['scenes'])}")
    print(f"   -> Prompt: {scene['p'][:60]}...")

    # FIX #3: Wrap fal_client call in try/except with basic retry logic.
    max_retries = 3
    fal_video_url = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f" Calling Fal.ai Cloud GPU (attempt {attempt}/{max_retries})...")
            video_result = fal_client.subscribe(
                "fal-ai/wan-i2v",
                arguments={
                    "prompt": scene["p"],
                    "image_url": start_frame,
                    "aspect_ratio": "16:9"
                }
            )
            fal_video_url = video_result['video']['url']
            print(" Fal generation complete!")
            break
        except Exception as e:
            print(f"   Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Fal.ai generation failed after {max_retries} attempts for scene {idx+1}."
                ) from e

    print(" Downloading clip for continuity extraction...")
    # Use a temp file for the downloaded clip; clean up after upload.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        local_video_path = tmp.name

    try:
        with open(local_video_path, 'wb') as f:
            f.write(requests.get(fal_video_url).content)

        last_frame_url, clip_url = process_and_upload(local_video_path, state["session_id"], idx)
        print(f" Segment {idx+1} synced to Supabase!\n")
    finally:
        # Always clean up the local clip file, even if an error occurred.
        if os.path.exists(local_video_path):
            os.remove(local_video_path)

    return {
        "video_clips": [clip_url],
        "current_start_frame": last_frame_url,
        "index": idx + 1
    }


# GRAPH ORCHESTRATION

def after_script_writer(state: VideoState):
    """ Route to END if user aborted at HITL step, otherwise start generating."""
    if not state["scenes"]:
        return END
    return "generate"


def should_continue(state: VideoState):
    if state["index"] < len(state["scenes"]):
        return "generate"
    return END


workflow = StateGraph(VideoState)
workflow.add_node("script_writer", script_writer_node)
workflow.add_node("generate", generation_node)
workflow.set_entry_point("script_writer")

# Replaced unconditional edge with a conditional one to handle abort.
workflow.add_conditional_edges("script_writer", after_script_writer)
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

    raw_session_name = input("\nEnter a unique session name for Supabase (e.g., ad_run_01):\n> ").strip()
    # Sanitize the session name before using it in storage paths.
    session_name = sanitize_session_name(raw_session_name)
    if session_name != raw_session_name:
        print(f" Session name sanitized to: '{session_name}'")

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

    # Use session_name as thread_id so concurrent runs don't share checkpoint state.
    config = {"configurable": {"thread_id": session_name}}

    print("\n Starting Live Agent Pipeline...")
    for event in app.stream(initial_state, config=config):
        pass

    final_state = app.get_state(config)
    video_urls = final_state.values.get("video_clips", [])

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
