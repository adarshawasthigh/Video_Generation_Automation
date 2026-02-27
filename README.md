# 🎬 Autonomous Video Generation Agent

A stateful, LangGraph-powered backend agent that orchestrates LLMs, cloud GPU rendering, and local video manipulation to dynamically generate and stitch continuous video narratives.

Unlike standard AI API wrappers that generate isolated 5-second clips, this architecture utilizes a cyclic state machine to extract anchor frames, persist state to a PostgreSQL-backed storage bucket, and maintain mathematical visual continuity across an infinite sequence of generations.

## 🧠 System Architecture

This pipeline integrates deterministic orchestration with probabilistic AI models:

* **The Brain (Gemini 2.5 Pro):** Ingests raw user prompts or scripts, structuring them into a strict Pydantic JSON array of 5-second sequential scenes.
* **The State Machine (LangGraph):** Manages the chronological loop, passing the visual context of `Scene N` directly into the generation prompt for `Scene N+1`.
* **The Inference Engine (Fal.ai Cloud GPUs):** Executes the heavy Wan-2.1 image-to-video diffusion models via serverless API endpoints.
* **The Continuity Extractor (OpenCV):** Downloads generated segments, mathematically isolates the final frame, and securely injects it as the starting anchor for the next chronological loop.
* **The Database (Supabase):** Handles state persistence, securely upserting intermediate frames and MP4s via the `service_role` key to bypass Row-Level Security (RLS).
* **The Stitching Engine (FFmpeg):** An automated, lossless demuxer (`-c copy`) that merges the isolated database clips into a single, seamless master video without re-encoding pixels.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain
* **LLM:** Google Gemini 2.5 Pro
* **Video Generation:** Fal.ai (Wan 2.1)
* **Computer Vision & Media:** OpenCV, FFmpeg
* **Database & Storage:** Supabase (PostgreSQL)

## ⚙️ Prerequisites

* Python 3.10+
* **System Requirement:** FFmpeg MUST be installed on your machine's system path for the stitching module to work.
  * **Mac:** `brew install ffmpeg`
  * **Windows:** `winget install ffmpeg`
  * **Linux:** `sudo apt install ffmpeg`

## 🚀 Quickstart

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/Video_Generation_Automation.git](https://github.com/yourusername/Video_Generation_Automation.git)
cd Video_Generation_Automation

pip install -r requirements.txt

GEMINI_API_KEY=your_google_key
FAL_KEY=your_fal_ai_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key

python app.py
