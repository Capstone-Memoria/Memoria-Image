import os
import io
import base64
import requests
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image

# --- Configuration ---
MODEL_DIR = Path("models")
MODEL_NAME = "model.safetensors"
MODEL_PATH = MODEL_DIR / MODEL_NAME
MODEL_URL = "https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_V2-SFW.safetensors?download=true"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Global Variables ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
pipe = None

# --- Helper Functions ---
def download_model():
    """Downloads the model if it doesn't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Model not found. Downloading from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 # 1 Kibibyte

            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    done = int(50 * downloaded / total_size)
                    print(f"\rDownloading: [{'=' * done}{' ' * (50-done)}] {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end='')
            print("\nDownload complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            # Clean up incomplete download
            if MODEL_PATH.exists():
                os.remove(MODEL_PATH)
            raise RuntimeError(f"Failed to download model from {MODEL_URL}") from e
    else:
        print("Model found.")

def load_pipeline():
    """Loads the Stable Diffusion XL pipeline."""
    global pipe
    print(f"Loading pipeline on {DEVICE} with dtype {DTYPE}...")
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            MODEL_PATH,
            torch_dtype=DTYPE,
            variant="fp16" if DTYPE == torch.float16 else None, # Use variant="fp16" for float16
            use_safetensors=True
        )
        pipe.to(DEVICE)
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        raise RuntimeError("Failed to load the SDXL pipeline.") from e

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Check/download model and load pipeline on startup."""
    download_model()
    load_pipeline()

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the HTML test page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_image(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: str = Form("")
):
    """Generates an image based on positive and negative prompts."""
    if pipe is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Pipeline not loaded. Check server logs."}
        )

    print(f"Generating image with prompt: '{prompt}'")
    try:
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20, # Adjust as needed
            guidance_scale=7.0,     # Adjust as needed
            height=1024,
            width=1024
        ).images[0]

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={"image": img_str})

    except Exception as e:
        print(f"Error during image generation: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Image generation failed: {e}"}
        )

# --- Static Files (Optional) ---
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Run with Uvicorn (for development) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# Note: Running with `uvicorn main:app --reload` is recommended for development.
