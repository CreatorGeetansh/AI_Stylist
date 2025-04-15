# app/main.py

import logging
import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from dotenv import load_dotenv
import google.generativeai as genai # Import Gemini library

# --- Configuration & Setup ---
load_dotenv() # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
# Florence-2
MODEL_ID = os.getenv("MODEL_ID", "microsoft/florence-2-base")
PROCESSOR_ID = os.getenv("PROCESSOR_ID", "microsoft/florence-2-large") # Keeping large processor as per original
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() and DEVICE != "mps" else torch.float32 # MPS doesn't fully support float16 well with all ops

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Use "gemini-1.5-flash" or "gemini-1.5-pro" if available/preferred

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI recommendations will be disabled.")
    # raise ValueError("GEMINI_API_KEY environment variable not set.") # Alternatively, make it mandatory

# Application State
app_state = {
    "florence_model": None,
    "florence_processor": None,
    "gemini_model": None,
    "is_ready": False,
    "gemini_available": False,
}

# --- Model Loading & Pre-heating ---

def load_florence_model():
    """Loads the Florence-2 model and processor."""
    logger.info(f"Starting Florence-2 model loading: Model={MODEL_ID}, Processor={PROCESSOR_ID} on {DEVICE} ({DTYPE})")
    if MODEL_ID != PROCESSOR_ID:
        logger.warning(f"Using mismatched Florence model ({MODEL_ID}) and processor ({PROCESSOR_ID}). Recommended to use matching versions.")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True
        ).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(
            PROCESSOR_ID,
            trust_remote_code=True
        )
        logger.info("Florence-2 model and processor loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading Florence-2 model/processor: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Florence-2 model '{MODEL_ID}' or processor '{PROCESSOR_ID}'.") from e

def preheat_florence_model(model, processor):
    """Runs a dummy inference to pre-heat the Florence-2 model."""
    logger.info("Starting Florence-2 model pre-heating...")
    try:
        dummy_image = Image.new('RGB', (64, 64), color = 'black')
        task_prompt = "<CAPTION>" # Basic caption for preheating
        inputs = processor(text=task_prompt, images=dummy_image, return_tensors="pt").to(DEVICE, DTYPE)
        with torch.no_grad(): # Ensure no gradients are computed
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=50,
                num_beams=3,
                do_sample=False # More deterministic for preheating
            )
        processor.batch_decode(generated_ids, skip_special_tokens=False)
        logger.info("Florence-2 model pre-heating completed.")
    except Exception as e:
        logger.error(f"Error during Florence-2 pre-heating: {e}", exc_info=True)
        # Don't raise here, allow app to start but log the error

def configure_gemini():
    """Configures and returns the Gemini client."""
    if not GEMINI_API_KEY:
        logger.warning("Gemini API Key not configured. Skipping Gemini model setup.")
        return None
    try:
        logger.info(f"Configuring Gemini client for model: {GEMINI_MODEL_NAME}")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # Optional: Add a simple test call here if needed
        # model.generate_content("Test connection.")
        logger.info("Gemini client configured successfully.")
        return model
    except Exception as e:
        logger.error(f"Error configuring Gemini client: {e}", exc_info=True)
        return None # Allow app to run without Gemini if configuration fails


# --- FastAPI Lifespan Management (Startup/Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    model_loaded = False
    try:
        app_state["florence_model"], app_state["florence_processor"] = load_florence_model()
        preheat_florence_model(app_state["florence_model"], app_state["florence_processor"])
        model_loaded = True
    except Exception as e:
        logger.error(f"Florence-2 initialization failed: {e}", exc_info=True)
        # Allow startup without Florence-2 if needed, but log critical failure
        # Or raise the exception if Florence-2 is essential

    try:
        app_state["gemini_model"] = configure_gemini()
        if app_state["gemini_model"]:
            app_state["gemini_available"] = True
            logger.info("Gemini integration is active.")
        else:
             logger.warning("Gemini integration is inactive due to missing key or configuration error.")
    except Exception as e:
        logger.error(f"Gemini initialization failed: {e}", exc_info=True)
        # Allow startup without Gemini


    if model_loaded: # Only set ready if the core model (Florence) loaded
        app_state["is_ready"] = True
        logger.info("Application is ready (Florence-2 loaded).")
    else:
         logger.error("Application startup failed: Core model (Florence-2) did not load.")

    yield # Application runs here

    logger.info("Application shutdown...")
    app_state["florence_model"] = None
    app_state["florence_processor"] = None
    app_state["gemini_model"] = None
    app_state["is_ready"] = False
    app_state["gemini_available"] = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Resources cleaned up.")

# --- FastAPI Application Instance ---

app = FastAPI(
    title="AI Fashion Assistant",
    description="API using Florence-2 for image captioning and Gemini for outfit recommendations.",
    version="1.1.0", # Updated version
    lifespan=lifespan
)

# --- Static Files & Templates ---
# Ensure the paths are correct relative to where main.py is located
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


# --- Helper Functions ---

def download_image(url: str) -> Image.Image:
    """Downloads an image from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        logger.info(f"Successfully downloaded image from URL: {url}")
        return image
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download image from URL: {e}")
    except Exception as e:
        logger.error(f"Error processing image from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process image from URL: {e}")

def process_uploaded_image(file: UploadFile) -> Image.Image:
    """Reads an uploaded image file."""
    try:
        contents = file.file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"Successfully processed uploaded image: {file.filename}")
        return image
    except Exception as e:
        logger.error(f"Error processing uploaded file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not read or process uploaded image file: {e}")

def run_florence_inference(model, processor, image: Image.Image, task_prompt: str = "<CAPTION>") -> str:
    """Runs the Florence-2 model inference."""
    if not app_state["is_ready"] or model is None or processor is None:
        logger.error("Florence-2 inference requested but model is not ready.")
        raise HTTPException(status_code=503, detail="Captioning model is not available.")

    logger.info(f"Running Florence-2 inference with task prompt: {task_prompt}")

    try:
        # Ensure image is not too large (optional, but good practice)
        # max_size = (1024, 1024)
        # image.thumbnail(max_size, Image.Resampling.LANCZOS)

        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)

        with torch.no_grad(): # Disable gradient calculation for inference
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(DEVICE, DTYPE), # Ensure pixel values are on correct device/dtype
                max_new_tokens=1024, # As per original
                num_beams=3,
                do_sample=False # Usually better for captioning
            )

        # Decode generated IDs
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logger.info(f"Raw Florence-2 generated text: {generated_text}")

        # Post-process using the processor
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        logger.info(f"Parsed Florence-2 answer: {parsed_answer}")

        # Extract the specific result based on the task prompt
        caption = parsed_answer.get(task_prompt)

        if caption is None:
            # Fallback if the expected key isn't found
            caption = parsed_answer.get('caption', "Could not parse caption from model output.")
            logger.warning(f"Could not find key '{task_prompt}' in parsed answer. Using fallback. Full answer: {parsed_answer}")


        # Ensure result is a string
        if isinstance(caption, list):
            caption = caption[0] if caption else "Caption list was empty."
        elif not isinstance(caption, str):
             caption = str(caption) # Convert if it's some other type

        # Simple cleanup
        caption = caption.strip()

        logger.info(f"Florence-2 Inference successful. Final caption: {caption}")
        return caption

    except Exception as e:
        logger.error(f"Error during Florence-2 inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Florence-2 model inference failed: {e}")

def get_gemini_recommendations(caption: str) -> str:
    """Gets outfit recommendations from Gemini based on the caption."""
    if not app_state["gemini_available"] or not app_state["gemini_model"]:
        logger.warning("Gemini recommendations requested but Gemini is not available.")
        return "AI stylist recommendations are currently unavailable."

    gemini_model = app_state["gemini_model"]
    logger.info("Requesting recommendations from Gemini...")

    # --- Improved Prompt ---
    prompt = f"""
    You are a WORLD CLASS AI fashion stylist.
    A user has uploaded an image, and it has been described as: "{caption}"

    Based *only* on this description, please provide stylish and practical outfit recommendations. Suggest complementary items like tops, bottoms, shoes, outerwear, and accessories to create a complete, cohesive look.

    Focus on:
    - Color coordination and harmony.
    - Style matching (e.g., casual, formal, bohemian).
    - Suggesting specific types of items (e.g., "white sneakers", "a tailored blazer", "delicate gold necklace").
    - Briefly explaining *why* the suggested items work well together.

    Present the recommendations in a clear, easy-to-read format (e.g., using bullet points or short paragraphs for different parts of the outfit). Keep the tone positive and inspiring.
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            # Optional: Add safety settings if needed
            # safety_settings=[...]
            # Optional: Generation config
            # generation_config=genai.types.GenerationConfig(...)
            )

        # Check for blocked content or empty response
        if not response.parts:
             if response.prompt_feedback.block_reason:
                 logger.warning(f"Gemini request blocked. Reason: {response.prompt_feedback.block_reason}")
                 return f"Could not generate recommendations due to content policy: {response.prompt_feedback.block_reason}"
             else:
                 logger.warning("Gemini returned an empty response.")
                 return "AI stylist couldn't generate recommendations for this item."

        recommendations = response.text.strip()
        logger.info("Successfully received recommendations from Gemini.")
        # logger.debug(f"Gemini Raw Response Text:\n{recommendations}") # Uncomment for debugging
        return recommendations

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}", exc_info=True)
        return f"Error getting recommendations from AI stylist: {e}"


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    logger.info("Serving index.html")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict_image(
    request: Request,
    image_url: str = Form(None),
    file: UploadFile = File(None, alias="image_file"), # Use alias matching JS FormData key
):
    """
    Receives image (URL or upload), generates caption (Florence-2),
    and gets outfit recommendations (Gemini).
    Returns both caption and recommendations.
    """
    logger.info("Received request on /predict endpoint.")

    if not image_url and not file:
        logger.warning("Prediction request received without URL or file.")
        raise HTTPException(status_code=400, detail="Please provide either an 'image_url' or upload an 'image_file'.")

    if image_url and file:
        logger.warning("Prediction request received with both URL and file. Using file.")
        # Prioritize file if both are sent, matching original logic

    image: Image.Image | None = None
    input_source = "file" if file else "url"

    try:
        if file:
            logger.info(f"Processing uploaded file: {file.filename}, content type: {file.content_type}")
            if not file.content_type or file.content_type.split('/')[0] != 'image':
                 raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image (JPEG, PNG, GIF, etc.).")
            image = process_uploaded_image(file)
        elif image_url:
            logger.info(f"Processing image URL: {image_url}")
            image = download_image(image_url)

    except HTTPException as http_exc:
        # Log the specific HTTP exception detail before re-raising
        logger.error(f"HTTPException during image processing from {input_source}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error processing input image from {input_source}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process input image: {e}")

    if image is None:
         # This case should ideally be caught by the specific exceptions above
         logger.error("Image object is None after processing inputs, source: {input_source}. This should not happen.")
         raise HTTPException(status_code=500, detail="Internal error: Failed to obtain image object.")

    # --- Step 1: Get Caption from Florence-2 ---
    # Using <CAPTION> as it's more standard for general description than <DETAILED_CAPTION>
    # which might be geared towards object detection bounding boxes in some fine-tunings.
    # Stick with the processor's default if unsure, or try <MORE_DETAILED_CAPTION> if needed.
    caption_task_prompt = "<DETAILED_CAPTION>"
    caption = ""
    try:
        caption = run_florence_inference(
            app_state["florence_model"],
            app_state["florence_processor"],
            image,
            task_prompt=caption_task_prompt
        )
        logger.info(f"Generated caption: {caption}")
    except HTTPException as http_exc:
        # If captioning fails, maybe we still proceed? Or return error?
        # For now, let's return the error immediately.
        logger.error(f"Captioning failed: {http_exc.detail}")
        raise http_exc # Re-raise the specific HTTP error from inference
    except Exception as e:
        logger.error(f"Unexpected error during captioning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during image captioning: {e}")


    # --- Step 2: Get Recommendations from Gemini ---
    recommendations = "Recommendations unavailable." # Default
    if caption and app_state["gemini_available"]:
        try:
            recommendations = get_gemini_recommendations(caption)
            logger.info("Received recommendations.")
        except Exception as e:
            # Log the error but don't fail the whole request, just return default message
            logger.error(f"Error getting Gemini recommendations: {e}", exc_info=True)
            recommendations = "Error retrieving recommendations from AI stylist."
    elif not app_state["gemini_available"]:
         recommendations = "AI Stylist (Gemini) is not configured or available."
    elif not caption:
         recommendations = "Cannot generate recommendations without a valid caption."


    # --- Step 3: Return Results ---
    return JSONResponse(content={
        "caption": caption,
        "recommendations": recommendations
    })

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    florence_ready = app_state.get("is_ready", False)
    gemini_ready = app_state.get("gemini_available", False)
    status = "ok" if florence_ready else "service_unavailable" # Base status on core model

    return JSONResponse(content={
        "status": status,
        "models": {
             "florence2": {"ready": florence_ready},
             "gemini": {"ready": gemini_ready, "configured": bool(GEMINI_API_KEY) }
        }
    })

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly for development...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", # Bind to 0.0.0.0 to be accessible on network
        port=int(os.getenv("PORT", 8000)), # Allow port override via env var
        reload=True, # Enable auto-reload for development
        workers=1    # Use 1 worker for simplicity with reload
    )