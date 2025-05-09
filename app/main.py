# app/main.py
import logging
import io
import os
import re # Import regular expressions for parsing
import urllib.parse # Import for URL encoding
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions

# --- Configuration & Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt/" # Define Pollinations URL base

if not GEMINI_API_KEY:
    logger.error("FATAL: GEMINI_API_KEY not found. Application cannot start.")
    raise ValueError("GEMINI_API_KEY environment variable not set.")

app_state = {
    "gemini_model": None,
    "is_ready": False,
}

# --- Model Configuration & Lifespan ---
def configure_gemini():
    if not GEMINI_API_KEY:
        logger.error("Attempted to configure Gemini without an API key.")
        return None
    try:
        logger.info(f"Configuring Gemini client for model: {GEMINI_MODEL_NAME}")
        genai.configure(api_key=GEMINI_API_KEY)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            safety_settings=safety_settings
        )
        logger.info("Gemini client configured successfully.")
        return model
    except Exception as e:
        logger.error(f"Error configuring Gemini client: {e}", exc_info=True)
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    gemini_configured = False
    try:
        app_state["gemini_model"] = configure_gemini()
        if app_state["gemini_model"]:
            gemini_configured = True
            app_state["is_ready"] = True
            logger.info("Gemini integration is active. Application is ready.")
        else:
             logger.error("Gemini configuration failed. Application may not function correctly.")
             app_state["is_ready"] = False
    except Exception as e:
        logger.error(f"Critical error during Gemini initialization: {e}", exc_info=True)
        app_state["is_ready"] = False

    if not gemini_configured:
        logger.critical("Application startup failed: Core model (Gemini) could not be configured.")

    yield
    logger.info("Application shutdown...")
    app_state["gemini_model"] = None
    app_state["is_ready"] = False
    logger.info("Resources cleaned up.")

# --- FastAPI Application Instance ---
app = FastAPI(
    title="AI Fashion Assistant",
    description="API using Gemini for image analysis and outfit recommendations with visualizations.",
    version="2.1.0", # Updated version
    lifespan=lifespan
)

# --- Static Files & Templates ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# --- Helper Functions ---
def download_image(url: str) -> Image.Image:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 ...'} # Keep your headers
        response = requests.get(url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        logger.info(f"Successfully downloaded image from URL: {url[:50]}...") # Log shorter URL
        return image
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {url[:50]}...: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download image from URL: {e}")
    except Exception as e:
        logger.error(f"Error processing image from {url[:50]}...: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process image from URL: {e}")

def process_uploaded_image(file: UploadFile) -> Image.Image:
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

def get_gemini_image_description(image: Image.Image) -> str:
    if not app_state["is_ready"] or not app_state["gemini_model"]:
        logger.error("Gemini description requested but model is not ready.")
        raise HTTPException(status_code=503, detail="Image analysis model is not available.")

    gemini_model = app_state["gemini_model"]
    logger.info("Requesting image description from Gemini...")
    prompt_parts = [
        "Describe the clothing items and overall style shown in this image in detail. Focus on colors, types of garments, patterns, hairstyle and accessories. Make it brief but informative.",
        image
    ]
    try:
        response = gemini_model.generate_content(prompt_parts)
        if not response.parts:
             if response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 logger.warning(f"Gemini image description blocked. Reason: {block_reason}")
                 raise HTTPException(status_code=400, detail=f"Image analysis failed due to content policy: {block_reason}")
             else:
                 logger.warning("Gemini returned an empty response for image description.")
                 raise HTTPException(status_code=500, detail="Image analysis failed: No description generated.")
        description = response.text.strip()
        logger.info("Successfully generated image description from Gemini.")
        return description
    # Keep existing exception handling
    except google.api_core.exceptions.PermissionDenied:
        logger.error("Gemini API key is invalid or lacks permissions.", exc_info=True)
        raise HTTPException(status_code=500, detail="Image analysis service permission error.")
    except google.api_core.exceptions.ResourceExhausted:
        logger.error("Gemini API quota exceeded.", exc_info=True)
        raise HTTPException(status_code=429, detail="Image analysis service quota limit reached. Please try again later.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error calling Gemini API for image description: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating image description: {e}")


def get_gemini_recommendations(description: str) -> str:
    """
    Gets **multiple** outfit recommendations from Gemini based on the image description.
    Attempts to return a string containing numbered or clearly separated options.
    """
    if not app_state["is_ready"] or not app_state["gemini_model"]:
        logger.warning("Gemini recommendations requested but Gemini is not available.")
        return "AI stylist recommendations are currently unavailable (model not ready)."

    gemini_model = app_state["gemini_model"]
    logger.info("Requesting 3 outfit recommendations from Gemini...")

    # --- Updated Prompt ---
    prompt = f"""
    You are a WORLD CLASS AI fashion stylist.
    An image was analyzed, and the key clothing items and style were described as: "{description}"

    Based *only* on this description, please provide **exactly 3 RELATABLE outfit recommendations**.
    Each recommendation should suggest complementary items (tops, bottoms, shoes, outerwear, accessories) and a suitable hairstyle to create a complete, cohesive look.

    For each recommendation:
    - Focus on practicality, wearability, color coordination, and style matching.
    - Suggest specific types of items (e.g., "black Chelsea boots", "a cream-colored chunky knit sweater").
    - Briefly explain *why* the suggested items work well together.
    - Keep the tone positive and inspiring.

    **Format the output CLEARLY with each option numbered like this:**

    Option 1:
    [Detailed description of the first outfit recommendation, including hairstyle and reasoning.]

    Option 2:
    [Detailed description of the second outfit recommendation, including hairstyle and reasoning.]

    Option 3:
    [Detailed description of the third outfit recommendation, including hairstyle and reasoning.]

    Ensure each option starts exactly with "Option X:" on a new line. Avoid any preamble before "Option 1:".
    """

    try:
        response = gemini_model.generate_content(prompt)
        if not response.parts:
             if response.prompt_feedback.block_reason:
                 logger.warning(f"Gemini recommendations blocked. Reason: {response.prompt_feedback.block_reason}")
                 return f"Could not generate recommendations due to content policy: {response.prompt_feedback.block_reason}"
             else:
                 logger.warning("Gemini returned an empty response for recommendations.")
                 return "AI stylist couldn't generate recommendations based on the description."

        recommendations_text = response.text.strip()
        logger.info("Successfully received multi-recommendation text from Gemini.")
        # logger.debug(f"Gemini Raw Recommendation Response:\n{recommendations_text}")
        return recommendations_text

    # Keep existing exception handling
    except google.api_core.exceptions.PermissionDenied:
         logger.error("Gemini API key is invalid or lacks permissions for recommendations.", exc_info=True)
         return "Error getting recommendations: Service permission error."
    except google.api_core.exceptions.ResourceExhausted:
        logger.error("Gemini API quota exceeded during recommendations.", exc_info=True)
        return "Recommendation service quota limit reached. Please try again later."
    except Exception as e:
        logger.error(f"Error calling Gemini API for recommendations: {e}", exc_info=True)
        return f"Error getting recommendations from AI stylist: {e}"


def parse_recommendations(recommendations_text: str) -> list[dict[str, str]]:
    """
    Parses the Gemini multi-recommendation text into a list of dictionaries,
    each containing the text and a generated Pollinations image URL.
    """
    recommendations_list = []
    # Regex to find "Option X:" followed by the description until the next "Option X:" or end of string
    # DOTALL allows '.' to match newlines. Group 1 captures the number, Group 2 captures the text.
    matches = re.findall(r"Option (\d+):\s*(.*?)(?=\n\n?Option \d+:|\Z)", recommendations_text, re.DOTALL | re.IGNORECASE)

    if matches:
        logger.info(f"Parsed {len(matches)} recommendations using regex.")
        for i, match in enumerate(matches):
            option_number, option_text = match
            clean_text = option_text.strip()
            if clean_text:
                 # Create a simplified prompt for the image generator
                 # Focus on key items, colors, and overall style described in the option
                 # This requires some heuristics or could even be another LLM call (but keeping it simple here)
                 image_prompt_text = clean_text[:350] # Limit prompt length
                 encoded_prompt = urllib.parse.quote(image_prompt_text)
                 image_url = f"{POLLINATIONS_BASE_URL}{encoded_prompt}"

                 recommendations_list.append({
                     "option": f"Option {option_number}", # Keep the option number/title
                     "text": clean_text,
                     "image_url": image_url
                 })
            else:
                logger.warning(f"Option {option_number} text was empty after stripping.")

    # Fallback or alternative parsing if regex fails or format is different
    elif "Option 1:" in recommendations_text or "1." in recommendations_text: # Basic check
         logger.warning("Regex parsing failed, attempting basic split parsing.")
         # Simple split logic (less robust)
         parts = re.split(r'\n\s*Option \d+:', recommendations_text, flags=re.IGNORECASE)
         parts = [p.strip() for p in parts if p.strip()] # Remove empty parts
         if len(parts) > 0 and len(parts[0]) < 50: # Remove potential intro text if first part is short
             parts = parts[1:]

         for i, part_text in enumerate(parts):
             if len(recommendations_list) >= 3: break # Limit to 3 max
             image_prompt_text = f"Fashion sketch style, rough lines: A person wearing {part_text[:250]}"
             encoded_prompt = urllib.parse.quote(image_prompt_text)
             image_url = f"{POLLINATIONS_BASE_URL}{encoded_prompt}"
             recommendations_list.append({
                 "option": f"Suggestion {i+1}", # Generic title if numbering is lost
                 "text": part_text,
                 "image_url": image_url
             })

    else:
         # If no structure is found, treat the whole text as one recommendation (fallback)
         logger.warning("Could not parse multiple options. Treating response as single recommendation.")
         if recommendations_text and not recommendations_text.lower().startswith("error") and not recommendations_text.lower().startswith("could not"):
             image_prompt_text = f"Fashion sketch style, rough lines: A person wearing {recommendations_text[:250]}"
             encoded_prompt = urllib.parse.quote(image_prompt_text)
             image_url = f"{POLLINATIONS_BASE_URL}{encoded_prompt}"
             recommendations_list.append({
                 "option": "Style Idea",
                 "text": recommendations_text,
                 "image_url": image_url
             })

    logger.info(f"Final parsed recommendations count: {len(recommendations_list)}")
    return recommendations_list


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Serving index.html")
    # Add favicon path to context if needed, or handle directly in HTML
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict_image(
    request: Request,
    image_url: str = Form(None),
    file: UploadFile = File(None, alias="image_file"),
):
    logger.info("Received request on /predict endpoint.")
    if not app_state["is_ready"]:
        logger.error("Prediction requested but the service is not ready.")
        raise HTTPException(status_code=503, detail="Service Unavailable: AI model not configured.")

    if not image_url and not file:
        logger.warning("Prediction request without URL or file.")
        raise HTTPException(status_code=400, detail="Please provide either an 'image_url' or upload an 'image_file'.")

    image: Image.Image | None = None
    input_source = "file" if file else "url"
    try:
        if file:
            logger.info(f"Processing uploaded file: {file.filename}")
            if not file.content_type or not file.content_type.startswith('image/'):
                 raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Please upload an image.")
            image = process_uploaded_image(file)
        elif image_url:
            logger.info(f"Processing image URL: {image_url[:50]}...")
            image = download_image(image_url)
    except HTTPException as http_exc:
        logger.error(f"HTTPException during image processing from {input_source}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error processing input image from {input_source}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process input image: {e}")

    if image is None:
         logger.error(f"Image object is None after processing inputs, source: {input_source}.")
         raise HTTPException(status_code=500, detail="Internal error: Failed to obtain image object.")

    # --- Step 1: Get Description ---
    description = ""
    try:
        description = get_gemini_image_description(image)
        logger.info(f"Generated description (caption): {description[:100]}...")
    except HTTPException as http_exc:
        logger.error(f"Image description generation failed: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during image description: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during image analysis: {e}")

    # --- Step 2: Get & Parse Recommendations ---
    recommendations_list = [] # Default to empty list
    raw_recommendations_text = "Recommendations unavailable."
    if description:
        try:
            raw_recommendations_text = get_gemini_recommendations(description)
            logger.info("Received raw recommendations text from Gemini.")
            # Parse the raw text into structured list with image URLs
            recommendations_list = parse_recommendations(raw_recommendations_text)
            logger.info(f"Successfully parsed {len(recommendations_list)} recommendations.")
        except Exception as e:
            logger.error(f"Error getting or parsing Gemini recommendations: {e}", exc_info=True)
            # If parsing fails, we might still want to return the raw text? Or an error state.
            # For now, let recommendations_list remain potentially empty or partially filled.
            # Let's add the raw text to the response for debugging/fallback display
            recommendations_list = [{"option": "Raw AI Response", "text": raw_recommendations_text, "image_url": None}]

    elif not description:
         # If no description, cannot get recommendations
         recommendations_list = [{"option": "Stylist Note", "text": "Cannot generate recommendations without a valid image description.", "image_url": None}]


    # --- Step 3: Return Results ---
    # Return the original description AND the list of recommendation objects
    return JSONResponse(content={
        "caption": description,
        "recommendations": recommendations_list # Send the list of dicts
    })


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    gemini_ready = app_state.get("is_ready", False)
    status = "ok" if gemini_ready else "service_unavailable"
    return JSONResponse(content={
        "status": status,
        "models": {
             "gemini": {"ready": gemini_ready, "configured": bool(GEMINI_API_KEY and app_state.get('gemini_model')) }
        }
    })

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly for development...")
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("RELOAD", True)),
        workers=int(os.getenv("WORKERS", 1))
    )