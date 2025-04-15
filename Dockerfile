# Dockerfile (Recommended for Windows Dev -> Linux Deployment)

# Use a Python slim image based on Debian Bookworm (stable, good compatibility)
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install essential system dependencies using apt-get (Debian package manager)
# - libjpeg62-turbo: Common runtime dependency for Pillow JPEG handling
# - git & ca-certificates: Often needed by pip for certain installs
# Add any other *runtime* libraries needed by your Python packages here
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    git \
    ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- IMPORTANT for Windows Users ---
# Ensure your requirements.txt file uses LF line endings, not CRLF.
# Use a .gitattributes file or configure your editor.
# ---

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code from your local './app' directory
# into the container's '/app' directory (which is the WORKDIR)
# Ensure files copied (like main.py) also use LF line endings.
COPY ./app /app

# Expose the port the FastAPI app will run on inside the container
EXPOSE 8000

# Define the command to run the application using uvicorn
# It looks for 'main:app' in the WORKDIR (/app)
# Uses 0.0.0.0 to be accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

