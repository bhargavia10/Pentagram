import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

# Function to download the model during the build process
def download_model():
    from huggingface_hub import login
    from diffusers import AutoPipelineForText2Image
    import torch
    
    # Get the HuggingFace API Key from .env
    huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

    # Verify if API key is available
    if not huggingface_api_key:
        raise ValueError("HUGGINGFACE_API_KEY is not set in environment variables.")
    
    # Login to HuggingFace
    login(huggingface_api_key)
    
    # Download the model using the AutoPipelineForText2Image class
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_auth_token="asdfghjkl",
    )
    del pipeline  # Free memory after download

# Build the image and include necessary dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests")
    .run_function(download_model)
)

# Create a Modal app instance with the image
app = modal.App(
    "sd-demo",
    image=image,
)

# Define a class-based Modal app with secrets and GPU
@app.cls(
    image=image,
    gpu="A10G",
    container_idle_timeout=300,
    secrets=[
        modal.Secret.from_name("API_KEY"),
        modal.Secret.from_name("HUGGINGFACE_API_KEY"),
    ],
)

class Model:

    # This function loads the model weights when the app is started
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        # Load model with Hugging Face token
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_auth_token="asdfghjkl",
        )
        # Move the model to the GPU for faster processing
        self.pipe.to("cuda")
        # Retrieve the API key from the environment variables
        self.API_KEY = os.environ["API_KEY"]

    # Web endpoint to handle image generation requests
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):
        try:
            # Get the API key from the request headers to verify authorization
            api_key = request.headers.get("X-API-Key")
            if api_key != self.API_KEY:
                raise HTTPException(status_code=401, detail="Unauthorized")

            # Generate an image based on the prompt
            image = self.pipe(prompt, num_inference_steps=75, guidance_scale=8.5, height=500, width=500).images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            # Return the image as a response
            return Response(content=buffer.getvalue(), media_type="image/jpeg")
        
        # If an error occurs, raise an HTTP Exception with an error message
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Function to keep the app "warm" by periodically making health and generation requests
@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("API_KEY")],
)
def keep_warm():
    health_url = "https://bhargavialluri12--sd-demo-model-health.modal.run"
    generate_url = "https://bhargavialluri12--sd-demo-model-generate.modal.run"

    # Make a GET request to the health check URL to monitor the app's status
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")

    # Make a GET request to the image generation endpoint
    headers = {"X-API-Key": os.environ["API_KEY"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")
