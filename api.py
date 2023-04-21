from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np
from PIL import Image
import os
import io

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

directory = "generatedimages"
if not os.path.exists(directory):
    os.makedirs(directory)

class Model_generate():
    def __init__(self, model_name, device):
        self.device=device
        self.model_id = model_name

        dpm = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

    def load_model(self):
        return self.pipe

    def generate_image(self, prompt):
        image = self.pipe(prompt).images[0]
        image = np.asarray(image)
        im = Image.fromarray(image)
        filename = f"{prompt}.png"
        filepath = os.path.join(directory, filename)
        im.save(filepath)
        return filepath

    def generate_image2(self, prompt):
        image = self.pipe(prompt).images[0]
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        filename = f"{prompt}.png"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes.read())
        return filepath

model_name = os.getenv("MODEL_NAME")
device = os.getenv("DEVICE")
model = Model_generate(model_name=model_name, device=device)

@app.get("/")
def index():
    return {"message": "Welcome to the app"}

@app.post('/generate_image')
def generate_image(prompt: str):
    if not prompt:
        return {"error": "Please provide a prompt"}
    filepath = model.generate_image2(prompt)
    return FileResponse(filepath, media_type='image/png')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
