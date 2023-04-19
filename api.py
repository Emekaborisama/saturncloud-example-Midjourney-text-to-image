from flask import Flask, request, send_from_directory,send_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np
import accelerate
from PIL import Image
import os

app = Flask(__name__)

directory = "generatedimages"
if not os.path.exists(directory):
    os.makedirs(directory)

class Model_generate():
    def __init__(self, model_name,device):
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
        filename = f"{prompt}.jpeg"
        filepath = os.path.join(directory, filename)
        im.save(filepath)
        return filepath
    

model_name = os.getenv("MODEL_NAME")
device = os.getenv("DEVICE")
model = Model_generate(model_name=model_name, device=device)


@app.route("/home")
def index():
  return("welcome to app")




@app.route('/generate_image', methods=['POST'])
def download():
    prompt = request.args.get('prompt')
    if prompt is None:
        return "Please provide a prompt"
    # model_name = os.getenv("MODEL_NAME")
    # device = os.getenv("DEVICE")
    # model = Model_generate(model_name=model_name, device=device)
    filepath = model.generate_image(prompt)
    return send_file(filepath, as_attachment=True)
    # return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run( host= '0.0.0.0', port = 8000, debug=True)
