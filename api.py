# flask api

from flask import Flask, request, send_from_directory,send_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import numpy as np
import accelerate
from PIL import Image
import os
from flask_cors import CORS
import io
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_name = os.getenv("MODEL_NAME")
device = os.getenv("DEVICE")



app = Flask(__name__)
CORS(app)



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
        filename = f"{prompt}.png"
        filepath = os.path.join(directory, filename)
        # img_data = io.BytesIO()
        # image.save(img_data, "PNG")
        # img_data.seek(0)
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
    


model = Model_generate(model_name=model_name, device=device)



@app.route("/home")
def index():
  return("welcome to app")




@app.route('/generate_image', methods=['GET','POST'])
def download():
    prompt = request.args.get('prompt')
    if prompt is None:
        return "Please provide a prompt"
    # model_name = os.getenv("MODEL_NAME")
    # device = os.getenv("DEVICE")
    # model = Model_generate(model_name=model_name, device=device)
    filepath = model.generate_image(prompt)
    return send_file(filepath, mimetype='image/png')
    # return send_from_directory(directory, filepath, as_attachment=True)

if __name__ == '__main__':
    app.run( host= '0.0.0.0', port = 8080, debug=True)
