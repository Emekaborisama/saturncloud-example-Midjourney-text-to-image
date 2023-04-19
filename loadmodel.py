from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import accelerate
from PIL import Image
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
directory = "generatedimages"
if not os.path.exists(directory):
    os.makedirs(directory)


# model_id = "Joeythemonster/anything-midjourney-v-4-1"
# dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# pipe.enable_attention_slicing()



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


    def generate_image(self,prompt):
        image = self.pipe(prompt).images[0]
        image = np.asarray(image)
        im = Image.fromarray(image)
        im.save(f"{directory}/{prompt}.jpeg")
        # im.save(".generatedimages/"+str(prompt)+".jpeg")
        return str(prompt)
    
    
    
# model_name = os.getenv("MODEL_NAME")
# device = os.getenv("DEVICE")
# model = Model_generate(model_name=model_name, device=device)

# model.generate_image(prompt="nasa moon")
