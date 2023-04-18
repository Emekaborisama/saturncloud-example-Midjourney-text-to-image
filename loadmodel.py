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





class Model_generate():
    def __init__(self, model_name,device):
        self.device=device
        self.model_id = model_name
        

        dpm = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float32)
        
        
    def load_model(self):
        self.pipe = self.pipe.to(self.device)
        return self.pipe


    def generate_image(self,prompt):
        image = self.pipe(prompt).images[0]
        image = np.asarray(image)
        im = Image.fromarray(image)
        im.save(f"{directory}/{prompt}.jpeg")
        # im.save(".generatedimages/"+str(prompt)+".jpeg")
        return str(prompt)
    
    
    

