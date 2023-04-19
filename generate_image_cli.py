import click
# from loadmodel.Model_generate import Model_generate
import os
import args 

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
        """initialize model object """
        self.device=device
        self.model_id = model_name
        dpm = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        
    def load_model(self):
        """load model """
        return self.pipe


    def generate_image(self,prompt):
        """generate image """
        image = self.pipe(prompt).images[0]
        image = np.asarray(image)
        im = Image.fromarray(image)
        im.save(f"{directory}/{prompt}.jpeg")
        # im.save(".generatedimages/"+str(prompt)+".jpeg")
        return str(prompt)
    
    







model_name = os.getenv("MODEL_NAME")
device = os.getenv("DEVICE")
model = Model_generate(model_name=model_name, device=device)
@click.command()
# @click.option('--model_name', type=str, help='Name of the pre-trained model to use', required=True)
# @click.option('--device', type=str, help='Device to use for running the model (e.g. cuda or cpu)', required=True)
@click.option('--prompt', type=str, help='Prompt to generate image from', default='nasa moon')
def generate_image_cli(prompt):
    """Generate image from a prompt"""
    # create Model_generate object
    # generate image and save to file
    image_path = model.generate_image(prompt=prompt)

    click.echo(f"Image saved to {image_path}")

if __name__ == '__main__':
    generate_image_cli()
