from diffusers import DiffusionPipeline
import torch

class Interfaces():
    def __init__(self):
        pass
    
    def get_sd_pipe(self):
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.bfloat16, 
                # device_map="cuda"
            )
            # SD in CPU
            pipe = pipe.to("cpu")
            pipe.vae.enable_tiling()
            print("SD pipe started succesfully.")
            return pipe
            

if __name__ == "__main__":
    pass    