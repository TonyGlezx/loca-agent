import torch
import uuid6 # From your previous uuid6 installation
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# 1. Use the 1.3B model and correct torch_dtype
pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", # Use the 1.3B version
    torch_dtype=torch.bfloat16
)

# 2. Aggressive VRAM saving techniques
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# 3. Generate frames (Video pipelines return 'frames', not 'images')
print("Generating video... this will take a while!")
output = pipe(prompt)
video_frames = output.frames[0]

# 4. Save using the correct video exporter
file_id = str(uuid6.uuid7())
path = f"{file_id}.mp4"

export_to_video(video_frames, path, fps=16)
print(f"Cool shit saved to: {path}")