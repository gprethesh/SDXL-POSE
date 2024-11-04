from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.controlnet = ControlNetModel.from_pretrained(
            "xinsir/controlnet-union-sdxl-1.0",
            torch_dtype=torch.float16
        )
        
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        image: Path = Input(description="Input image for conditioning"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        num_inference_steps: int = Input(default=30),
        guidance_scale: float = Input(default=7.5),
        seed: int = Input(default=-1),
    ) -> Path:
        if seed != -1:
            torch.manual_seed(seed)
        
        control_image = load_image(str(image))
        
        output = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        output_path = "/tmp/output.png"
        output.save(output_path)
        return Path(output_path)
