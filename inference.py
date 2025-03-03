import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
from tqdm import tqdm

def inference_diffusion(prompt, 
                        trained_model_dir="./unet_trained", 
                        output_dir="output",
                        image_nums = 1,
                        num_inference_steps=50):
    model_name = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the original pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.to(device)
    pipeline.safety_checker = None

    # Load your fine-tuned UNet model
    fine_tuned_unet = UNet2DConditionModel.from_pretrained(trained_model_dir)
    fine_tuned_unet.to(device)

    # Replace the original UNet in the pipeline with your fine-tuned model
    pipeline.unet = fine_tuned_unet

    # Generate and save images
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    pbar = tqdm(range(1,image_nums+1))
    for i in pbar:
        image = pipeline(prompt, num_inference_steps= num_inference_steps, guidance_scale = 1.0).images[0]
        image.save(os.path.join(output_dir, f"output_image_{i}.png"))
        pbar.set_description(f"Generated {i}/{image_nums} image(s)")
    print(f"Generated {i} images and saved in {output_dir}")

if __name__ == "__main__":
    inference_diffusion(image_nums = 100,
                        prompt="an endoscopy image with polyp", #characterized by redness, swelling, and irritation of the tissue, which could indicate an underlying condition or infection in the gastrointestinal tract.",
                        num_inference_steps= 100,
                        trained_model_dir="checkpoints/combine_fake_data_20/epoch_5",
                        output_dir="output/combine_fake_data_25")

