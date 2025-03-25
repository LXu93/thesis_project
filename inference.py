import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os
import random
from tqdm import tqdm
from compel import Compel
from util import prompt_generator

def inference_diffusion(prompt, 
                        pathology = None,
                        trained_model_dir="./unet_trained", 
                        output_dir="output",
                        image_nums = 1,
                        num_inference_steps=50,
                        guidance_scale = 8,
                        random_gs = False,
                        gen = None):
    model_name = "runwayml/stable-diffusion-v1-5"
    #model_name = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the original pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.to(device)
    pipeline.safety_checker = None

    # Load fine-tuned UNet model
    if trained_model_dir.endswith('.pth'):
        pipeline.unet.load_state_dict(torch.load(trained_model_dir, weights_only=True))
    else:    
        fine_tuned_unet = UNet2DConditionModel.from_pretrained(trained_model_dir)
        fine_tuned_unet.to(device)
        # Replace the original UNet in the pipeline with your fine-tuned model
        pipeline.unet = fine_tuned_unet

    compel_proc = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
    # Generate and save images
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with torch.no_grad():
        pbar = tqdm(range(1,image_nums+1))
        for i in pbar:
            if pathology:
                prompt = prompt_generator(pathology, section="rondam")
            if random_gs:
                guidance_scale = random.randrange(random_gs[0],random_gs[1])
            prompt_embeds = compel_proc(prompt)
            image = pipeline(prompt_embeds=prompt_embeds, num_inference_steps= num_inference_steps, guidance_scale = guidance_scale, genertor=gen).images[0]
            image.save(os.path.join(output_dir, f"output_image_{i}.png"))
            pbar.set_description(f"Generated {i}/{image_nums} image(s)")
        print(f"Generated {i} images and saved in {output_dir}")

if __name__ == "__main__":
    inference_diffusion(image_nums = 100,
                        prompt="endoscopy image with polyp, with no bubbles",
                        pathology="polyp", 
                        num_inference_steps= 100,
                        guidance_scale = 10,
                        #random_gs=(6,16),
                        #gen=torch.manual_seed(42),
                        trained_model_dir="checkpoints/polyp_section/epoch_7",
                        output_dir="output/polyp_section/polyp/epoch_7")

