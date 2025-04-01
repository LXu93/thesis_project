import torch
from diffusers import  StableDiffusionControlNetPipeline,ControlNetModel, UNet2DConditionModel
import os
import random
import numpy as np
from safetensors.torch import load_file
from PIL import Image, ImageOps
from tqdm import tqdm
from compel import Compel
from util import prompt_generator

def inference_controlNet(prompt, 
                        control_image,
                        pathology = None,
                        unet_dir="./unet_trained", 
                        controlnet_dir="./unet_trained", 
                        output_dir="output",
                        image_nums = 1,
                        num_inference_steps=50,
                        guidance_scale = 8,
                        random_gs = False,
                        gen = None):
    model_name = "runwayml/stable-diffusion-v1-5"
    #model_name = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    '''
    fine_tuned_unet = UNet2DConditionModel.from_pretrained(unet_dir)

    state_dict = load_file(controlnet_dir)
    conv_weight_1ch = state_dict["controlnet_cond_embedding.conv_in.weight"]  # [C_out, 1, 3, 3]

    # Expand to 3 channels
    conv_weight_3ch = conv_weight_1ch.repeat(1, 3, 1, 1) / 3.0  # [C_out, 3, 3, 3]
    state_dict["controlnet_cond_embedding.conv_in.weight"] = conv_weight_3ch
    '''
    controlnet = ControlNetModel.from_pretrained(controlnet_dir)
    #controlnet.load_state_dict(state_dict, strict=False)
    # Load the original pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_name, controlnet=controlnet)
    pipeline.to(device)
    pipeline.safety_checker = None
    
    # Load fine-tuned UNet model
    if unet_dir.endswith('.pth'):
        pipeline.unet.load_state_dict(torch.load(unet_dir, weights_only=True))
    else:    
        fine_tuned_unet = UNet2DConditionModel.from_pretrained(unet_dir)
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
                prompt = prompt_generator(pathology, section="unknown")
            if random_gs:
                guidance_scale = random.randrange(random_gs[0],random_gs[1])
            prompt_embeds = compel_proc(prompt)
            gen_image = pipeline(prompt_embeds=prompt_embeds, image=control_image,num_inference_steps= num_inference_steps, controlnet_conditioning_scale=2.0, guidance_scale = guidance_scale, genertor=gen).images[0]
            gen_image.save(os.path.join(output_dir, f"output_image_{i}.png"))
            pbar.set_description(f"Generated {i}/{image_nums} image(s)")
        print(f"Generated {i} images and saved in {output_dir}")

if __name__ == "__main__":
    from torchvision import transforms
    mask = Image.open(r"D:\Lu\data_thesis\galar_CAM\train\colon\non-polyp\20_frame_064780.PNG")
    #mask = Image.open(r"D:\Lu\data_thesis\galar_CAM\train\colon\polyp\11_frame_079530.PNG")
    #mask = Image.open(r"D:\Lu\data_thesis\galar_CAM\train\colon\non-polyp\17_frame_138490.PNG")
    #mask = mask.resize((320,320))
    #mask = Image.new('L', (512,512))
    mask = np.array(mask)
    mask = np.stack([mask]*3, axis=-1)  # Shape: H x W x 3
    mask = Image.fromarray(mask)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    #mask = transform(mask).unsqueeze(0)
    inference_controlNet(image_nums = 1,
                        prompt="endoscopy image in small intestine, with a polyp",
                        control_image=mask,
                        #pathology="polyp", 
                        num_inference_steps= 50,
                        guidance_scale = 5,
                        #random_gs=(6,16),
                        gen=torch.manual_seed(60),
                        unet_dir=r"checkpoints/polyp_section/epoch_6",
                        controlnet_dir=r"checkpoints\polyp_controlnrt_depth\epoch_6",
                        output_dir="output/polyp_controlnet/polyp/stomach")

