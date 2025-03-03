import os
import argparse
import torch
from diffusers import StableDiffusionPipeline,  DDIMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from data.dataset import *
import wandb
from tqdm import tqdm
from util import Const

def train_stable_diffusion(dataset, 
                           universal_prompt=None, 
                           check_points_dir = "checkpoints", 
                           fine_tune=None,
                           epoch_num = 20):
    # Parameters
    model_name = "runwayml/stable-diffusion-v1-5" # Pretrained model checkpoint
    num_epochs = epoch_num
    batch_size = 8
    learning_rate = 5e-6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #wandb.init(project="SD1.5-WCE-inflammation")

    # capture a dictionary of hyperparameters with config
    #wandb.config = {"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size, "train_timestep":1000}

    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.to("cuda")
    if fine_tune:
        pipeline.unet = UNet2DConditionModel.from_pretrained(fine_tune).to(device)
    vae = pipeline.vae  # Variational Autoencoder
    
    #pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, num_train_timesteps = 2000)  # Adjust total diffusion steps
    text_encoder = pipeline.text_encoder  # CLIP Text Encoder
    unet = pipeline.unet  # U-Net for denoising

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Training loop

    unet.train()
    if not os.path.isdir(check_points_dir):
        os.makedirs(check_points_dir)
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            pixel_values = batch["image"].to(device)
            if universal_prompt:
                input_ids = pipeline.tokenizer([universal_prompt]*batch_size, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
            else:
                input_ids = pipeline.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)

            # Encode text
            encoder_hidden_states = text_encoder(input_ids)["last_hidden_state"].to(device)

            # Encode images into latent space
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215  # Scaling factor
            
            # Generate noise
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

            # Add noise to latents
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})
            #wandb.log({"train loss":loss})

        # Save model checkpoint
        if (epoch + 1) % 1 == 0:
            unet.save_pretrained(os.path.join(check_points_dir, f"epoch_{epoch+1}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir', default="D:\Lu\data_thesis")
    parser.add_argument('--pathology', default="polyp")
    parser.add_argument('--use_dataset')
    parser.add_argument('--id', default="combine_fake_data_20")
    parser.add_argument('--unviversal_prompt')
    parser.add_argument('--finetune', default=None)
    parser.add_argument('--epoch_num', default=10)

    args = parser.parse_args()

    dataset = prepare_data(["SUN2CAP",
                            "CAPTIV8",
                            "Mark-data"], 
                            Const.Text_Annotation[args.pathology])
    
    check_points_dir = os.path.join("checkpoints",args.id)

    train_stable_diffusion(dataset, 
                           universal_prompt="an endoscopy image with polyp", 
                           check_points_dir= check_points_dir,
                           fine_tune="checkpoints/combine_fake_data/epoch_20",
                           epoch_num=args.epoch_num)
