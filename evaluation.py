import torch
_ = torch.manual_seed(123)
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import *


def get_kid_score(real_dataloader, fake_dataloader):
    kid = KernelInceptionDistance(subset_size=50)
    for real_samples in tqdm(real_dataloader):
        real_images = real_samples["image"].mul(255).to(torch.uint8)
        kid.update(real_images, real=True)
    for fake_samples in tqdm(fake_dataloader):
        fake_images = fake_samples["image"].mul(255).to(torch.uint8)
        kid.update(fake_images, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean, kid_std

def get_fid_score(real_dataloader, fake_dataloader):
    fid = FrechetInceptionDistance(feature=2048)
    for real_samples in tqdm(real_dataloader):
        real_images = real_samples["image"].mul(255).to(torch.uint8)
        fid.update(real_images, real=True)
    for fake_samples in tqdm(fake_dataloader):
        fake_images = fake_samples["image"].mul(255).to(torch.uint8)
        fid.update(fake_images, real=False)
    fid_score = fid.compute()
    
    return fid_score


if __name__ == "__main__":
    dataset = prepare_data(["CAPTIV8", 
                "Mark-data"], Const.Text_Annotation["polyp"], normalize=False, resize=(299,299))

    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        ])
    generated_data = GeneratedDataSet("output\combine_fake_data_25", "polyp",transform)

    real_images_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=True)
    fake_images_loader = DataLoader(generated_data, batch_size=8, shuffle=False, drop_last=True)

    print(get_kid_score(real_images_loader, fake_images_loader))
    print(get_fid_score(real_images_loader, fake_images_loader))