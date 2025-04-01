import argparse
from data.dataset import *
from classification import clsNet, gradCAM
import torch
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import classification_report


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_dataset = prepare_data(["Galar"], 
                            used_classes=['non-polyp'],
                            phase="train",
                            #need_tensor=False,
                            resize=False,
                            #resize=(224,224),
                            augment=False,
                            #normalize_mean=False,
                            normalize_mean=[0.485, 0.456, 0.406],
                            normalize_std=[0.229, 0.224, 0.225],
                            database_dir="D:\Lu\data_thesis")
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classes = ['non-polyp','polyp']
    #classes = Const.Text_Annotation["technical"]
    
    model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)
    #model = PeftModel.from_pretrained(model, r'checkpoints\ResNet_cls_polyp2\best').to(device)
    #model = model.merge_and_unload()
    model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_polyp\best.pth', weights_only=True))

    save_dir = r"D:\Lu\data_thesis\galar_CAM\train"
    #evaluate_model(device, model, test_loader, classes)
    for batch in tqdm(data_loader):
        input_tensor = batch["image"].to(device)
        names = batch["name"]
        labels = batch["label"]
        sections = batch["section"]
        image_names = [os.path.join(section,label, name) for section,label,name in zip(sections,labels,names)]
        widths = batch["org_width"]
        heights = batch["org_height"]
        sizes = [(width, height) for width, height in zip(widths, heights)]
        cams, _ = gradCAM.extract_cam(device, model, input_tensor,target_class=1)

        #print(sizes[0])
        gradCAM.postprocess_and_save_cam(
            cams,                      # Tensor: [B, 1, H, W] or [B, H, W]
            save_dir,                  # Output folder
            image_names,          # Optional: ["name1.png", "name2.png", ...]
            sizes,
            threshold=0.3          # Output size (H, W)
            )