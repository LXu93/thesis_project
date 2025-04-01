import argparse
from data.dataset import *
from classification import clsNet
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

from util import gradCAM

def evaluate_model(device, model, dataloader, classes=None):
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed
        for batch in tqdm(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"]
            labels = [classes.index(target) for target in labels]

            # Forward pass
            outputs = model(images)
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class indices

            all_preds.extend(preds.cpu().numpy())  # Move to CPU and store
            all_labels.extend(labels)  # Move to CPU and store

    # Compute classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    return all_preds, all_labels  # Return for further analysis

def visual_cam(device, model, image, target_class = None, ground_truth=None):
    model = model.to(device)
    model.eval()

    # Define the target layer for Grad-CAM
    target_layer = model.resnet.layer4[-1]

    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run Grad-CAM
    grad_cam = LayerGradCam(model, target_layer)
    model.zero_grad()
    output = model(input_tensor)
    softmax = nn.Softmax(dim=1)
    output = softmax(output)
    class_idx = torch.argmax(output).item()  # Get predicted class
    if not target_class:
        attributions = grad_cam.attribute(input_tensor, target=class_idx)
    else:
        attributions = grad_cam.attribute(input_tensor, target=target_class)
    print(output)

    # Convert attributions to numpy for visualization
    width, height = image.size
    attributions = F.interpolate(attributions, size=(width, height), mode='bilinear', align_corners=False)
    #attributions = LayerAttribution.interpolate(attributions, (width, height))
    attributions = attributions.permute(0,2,3,1)
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions = np.maximum(attributions, 0)
    if np.max(attributions) == np.min(attributions) and np.max(attributions)>0:
        attributions = np.ones(attributions.shape)
    elif np.max(attributions)>0:
        attributions = (attributions - np.min(attributions))/(np.max(attributions) - np.min(attributions))
    
    # Display Grad-CAM Heatmap
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.imshow(attributions, cmap="jet",interpolation='bilinear', alpha=0.2)  # Overlay heatmap
    if ground_truth:
        plt.title(f'predicted class: {classes[class_idx]}, ground truth: {ground_truth}')
    else:
        plt.title(f'predicted class: {classes[class_idx]}')
    plt.axis("off")
    plt.show()
    


# Run evaluation

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_dataset = prepare_data(["Galar"], 
                            used_classes=None,#["polyp"],
                            phase="test",
                            need_tensor=False,
                            resize=False,
                            #resize=(224,224),
                            augment=False,
                            normalize_mean=False,
                            #normalize_mean=[0.485, 0.456, 0.406],
                            #normalize_std=[0.229, 0.224, 0.225],
                            database_dir="D:\Lu\data_thesis")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    classes = ["non-polyp", "polyp"]
    #classes = Const.Text_Annotation["technical"]
    
    model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)
    #model = PeftModel.from_pretrained(model, r'checkpoints\ResNet_cls_polyp2\best').to(device)
    #model = model.merge_and_unload()
    model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_polyp\best.pth', weights_only=True))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #evaluate_model(device, model, test_loader, classes)
  
    data = test_dataset[600]
    #gradCAM.visual_cam(device, model, test_dataset[300]['image'], target_class=1, ground_truth=test_dataset[300]['label'], classes=classes)
    attribution, idx = gradCAM.extract_cam(device, model, transform(data['image']).unsqueeze(0).to(device), target_class=1)
    size = (data['org_height'],data['org_width'])
    cam_np = gradCAM.postprocess_attribution(attribution, size, threshold=0)
    count, contours = gradCAM.count_by_contour(cam_np)
    print(count)
    fig, axs = plt.subplots(1, 2)
    gradCAM.visual_contour(cam_np, contours, axs[0])
    axs[1].imshow(data['image'])
    axs[1].set_title(f'predicted: {classes[idx]}')
    plt.show()
