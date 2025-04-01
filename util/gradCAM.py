import os
from PIL import Image
import torch
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import cv2

def extract_cam(device, model, input_tensor, target_class=None):
    model = model.to(device)
    model.eval()

    # Define the target layer for Grad-CAM
    target_layer = model.resnet.layer4[-1]

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
    return attributions, class_idx

def postprocess_attribution(attribution, target_size, threshold=0):
    cam = F.interpolate(attribution, size=target_size, mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()  # [H, W]

    # Normalize to [0, 1]
    cam = np.maximum(cam,0)
    if np.max(cam) == np.min(cam) and np.max(cam)>0:
        cam_norm = np.ones(cam.shape)
    elif np.max(cam) == np.min(cam) and np.max(cam)==0:
        cam_norm = np.zeros(cam.shape)
    elif np.max(cam)>0:
        cam_norm = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
    
    cam_thresh = np.where(cam_norm >= threshold, cam_norm, 0) 
    return cam_thresh

def postprocess_and_save_cam(
    cams,                      # Tensor: [B, 1, H, W] or [B, H, W]
    main_dir,                  # Output folder
    image_names=None,          # Optional: ["name1.png", "name2.png", ...]
    sizes=None,           # Output size (H, W)
    threshold=0.2 
):
     
    B = cams.shape[0]

    for i in range(B):
        cam = cams[i]  # Shape: [1, H, W]
        #print(cam.shape)
        target_size = sizes[i] if sizes and i < len(sizes) else cam.shape[-2:]
        cam = cam.unsqueeze(0) #[1, C, H, W]
        cam_thresh = postprocess_attribution(cam, target_size, threshold)
        cam_uint8 = np.uint8(cam_thresh * 255)

        # Save as grayscale image
        img = Image.fromarray(cam_uint8, mode='L')
        name = image_names[i] if image_names and i < len(image_names) else f"cam_{i}.png"
        save_dir = os.path.join(main_dir, *name.split('\\')[:-1])
        save_path = os.path.join(main_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        img.save(save_path)

def visual_cam(device, model, image, target_class = None, ground_truth=None, classes = None):

    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run Grad-CAM
    attributions, class_idx = extract_cam(device, model, input_tensor, target_class)
    #print(attributions.shape)
    # Convert attributions to numpy for visualization
    width, height = image.size
    attributions = F.interpolate(attributions, size=(width, height), mode='bilinear', align_corners=False)
    #attributions = LayerAttribution.interpolate(attributions, (width, height))
    attributions = attributions.permute(0,2,3,1)
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions = np.maximum(attributions, 0)
    if np.max(attributions) == np.min(attributions) and np.max>0:
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
    if classes:
        if ground_truth:
            plt.title(f'predicted class: {classes[class_idx]}, ground truth: {ground_truth}')
        else:
            plt.title(f'predicted class: {classes[class_idx]}')
    plt.axis("off")
    plt.show()

def count_by_contour(sal_np, threshold=0.6, min_area=40):
    binary = (sal_np > threshold).astype(np.uint8) * 255
    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    polyp_count = len(significant_contours)

    return polyp_count, significant_contours

def visual_contour(cam_np, contours, ax=None):
    img = cv2.merge((cam_np,cam_np,cam_np))
    cv_img = img.copy()
    cv2.drawContours(cv_img, contours, -1, (255, 0, 0), 2)  # Draw in blue
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(cv_img)
    
    if ax.figure:
        if len(ax.figure.axes) == 1:  # Only show if it's a standalone plot
            plt.tight_layout()
            plt.show()
    