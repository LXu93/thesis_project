import argparse
from data.dataset import *
from classification import clsNet
import torch
from captum.attr import LayerGradCam, LayerAttribution
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

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
            preds = torch.argmax(outputs, dim=1)  # Get predicted class indices

            all_preds.extend(preds.cpu().numpy())  # Move to CPU and store
            all_labels.extend(labels)  # Move to CPU and store

    # Compute classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    return all_preds, all_labels  # Return for further analysis

def cam(device, model, image):
    # Load pretrained model (ResNet-50)
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
    #model.zero_grad()
    output = model(input_tensor)
    class_idx = torch.argmax(output).item()  # Get predicted class
    attributions = grad_cam.attribute(input_tensor, target=class_idx)
    print(class_idx)
  

    # Convert attributions to numpy for visualization
    attributions = LayerAttribution.interpolate(attributions, (512, 512))
    attributions = attributions.permute(0,2,3,1)
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions = np.maximum(attributions, 0)  # ReLU-like effect
    
    # Display Grad-CAM Heatmap
    plt.imshow(image)
    plt.imshow(attributions, cmap="jet",interpolation='bilinear', alpha=0.2)  # Overlay heatmap
    plt.axis("off")
    plt.show()
    


# Run evaluation

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    test_dataset = prepare_data(["Galar"], 
                            used_classes=None,
                            phase="test",
                            need_tensor=False,
                            resize=False,
                            #resize=(224,224),
                            augment=False,
                            normalize_mean=False,
                            #normalize_mean=[0.485, 0.456, 0.406],
                            #ormalize_std=[0.229, 0.224, 0.225],
                            database_dir="D:\Lu\data_thesis")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    classes = ["non-polyp", "polyp"]
    
    model = clsNet.ResNetClsNet(num_classes=len(classes), fine_tune=False)
    model.load_state_dict(torch.load(r'checkpoints\ResNet_cls_polyp\best.pth'))

    #evaluate_model(device, model, test_loader, classes)
    cam(device, model, test_dataset[501]['image'])