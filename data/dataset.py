import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import sys
from os.path import dirname as dir
sys.path.append(dir(sys.path[0]))
from util import Const, standard_label, prompt_generator

def read_path_and_labels(csv_file_path, used_classes = None, phase = None):
    df = pd.read_csv(csv_file_path)
    if phase is not None:
      df = df[df["Split"]==phase]
    if used_classes is not None:
      df = df[df["Label"].isin(used_classes)]
    images_path_list = df["Image Path"].tolist()
    label_list = df["Label"].tolist()
    section_list = []
    if "Section" in df.columns:
       section_list = df["Section"].tolist()
    return images_path_list, label_list, section_list

class MyDataSet(torch.utils.data.Dataset):
  def __init__(self, image_path_list, label_list, section_list = [], transform=None):
    super().__init__()
    self.image_path_list = image_path_list
    self.label_list = label_list
    self.section_list = section_list
    self.transform = transform

  def __len__(self):
    return len(self.image_path_list)

  def __getitem__(self, idx):
    image = Image.open(self.image_path_list[idx])
    label = self.label_list[idx]
    if self.section_list:
      section = self.section_list[idx]
    else:
      section = "unknown"
    if self.transform:
      image = self.transform(image)
    return {"image": image, "label": label, "section": section}
  
def prepare_data(dataset_list, 
                 used_classes = None, 
                 need_tensor = True,
                 augment = False,
                 normalize_mean = [0.5]*3, 
                 normalize_std = [0.5] *3,
                 resize=(320,320), 
                 database_dir="D:\Lu\data_thesis",
                 phase=None):
    database_dir = database_dir
    data_summary = pd.read_excel(os.path.join(database_dir,"dataset_summary.xlsx"))
    datasets = []
    for dataset in dataset_list:
        dataset_dir = data_summary[data_summary["Name"]==dataset]["Dataset dir"].item()
        dataset_dir = os.path.join(*dataset_dir.split("\\")) # adapt Linux
        csv_path = data_summary[data_summary["Name"]==dataset]["Label File"].item()
        csv_path = os.path.join(*csv_path.split("\\")) # adapt Linux
        image_path_list, label_list, section_list = read_path_and_labels(os.path.join(database_dir,dataset_dir,csv_path), used_classes, phase)
        image_path_list = [os.path.join(database_dir,dataset_dir,*image_path.split("\\")) for image_path in image_path_list]
        preprocess_list = []
        if dataset in Const.CROP.keys():
           preprocess_list.append(transforms.CenterCrop(Const.CROP[dataset]))
        if resize:
           preprocess_list.append(transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR))
        if augment:
           preprocess_list += [transforms.RandomHorizontalFlip(p=0.3),
                               transforms.RandomVerticalFlip(p=0.3),
                              transforms.RandomRotation(degrees=(-30,30))]
        if need_tensor:
           preprocess_list.append(transforms.ToTensor())
        if normalize_mean and normalize_std:
           preprocess_list.append(transforms.Normalize(normalize_mean, normalize_std))
        preprocess = transforms.Compose(preprocess_list)
        datasets.append(MyDataSet(image_path_list, label_list, section_list, transform=preprocess))
    dataset = ConcatDataset(datasets)
    return dataset

class GeneratedDataSet(torch.utils.data.Dataset):
  def __init__(self, image_dir, label, transform=None):
    super().__init__()
    self.image_dir = image_dir
    self.label = label
    self.transform = transform
    self._image_path_list = []
    for _, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                self._image_path_list.append(os.path.join(image_dir, file))

  def __len__(self):
    return len(self._image_path_list)

  def __getitem__(self, idx):
    image = Image.open(self._image_path_list[idx])
    text = self.label
    if self.transform:
      image = self.transform(image)
    return {"image": image, "label": text}

if __name__ == "__main__":
  dataset = prepare_data(["Galar-tech"], None)#Const.Text_Annotation["technical"])
  print(len(dataset))
  dataloader = DataLoader(dataset, batch_size=8)
  batch = next(iter(dataloader))
  print(prompt_generator(standard_label(dataset[17000]["label"]), ' '.join(dataset[17000]["section"].split('_')), "an endoscopy image"))