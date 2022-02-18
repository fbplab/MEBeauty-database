import os
import pandas as pd
from progiter import ProgIter
import torch
from torch.autograd import Variable
import cv2
import time
from torchvision import transforms, models, io
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MEBeauty(Dataset):
    
    """Facial Beauty Dataset"""

    def __init__(self, root_dir, train_scores, test_scores, train = True, 
                    transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
       
    
        if train == True:
            folder = train_scores
        else:
            folder = test_scores
            
        self.root_dir = root_dir
        self.images_scores = pd.read_csv(os.path.join(self.root_dir, folder))
        self.transform = transform

    def __len__(self):
        
        return len(self.images_scores)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.images_scores.iloc[idx, 0])
        #image = io.read_image(img_name)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        score = self.images_scores.iloc[idx, 1]
        if self.transform is not None:
            image = self.transform(image)
            
        return image, score


def train_test_data(train_scores, test_scores,root_dir = '', train_augmentation = False, batch = 16, num_workers = 16, pin_memory = True):
    
    # get train and test dataloader from MEBeauty dataset
    
    image_size = (256,256)
    
    if train_augmentation == True:
        
        transform_train  = transforms.Compose([transforms.ToTensor(),
                              transforms.Resize(image_size),
                             # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),                 
                              transforms.RandomRotation(degrees=30),
                              transforms.GaussianBlur(kernel_size=501),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip()
                              ])
    else:
        
        transform_train  = transforms.Compose([transforms.ToTensor(),
                              transforms.Resize(image_size),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  
                              ])
        
    
    transform_test  = transforms.Compose([transforms.ToTensor(),
                          transforms.Resize(image_size),
                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  
                          ])
    
    train_data = MEBeauty(root_dir, train_scores,test_scores, train = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch, shuffle =True, num_workers = num_workers, pin_memory = pin_memory)
    test_data = MEBeauty(root_dir, train_scores,test_scores, train = False, transform = transform_test)
    testloader = torch.utils.data.DataLoader(train_data, batch_size = batch, shuffle =True, num_workers = num_workers, pin_memory = pin_memory)
    
    return trainloader, trainloader
  
def plot_training(train_losses, valid_losses):
    
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.xlabel("epoch")
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.legend()
   