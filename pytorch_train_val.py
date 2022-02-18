# train, test on MEBeauty and plot the results

import pytorch_mebeauty_dataset
import argparse
from progiter import ProgIter
import torch
import time
from torchvision import models
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim



def fit(model, loss_func, opt, train_dl, valid_dl, device = "cpu", epochs = 10):
    
    
    train_losses = []
    val_losses = []

    
    best_model_wts = model.state_dict()
    best_loss = 100.0
    
    for epoch in range(epochs):
        
        model.train()
        loss_sum = 0
        loss_sum = 0
        for xb, yb in ProgIter(train_dl):
            xb, yb = xb.to(device), yb.to(device)
  
            loss = loss_func(model(xb).reshape(-1), yb.float())
            loss_sum += loss.item()
            
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('train loss {:.3f}'.format(loss_sum / len(train_dl)))    
        train_losses.append(loss_sum / len(train_dl))
        model.eval()
        loss_sum = 0
        correct = 0
        num = 0
        with torch.no_grad():
            for xb, yb in ProgIter(valid_dl):
                xb, yb = xb.to(device), yb.to(device)
                
                probs = model(xb).reshape(-1)
                loss_sum += loss_func(probs, yb).item()
                
                _, preds = torch.max(probs, axis=-1)
                correct += (preds == yb).sum().item()
                num += len(xb)
        val_loss = loss_sum / len(valid_dl)         
        print('val loss {:.3f}'.format(val_loss))          
        val_losses.append(val_loss)
        
        # если достиглось лучшее качество, то запомним веса модели
        if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = model.state_dict()
        else:
                print("Loss was better in the previous epoch")
        
        torch.save(model, './pytorch_trained_models/model_'+ str(epoch) + time.strftime("%Y%m%d-%H%M%S") +'.pht')
     
    return train_losses, val_losses


def model_preparation(base_model, device):
    
    if base_model == 'densenet':   
        model = models.densenet161(pretrained = True, progress = False)
        in_features  = 2208 
        print("\n The model is fine-tuned on DenseNet \n")
            
    elif base_model == 'mobilenet':   
        model = models.mobilenet_v2(pretrained = True, progress = False)
        in_features  = 1280 
        print("\n The model is fine-tuned on MobileNet \n")
        
    elif base_model == 'alexnet':    
        model = models.alexnet(pretrained = True, progress = False)
        in_features  = 9216
        print("\n The model is fine-tuned on AlexNet \n")
        
    else:  
        model = models.vgg16(pretrained = True, progress = False)
        in_features  = 25088
        print("\n The model is fine-tuned on VGG16 \n")
        
    
    #model = models.vgg16(pretrained = True, progress = False)
    model.classifier = nn.Sequential(
    nn.Linear(in_features=in_features, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=1, bias=True)
    )

    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_model', type=str, help='base model',
                         default = "vgg16")
    parser.add_argument('--train_augmentation', type=bool, help='train augmentation?',
                         default = False)
    parser.add_argument('--train_scores', type=str, help='csv file with scores for training',
                         default = 'scores/train_crop.csv')
    parser.add_argument('--test_scores', type=str, help='csv file with scores for validation',
                         default = 'scores/test_crop.csv')
    parser.add_argument('--batch_size', type=int, help='batch size',
                         default = 16)
    parser.add_argument('--epochs', type=int, help='number of epochs',
                         default = 25)
    parser.add_argument('--num_workers', type=int, help='number of workers',
                         default = 8)
    parser.add_argument('--pin_memory', type=int, help='pin_memory',
                         default = True)
    args = parser.parse_args()
    
    base_model = args.base_model
    train_aug = args.train_augmentation
    train_scores = args.train_scores
    test_scores = args.test_scores
    batch = args.batch_size
    epochs = args.epochs
    num_workers = args.num_workers
    pin_memory = args.pin_memory
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    print('The network is training on',device)
    traindata, testdata = pytorch_mebeauty_dataset.train_test_data(train_scores, test_scores, train_augmentation = train_aug, 
                                                           batch = batch, num_workers = num_workers, pin_memory = pin_memory) # train and test dataloaders
    
    model, criterion, optimizer = model_preparation(base_model, device)
    
    train_loss, val_loss = fit(model, criterion, optimizer, traindata, testdata, device, epochs)
    
    #mebeauty_dataset.plot_training(train_loss, val_loss)
    
    