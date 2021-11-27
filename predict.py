
#predict the beauty score of a face on an image
import argparse
from PIL import Image
import torch
from torchvision import transforms

imsize = (500, 500)
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def image_loader(path_image):
    
    """load image, returns cuda tensor"""
    image = Image.open(path_image)
    image = loader(image).float()
    image = image.view(1, *image.shape)

    return image.to(device)  #assumes that you're using GPU


def predict(path_model, image):
    
    model = torch.load(path_model)
    model = model.to(device)
    model.eval()
    result = model(image)
    
    return result
  
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image file name',
                         default = 'inference_samples/girl.jpg' )
    parser.add_argument('--model_path', type=str, help='pytorch model file name',
                         default = 'models/model.pht' )
    args = parser.parse_args()
    
    image_path = args.image_path
    model_path = args.model_path
    image = image_loader(image_path)
    score = predict(model_path, image)
    print('Beauty score: {:.2f}'.format(score.item()))