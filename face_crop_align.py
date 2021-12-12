import os
import cv2
from deepface import DeepFace
import argparse
import numpy as np
import shutil
from PIL import Image

def ignore_files(dir, files):
    
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def crop(images_path, results_path, method):
    
    
    if os.path.isdir(images_path) == False:
        
        print('Can not find the folder ', images_path)
        
        return
    
    if os.path.isdir(results_path) == False:
        
        shutil.copytree(images_path, results_path, ignore=ignore_files)
        
        print('The folder', results_path, 'has been created')
     
    else: 
        
        print('The folder ', results_path, ' already exist, use another folder name to save the result images')
    
        return
    
    for path, subdirs, files in os.walk(images_path):
        for name in files:
            fullname = os.path.join(path, name)
            
            
            
            #face detection and alignment
            try:
                detected_face = DeepFace.detectFace(img_path = fullname, detector_backend = method)
                cv2.imwrite(fullname.replace(images_path, results_path), cv2.cvtColor(detected_face*255, cv2.COLOR_RGB2BGR))
                print(fullname.replace(images_path, results_path))
            except:
            
                print('The file ', fullname.replace('images', 'results'), ' has no face')
            
            
             
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='image folder name',
                         default = 'images' )
    parser.add_argument('--results_path', type=str, help='where to save the crop pics',
                         default = 'crop_align_images' )
    parser.add_argument('--method', type=str, help='the crop method can be opencv, dlib, mtcnn, ssd, retinanet',
                         default = 'opencv' )
    args = parser.parse_args()
    
    images_path = args.images_path
    results_path = args.results_path
    method = args.method
    crop(images_path, results_path, method)
