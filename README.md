# Multi-ethnic MEBeauty dataset and facial attractiveness assessment

2550 images of Black, Asian, Caucasian, Hispanic, Indian, Mideastern female and male faces, rated by about 300 individuals with various cultural and social background.<br />
Generic(average) and personal scores are included in the `/scores/train_2022, test_2022, val_2022.txt` and `/scores/generic_scores_all_2022.xlsx`, respectively. The 'date' personal type scores are located in `/scores/generic_scores_all_2022.xlsx(column mean)`

The dataset collection (AWS SageMaker), cleaning and analysis can be found in `MEBeauty_creation_cleaning`

![alt text](https://github.com/fbplab/MEBeauty-database/blob/main/ME3.png?raw=true)

### Face cropping and alignment

Crop all the files in the folder and its subfolders and save it to a new folder with the same structure.<br />
Based on DeepFace https://github.com/serengil/deepface

Options: opencv (haar cascade + Adaboost), dlib (HOG + linear SVM), mtsnn, retinaface (RetinaNet)

```bash
 python face_crop_align.py  --images_path [path to the folder with images] 
 --results_path [folder where the cropped images should be saved] 
 --method [one of the backends mentioned above]
    
```

Default backend `opencv`

Images cropped with different backends are located in `cropped_images`

## Deep leaning (Pytorch)

Pytorch dataset, dataloaders are located in `pytorch_mebeauty_dataset.py`, some trained models can be found in `pytorch_trained_models.py`

### Train the model with one of the pretrained base model (Pytorch)

Default train_crop/test_crop.csv folders are average on cropped images. If you want to train on personal scores, please firstly create csv files by using `create_score_lists.ipynb`

```bash
python pytorch_train_val.py --base_model [choosen base model] --train_scores [csv file with train scores]
                                 --test_scores [csv file with test scores]
                                 --train_augmentation [augmented train set or not]
                                 --batch_size [batch size] --epochs [number of epochs]
```
Please note that data augmentation is time-consuming.

Possible base models: densenet, mobilenet, alexnet, vgg16 (default)

### Prediction (Pytorch) 

```bash

 python pytorch_predict.py  --image_path [path to the image for prediction] --model_path [pretrained model filename]
    
```
Default image  - `/inference_samples/girl.jpg`

Default model `/models/model.pht`

### Shallow features and predictors

Eigenface, Gabor, HOG, Landmarks(`landmarks.csv`), SIFT feature extraction and training shallow predictor by them are presented in `Eigenface,Geom.features,HOG,Gabor,SIFT+shallow predictor.ipynb, get_landmarks_geom.features.ipynb` 

### Facenet features

Generation of facenet 512 features, face comparison based in these features are located in `run crop, facenet 512 embedding extractors, facenet comparison.ipynb` and can be used to train shallow and deep predictors. The generated features are in the folder `FaceNet_512_features`

### Keras models and other

Addicitional codes and models for MEBeauty dataset can be found by the link https://github.com/irina-lebedeva/facial_beauty_prediction.


## Citation
If you use this code for your research, please cite our paper.
```
@article{lebedeva2021mebeauty,
  title={MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author={Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2021},
  publisher={Springer}
}
```

Note: The MEBeauty can be only used for non-commercial research purpose.

For any questions about this database please contact the authors by sending email to irina.val.lebedeva@gmail.com

