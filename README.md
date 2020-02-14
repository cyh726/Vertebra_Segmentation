# Vertebra_Segmentation
* **environment:** python 3.6.9 & tensorflow1.15
## Requirements
You may install the requirements by running the following command
```
sudo pip3 install -r requirements.txt
```
## Execute
You may run the program by running the following command
```
python image_processing.py
```
## GUI Function
1.	Read an input image and its ground truth 
2.	Show predict results 
3.	Show the overlapping of segmentation result(red) on the original input image  
4.	Show the evaluation result (DC) 
## Image Processing
1. Histogram Equalization
2. Image Sharpening 
## Model
* **Model:** UNet
* **Loss:** binary crossentropy 
## Result
![](https://i.imgur.com/BYhmwUA.png)
