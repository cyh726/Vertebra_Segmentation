import os
import glob
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from model import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
from tensorflow.keras.models import load_model

os.chdir("data")
data_w = 160
data_h = 384
batch_size = 1
img_test_x1 = np.zeros((20, data_h, data_w, 1))
img_test_y1 = np.zeros((20, data_h, data_w, 1))
img_test_x2 = np.zeros((20, data_h, data_w, 1))
img_test_y2 = np.zeros((20, data_h, data_w, 1))
img_test_x3 = np.zeros((20, data_h, data_w, 1))
img_test_y3 = np.zeros((20, data_h, data_w, 1))
img_x1 = np.zeros((40, data_h, data_w, 1))
img_y1 = np.zeros((40, data_h, data_w, 1))
img_x2 = np.zeros((40, data_h, data_w, 1))
img_y2 = np.zeros((40, data_h, data_w, 1))
img_x3 = np.zeros((40, data_h, data_w, 1))
img_y3 = np.zeros((40, data_h, data_w, 1))
train_X = np.zeros((batch_size, data_h, data_w, 1))
train_Y = np.zeros((batch_size, data_h, data_w, 1))
i = 0 
def data_pre(file):
    im = Image.open(file)
    im = im.convert('L')
    im = im.resize((data_w, data_h),Image.ANTIALIAS)
    im =ImageOps.equalize(im)
    pixel2 = im.load() 
    im= im.filter(ImageFilter.SHARPEN)
    '''im = ImageEnhance.Sharpness(im)
    sharpness = 20.0
    im = im.enhance(sharpness)'''
    pixel = im.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            l = pixel2[x,y]
            if x<40 or x>119:
                pixel[x,y]=int(l/2)
    im = np.reshape(np.array(im), (1, data_h, data_w, 1))
    return im
for file in glob.glob(os.path.join("f01/image/"+"*.png")):
    im = data_pre(file)
    img_test_x1[i] = im
    i+=1
i = 0 
for file in glob.glob(os.path.join("f01/label/"+"*.png")):
    im = Image.open(file)
    im = im.convert('L')
    im = np.array(im.resize((data_w, data_h),Image.ANTIALIAS))
    im = np.reshape(im, (1, data_h, data_w, 1))
    img_test_y1[i] = im
    i+=1
i = 0 
for file in glob.glob(os.path.join("f02/image/"+"*.png")):
    im = data_pre(file)
    img_test_x2[i] = im
    i+=1
i = 0 
for file in glob.glob(os.path.join("f02/label/"+"*.png")):
    im = Image.open(file)
    im = im.convert('L')
    im = np.array(im.resize((data_w, data_h),Image.ANTIALIAS))
    im = np.reshape(im, (1, data_h, data_w, 1))
    img_test_y2[i] = im
    i+=1
i = 0 
for file in glob.glob(os.path.join("f03/image/"+"*.png")):
    im = data_pre(file)
    img_test_x3[i] = im
    i+=1
i = 0 
for file in glob.glob(os.path.join("f03/label/"+"*.png")):
    im = Image.open(file)
    im = im.convert('L')
    im = np.array(im.resize((data_w, data_h),Image.ANTIALIAS))
    im = np.reshape(im, (1, data_h, data_w, 1))
    img_test_y3[i] = im
    i+=1

def normalize_data(data_x, data_y):
    if(np.max(data_x) > 1):
        data_x = data_x / 255
        data_y = data_y / 255
        data_y[data_y > 0.5] = 1
        data_y[data_y <= 0.5] = 0
    return data_x, data_y

def batch_data(data_x, data_y, batch_size, batch_epoch):
    data_x, data_y = normalize_data(data_x, data_y)
    data_x = data_x[batch_epoch*batch_size:(batch_epoch+1)*batch_size]
    data_y = data_y[batch_epoch*batch_size:(batch_epoch+1)*batch_size]
    return data_x, data_y

def recover_data(data_y):
    m = 0.5
    data_y[data_y > m] = 1
    data_y[data_y <= m] = 0
    data_y = data_y * 255
    data_y = np.reshape(data_y, (data_h, data_w))
    return data_y

itr = 60
batch_epoch = 0
model = load_model('checkpoints_f1_ops/170_unet_f1.h5')
file_dic ='predict_f1_ops_170/'
if not os.path.exists(file_dic):
        os.makedirs(file_dic)

for j in range(20):
    test_X, test_Y = batch_data(img_test_x3, img_test_y3, batch_size, j)
    pred_Y = model.predict(test_X,batch_size=batch_size)
    p_Y = recover_data(pred_Y)
    img = Image.fromarray(p_Y.astype('uint8')) 
    #img = img.filter(ImageFilter.MedianFilter)
    #img2= img.filter(ImageFilter.CONTOUR)
    #pixel2 = img2.load() 
    '''img = img.convert('RGB')
    pixel = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            #if x>1 or x<159 or y>1 or y<383:
            r,g,b = pixel[x,y]
            l = pixel2[x,y]
            pixel[x,y] = (int(r+255-l),g,b)'''
            
    #img= img.filter(ImageFilter.RankFilter(5,13))
    img.save(file_dic+str(j+1)+'.png')