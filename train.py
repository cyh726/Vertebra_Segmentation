import os
import glob
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from model import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
from tensorflow.keras.models import save_model

os.chdir("data")
data_w = 160
data_h = 384
batch_size = 8
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

def concat(test_x1, test_x2, test_y1, test_y2):
    data_x = np.concatenate((test_x1, test_x2), axis = 0)
    data_y = np.concatenate((test_y1, test_y2), axis = 0)
    return data_x, data_y

def normalize_data(data_x, data_y):
    if(np.max(data_x) > 1):
        data_x = data_x / 255
        data_y = data_y / 255
        data_y[data_y > 0.5] = 1
        data_y[data_y <= 0.5] = 0
    return data_x, data_y

def processing_data(data_x, data_y):
    data_x, data_y = normalize_data(data_x, data_y)
    shuffled_indexes = np.random.permutation(data_x.shape[0])
    data_x = data_x[shuffled_indexes]
    data_y = data_y[shuffled_indexes]
    return data_x, data_y

'''
ops: 平均+gray+sharp 170
ops2: 平均+gray
ops3: 平均
'''
itr = 200
model = unet()
loss_record = []
accuracy_record = []
loss_id = np.linspace(0, itr/10-1, itr/10)
file_dic ='checkpoints_f3_ops/'
if not os.path.exists(file_dic):
        os.makedirs(file_dic)
'''
for i in range(itr):
    print("iteration = ", i+1)
    img_x1, img_y1 = concat(img_test_x1, img_test_x2, img_test_y1, img_test_y2)
    train_X, train_Y = processing_data(img_x1, img_y1)
    model.fit(train_X, train_Y, epochs=5, batch_size=batch_size, verbose=1)
    if i % 10 == 9:
        test_X, test_Y = processing_data(img_test_x3, img_test_y3)	
        loss, accuracy = model.evaluate(test_X, test_Y, batch_size, verbose=1)
        loss_record.append(loss)
        accuracy_record.append(accuracy)
        save_model(model, file_dic+str(i+1)+'_unet_f1.h5')
plt.plot(loss_id,loss_record)
plt.title("loss")
plt.savefig(file_dic+"loss_f1.png")
plt.figure()
plt.plot(loss_id,accuracy_record)
plt.title("accuracy")
plt.savefig(file_dic+"accuracy_f1.png")'''
'''
for i in range(itr):
    print("iteration = ", i+1)
    img_x2, img_y2 = concat(img_test_x2, img_test_x3, img_test_y2, img_test_y3)
    train_X, train_Y = processing_data(img_x2, img_y2)
    model.fit(train_X, train_Y, epochs=5, batch_size=batch_size, verbose=1)
    if i % 10 == 9:
        test_X, test_Y = processing_data(img_test_x1, img_test_y1)	
        loss, accuracy = model.evaluate(test_X, test_Y, batch_size, verbose=1)
        loss_record.append(loss)
        accuracy_record.append(accuracy)
        save_model(model, file_dic+str(i+1)+'_unet_f2.h5')
plt.plot(loss_id,loss_record)
plt.title("loss")
plt.savefig(file_dic+"loss_f2.png")
plt.figure()
plt.plot(loss_id,accuracy_record)
plt.title("accuracy")
plt.savefig(file_dic+"accuracy_f2.png")'''

for i in range(itr):
    print("iteration = ", i+1)
    img_x3, img_y3 = concat(img_test_x3, img_test_x1, img_test_y3, img_test_y1)
    train_X, train_Y = processing_data(img_x3, img_y3)
    model.fit(train_X, train_Y, epochs=5, batch_size=batch_size, verbose=1)
    if i % 10 == 9:
        test_X, test_Y = processing_data(img_test_x2, img_test_y2)	
        loss, accuracy = model.evaluate(test_X, test_Y, batch_size, verbose=1)
        loss_record.append(loss)
        accuracy_record.append(accuracy)
        save_model(model, file_dic+str(i+1)+'_unet_f3.h5')
plt.figure()
plt.plot(loss_id,loss_record)
plt.title("loss")
plt.savefig(file_dic+"loss_f3.png")
plt.figure()
plt.plot(loss_id,accuracy_record)
plt.title("accuracy")
plt.savefig(file_dic+"accuracy_f3.png")
