import os
import glob
import math
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps

os.chdir("data")
data_w = 160
data_h = 384
new_width = 160
new_height = 384
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
    im.save('1.png')
    im = np.reshape(np.array(im), (1, data_h, data_w, 1))
    return im
for file in glob.glob(os.path.join("f01/image/"+"*.png")):
    im = data_pre(file)
    img_test_x1[i] = im
    i+=1
'''
for file in glob.glob(os.path.join("f01/label/"+"*.png")):
    test_img_file2=file
    #test_img_file2='C:/Users/CYH/Desktop/p76074402_py/data/f03/label/0045.png'
    test_Y = Image.open(test_img_file2)  
    test_Y = test_Y.convert('L') 
    test_Y = np.array(test_Y.resize((new_width, new_height), Image.ANTIALIAS)) 
    test_Y = np.reshape(test_Y, (data_h, data_w))
    test_Y[test_Y > 128] = 255
    test_Y[test_Y <= 128] = 0
    #tmp3 = np.zeros(((len(test_Y)-2)),dtype=int)
    tmp2=[]
    for i in range(15,len(test_Y)-20):
        num=0
        for j in range(1,len(test_Y[0])-1):
            if test_Y[i][j]==0:
                num+=1
        if num >140:
            tmp2.append(i)

    tmp3 =[tmp2[0]]
    tmp4=[]
    for i in range(1,len(tmp2)):
        if tmp2[i]-tmp3[-1]>3:
            tmp4.append(tmp3[int(len(tmp3)/2)])
            tmp3 = [tmp2[i]]
        else:
            tmp3.append(tmp2[i])
            if i == len(tmp2)-1:
                tmp4.append(tmp3[int(len(tmp3)/2)])
    tmp3 =[tmp4[0]]
    for i in range(1,len(tmp4)):
        if tmp4[i]-tmp4[i-1]>40:
            tmp3.append(int((tmp4[i]+tmp4[i-1])/2))
            tmp3.append(tmp4[i])
        else:
            tmp3.append(tmp4[i])
    if 384-tmp3[-1]>50:
        tmp3.append(int((384+tmp3[-1])/2))

    tmp4 = tmp3.copy()
    tmp3 =[tmp4[0]]
    for i in range(1,len(tmp4)):
        if tmp4[i]-tmp4[i-1]>40:
            tmp3.append(int((tmp4[i]+tmp4[i-1])/2))
            tmp3.append(tmp4[i])
        else:
            tmp3.append(tmp4[i])
    if 384-tmp3[-1]>50:
        tmp3.append(int((384+tmp3[-1])/2))
    if tmp3[0]-0>45:
        tmp3=[int((tmp3[0])/2)]+tmp3
    print(len(tmp3))
    print(tmp3)

    ori_img = Image.open(test_img_file2)
    ori_img = ori_img.convert('RGB')
    ori_img = ori_img.resize((new_width, new_height), Image.ANTIALIAS)
    pixel = ori_img.load() 
    for x in range(ori_img.size[0]):
        for y in range(ori_img.size[1]):
            r,g,b = pixel[x,y]
            if y in tmp3:
                pixel[x,y] = (r+255,g,b)
    ori_img.show()'''
'''for i in range(1,len(test_Y)-1):
        tmp2 = []
        for j in range(1,len(test_Y[0])-1):
            if test_Y[i][j]==255:
                if test_Y[i+1][j]==0:
                    tmp2.append(j)
        tmp3[i-1]=len(tmp2)
    z = 0
    tmp4=[]
    tmp5=[]
    for k in range(32):
        a=np.argmax(tmp3[12*(k):12*(k+1)])
        tmp4.append(a+12*(k))
        if k % 2 ==1:
            if tmp4[k]-tmp4[k-1]>=20:
                tmp5.append(tmp4[k-1])
                tmp5.append(tmp4[k])
            else:
                if tmp3[tmp4[k]]>tmp3[tmp4[k-1]]:
                    tmp5.append(tmp4[k])
                elif tmp3[tmp4[k]]<tmp3[tmp4[k-1]]:
                    tmp5.append(tmp4[k-1])
    tmp4 = tmp5.copy()
    for a in range(len(tmp5)-1):
        if tmp5[len(tmp5)-1-a]-tmp5[len(tmp5)-2-a]<6:
            del tmp4[len(tmp5)-2-a]
    tmp5 = tmp4.copy()
    for a in range(len(tmp5)-1):
        if tmp3[tmp5[len(tmp5)-1-a]]<6:
            del tmp4[len(tmp5)-1-a]
    print(len(tmp4))
    print(tmp4)
    ori_img = Image.open(test_img_file2)
    ori_img = ori_img.convert('RGB')
    ori_img = ori_img.resize((new_width, new_height), Image.ANTIALIAS)
    pixel = ori_img.load() 
    for x in range(ori_img.size[0]):
        for y in range(ori_img.size[1]):
            r,g,b = pixel[x,y]
            if y in tmp4:
                pixel[x,y+1] = (r+255,g,b)'''
    
'''    
tmp4 = np.argsort(-tmp3)
print(tmp4)
tmp5 = tmp4[:16]
print(tmp5)
tmp6 = tmp5[np.argsort(tmp5)]
print(tmp6)'''
'''
def data_pre(file):
    im = Image.open(file)
    im = im.convert('L')
    im = im.resize((data_w, data_h),Image.ANTIALIAS)
    pixel2 = im.load() 
    im= im.filter(ImageFilter.SHARPEN)
    im = ImageEnhance.Sharpness(im)
    sharpness = 5.0
    im = im.enhance(sharpness)
    pixel = im.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            l = pixel2[x,y]
            if x<40 or x>119:
                pixel[x,y]=int(l/2)
    im.show()
    im = np.reshape(np.array(im), (1, data_h, data_w, 1))

    return im
for file in glob.glob(os.path.join("f01/image/"+"*.png")):
    im = data_pre(file)
    i+=1'''
'''tmp = np.zeros((data_h, data_w),dtype=int)
im2 = np.reshape(np.array(r_window_img), (data_h, data_w))
for i in range(1,len(im2)-1):
    for j in range(30,130):
        if im2[i][j]>150:
            #if p_Y[i-1][j]==0 or p_Y[i+1][j]==0 or p_Y[i][j+1]==0 or p_Y[i][j+1]==0:# or p_Y[i-1][j+1]==0 or p_Y[i+1][j+1]==0 or p_Y[i-1][j-1]==0 or p_Y[i+1][j-1]==0 :
            tmp[i][j]=255
            tmp[i-1][j]=255
            tmp[i+1][j]=255
            tmp[i][j-1]=255
            tmp[i-1][j-1]=255
            tmp[i+1][j-1]=255
            tmp[i][j+1]=255
            tmp[i-1][j+1]=255
            tmp[i+1][j+1]=255
pixel3 = r_window_img.load() 
for x in range(r_window_img.size[0]):
    for y in range(r_window_img.size[1]):
        if tmp[y,x]==255:
            pixel3[x,y]=255'''
'''
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
r_big = 0
l_big = 384
for file in glob.glob(os.path.join("f03/label/"+"*.png")):
    im = Image.open(file)
    im = im.convert('L')
    im = np.array(im.resize((data_w, data_h),Image.ANTIALIAS))
    for j in range(len(im)):
       for k in  range(len(im[0])):
           if im[j][k]==255:
                if k>r_big:
                    r_big=k
                if k<l_big:
                    l_big=k
    i+=1
print(r_big)
print(l_big)
'''