import os
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog, StringVar, IntVar, DoubleVar
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
from tensorflow.keras.models import load_model
os.chdir("data")
window = tk.Tk()
window.title('my window')
window.geometry('1000x550')
img_file = 'f02/image/0040.png'
img_file2 = 'f02/label/0040.png'
img_file3 = 'predict_f3_ops_110/0020.png'
img_file4 = 'f02/label/0040.png'
test_img_file = StringVar()
test_img_file2 = StringVar()
model_file = StringVar()
var0 = StringVar() 
var1 = StringVar() 
var2 = StringVar()
var3 = StringVar()
var4 = StringVar()
var5 = StringVar()
var6 = StringVar()
var7 = StringVar()
var8 = StringVar()
var9 = StringVar()
var10 = StringVar()
var11 = StringVar()
var12 = StringVar()
var13 = StringVar()
var14 = StringVar()
var15 = StringVar()
var16 = StringVar()
var17 = StringVar()
var18 = StringVar()
varavg = StringVar()
vartotal = StringVar()
new_width = 160
new_height = 384
data_w = 160
data_h = 384

## ************************************************************************************
# image
#window_img = Image.open(img_file)   
#window_img = window_img.convert('L') 
#r_window_img = window_img.resize((new_width, new_height), Image.ANTIALIAS)
'''r_window_img= r_window_img.filter(ImageFilter.SHARPEN)
r_window_img = ImageEnhance.Sharpness(r_window_img)
sharpness = 20.0
r_window_img = r_window_img.enhance(sharpness)'''
#tk_img = ImageTk.PhotoImage(r_window_img)  
label_img = tk.Label(window, width=new_width, height=new_height)   
label_img.place(x=20,y=100) 

#window_img2 = Image.open(img_file2)  
#r_window_img2 = window_img2.resize((new_width, new_height), Image.ANTIALIAS)  
#tk_img2 = ImageTk.PhotoImage(r_window_img2)  
label_img2 = tk.Label(window, width=new_width, height=new_height)   
label_img2.place(x=220,y=100) 

#window_img3 = Image.open(img_file)  
#r_window_img3 = window_img3.resize((new_width, new_height), Image.ANTIALIAS)   
#tk_img3 = ImageTk.PhotoImage(r_window_img3)  
label_img3 = tk.Label(window, width=new_width, height=new_height)   
label_img3.place(x=620,y=100) 

#window_img4 = Image.open(img_file)  
#r_window_img4 = window_img4.resize((new_width, new_height), Image.ANTIALIAS)   
#tk_img4 = ImageTk.PhotoImage(r_window_img4)  
label_img4 = tk.Label(window, width=new_width, height=new_height)   
label_img4.place(x=420,y=100) 
## ************************************************************************************

def red(test_Y):
    test_Y[test_Y > 128] = 255
    test_Y[test_Y <= 128] = 0
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
    return tmp3

def size_cal(red_line,test_Y):
    tmp = []
    tmp2 =[]
    for n, r in enumerate(red_line):
        num=0
        minnum=159
        maxnum=0
        if n == 0:
            for i in range(0,r):
                for j in range(data_w):
                    if test_Y[i][j]==255:
                        maxnum=max(j,maxnum)
                        minnum=min(j,minnum)
            tmp.append(minnum)
            tmp2.append(maxnum)
        else:
            for i in range(red_line[n-1],r):
                for j in range(data_w):
                    if test_Y[i][j]==255:
                        maxnum=max(j,maxnum)
                        minnum=min(j,minnum)
            tmp.append(minnum)
            tmp2.append(maxnum)
    print(tmp)
    print(tmp2)
    return tmp,tmp2

def draw(red_line,tmp,tmp2):
    snum=3
    tmp3 = np.zeros((data_h, data_w),dtype=int)
    for n, r in enumerate(red_line):
        num=0
        if tmp2[n]+snum >159:
                tmp2[n]=150
        for i in range(tmp[n]-snum,tmp2[n]+snum):
            if n == 0:
                tmp3[0][i]=255
                tmp3[r-1][i]=255
            else:
                tmp3[r-1][i]=255
        if n == 0:
            for i in range(0,r):
                tmp3[i][tmp[n]-snum]=255
                tmp3[i][tmp2[n]+snum]=255
        else:
            for i in range(red_line[n-1],r):
                tmp3[i][tmp[n]-snum]=255
                tmp3[i][tmp2[n]+snum]=255
    return tmp3
                
def open_file():
    global test_img_file
    global test_img_file2
    test_img_file =  filedialog.askopenfilename(title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
    if test_img_file:
        if test_img_file[-4:]=='.png':
            print(test_img_file)
            sStr1 = test_img_file
            sStr2 = 'data/f0'
            pos = sStr1.find(sStr2)
            window_img = Image.open(test_img_file) 
            window_img = window_img.convert('L') 
            r_window_img = window_img.resize((new_width, new_height), Image.ANTIALIAS) 
            '''r_window_img =ImageOps.equalize(r_window_img)
            pixel2 = r_window_img.load() 
            #r_window_img= r_window_img.filter(ImageFilter.SHARPEN )
            r_window_img = ImageEnhance.Sharpness(r_window_img)
            sharpness = 1.5
            r_window_img = r_window_img.enhance(sharpness)
            pixel = r_window_img.load()
            for x in range(r_window_img.size[0]):
                for y in range(r_window_img.size[1]):
                    l = pixel2[x,y]
                    if x<40 or x>119:
                        pixel[x,y]=int(l/2) '''
            '''r_window_img = ImageEnhance.Sharpness(r_window_img)
            sharpness = 20.0
            r_window_img = r_window_img.enhance(sharpness)'''
            tk_img = ImageTk.PhotoImage(r_window_img)  
            label_img.configure(image=tk_img)
            label_img.image = tk_img

            test_img_file2=sStr1[:pos+9]+"label"+sStr1[pos+14:]
            print(test_img_file2)
            window_img2 = Image.open(test_img_file2)  
            ori_img = window_img2.convert('RGB')
            window_img2 = window_img2.convert('L') 
            r_window_img2 = window_img2.resize((new_width, new_height), Image.ANTIALIAS) 
            tk_img2 = ImageTk.PhotoImage(r_window_img2)  
            label_img2.configure(image=tk_img2)   
            label_img2.image = tk_img2
            '''test_Y = np.reshape(np.array(r_window_img2), (data_h, data_w))
            red_line = red(test_Y)
            red_line.append(data_h-1)
            tmp,tmp2 = size_cal(red_line,test_Y)
            tmp3 =draw(red_line,tmp,tmp2)

            ori_img = ori_img.resize((new_width, new_height), Image.ANTIALIAS)
            pixel = ori_img.load() 
            for x in range(ori_img.size[0]):
                for y in range(ori_img.size[1]):
                    r,g,b = pixel[x,y]
                    if int(tmp3[y][x]) ==255:
                        pixel[x,y] = (255,0,0)
            #r_window_img2 =ImageOps.equalize(r_window_img2)
            #r_window_img2= r_window_img2.filter(ImageFilter.SHARPEN )
            tk_img2 = ImageTk.PhotoImage(ori_img)  
            label_img2.configure(image=tk_img2)   
            label_img2.image = tk_img2 '''

def choose_model():
    global model_file
    model_file = filedialog.askopenfilename(title = "Select file",filetypes = (("h5 files","*.h5"),("all files","*.*")))
    if model_file:
        print(model_file)

def normalize_data(data_x, data_y):
    if(np.max(data_x) > 1):
        data_x = data_x / 255
        data_y = data_y / 255
        data_y[data_y > 0.5] = 1
        data_y[data_y <= 0.5] = 0
    return data_x, data_y


def recover_data(data_y):
    data_y[data_y > 0.5] = 1
    data_y[data_y <= 0.5] = 0
    data_y = data_y * 255
    data_y = np.reshape(data_y, (data_h, data_w))
    return data_y

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


def dc_cal(red_line,p_Y,test_Y):
    test_Y[test_Y > 128] = 255
    test_Y[test_Y <= 128] = 0
    red_line.append(len(p_Y)-1)
    dc_text = np.zeros((len(red_line)),dtype=float)
    tmp,tmp2 = size_cal(red_line,test_Y)
    snum=3
    for n, r in enumerate(red_line):
        num=0
        num2=0
        num3=0
        if tmp2[n]+10 >159:
                tmp2[n]=150
        if n == 0:
            for i in range(0,r):
                for j in range(tmp[n]-snum,tmp2[n]+snum):
                    if p_Y[i][j] == 255:
                        num2+=1
                    if test_Y[i][j] == 255:
                        num3+=1
                    if p_Y[i][j]==test_Y[i][j]==255:
                        num+=1
            #dc_text[n] = num/(r*(tmp2[n]-tmp[n]+2*snum))
            dc_text[n] = 2*num/((num2+num3))
        else:
            for i in range(red_line[n-1],r):
                for j in range(tmp[n]-snum,tmp2[n]+snum):
                    if p_Y[i][j] == 255:
                        num2+=1
                    if test_Y[i][j] == 255:
                        num3+=1
                    if p_Y[i][j]==test_Y[i][j]==255:
                        num+=1
            #dc_text[n] = num/((r-red_line[n-1])*(tmp2[n]-tmp[n]+2*snum))
            dc_text[n] = 2*num/((num2+num3))
    num=0
    num2=0
    num3=0
    for i in range(data_h):
        for j in range(data_w):
            if p_Y[i][j] == 255:
                num2+=1
            if test_Y[i][j] == 255:
                num3+=1
            if p_Y[i][j]==test_Y[i][j]==255:
                num+=1
    #dcavg = num/(data_h*data_w)
    dctotal = 2*num/((num2+num3))
    return dc_text,dctotal
def run():
    global model_file
    global test_img_file
    global test_img_file2
    #test_img_file='C:/Users/CYH/Desktop/p76074402_py/data/f01/image/0012.png'
    #test_img_file2='C:/Users/CYH/Desktop/p76074402_py/data/f01/label/0012.png'
    #model_file='C:/Users/CYH/Desktop/p76074402_py/data/checkpoints_f2/200_unet_f2.h5'
    if model_file and test_img_file:
        print('sample :', test_img_file)
        print('groundtruth :', test_img_file2)
        print('model :',  model_file)
        if (model_file[-3:]=='.h5' and test_img_file[-4:]=='.png'):
            print('******** run ***********')
            model = load_model(model_file)
            test_X = data_pre(test_img_file)
            test_Y = Image.open(test_img_file2)  
            test_Y = test_Y.convert('L') 
            test_Y = np.array(test_Y.resize((new_width, new_height), Image.ANTIALIAS)) 
            test_Y = np.reshape(test_Y, (data_h, data_w))
            red_line = red(test_Y)
            test_X, test_Y2 = normalize_data(test_X, test_Y)
            pred_Y = model.predict(test_X,batch_size=1)
            p_Y = recover_data(pred_Y)
            tmp = np.zeros((data_h, data_w),dtype=int)
            tmp2 = np.zeros((data_h, data_w),dtype=int)
            for i in range(1,len(p_Y)-1):
                for j in range(1,len(p_Y[0])-1):
                    if p_Y[i][j]==255:
                        if p_Y[i-1][j]==0 or p_Y[i+1][j]==0 or p_Y[i][j-1]==0 or p_Y[i][j+1]==0 or p_Y[i-1][j+1]==0 or p_Y[i+1][j+1]==0 or p_Y[i-1][j-1]==0 or p_Y[i+1][j-1]==0 :
                            tmp[i][j]=255
                            '''tmp[i-1][j]=255
                            tmp[i+1][j]=255
                            tmp[i][j-1]=255
                            tmp[i-1][j-1]=255
                            tmp[i+1][j-1]=255
                            tmp[i][j+1]=255
                            tmp[i-1][j+1]=255
                            tmp[i+1][j+1]=255'''
            dc_text,dctotal=dc_cal(red_line,p_Y,test_Y)
            print(dc_text)
            ori_img=Image.open(test_img_file)
            ori_img = ori_img.convert('RGB')
            ori_img = ori_img.resize((new_width, new_height), Image.ANTIALIAS)
            pixel = ori_img.load() 
            for x in range(ori_img.size[0]):
                for y in range(ori_img.size[1]):
                    r,g,b = pixel[x,y]
                    if int(tmp[y][x]) ==255:
                        pixel[x,y] = (255,0,0)
            tk_img3 = ImageTk.PhotoImage(ori_img)  
            label_img3.configure(image=tk_img3)   
            label_img3.image = tk_img3  
            pimg = Image.fromarray(p_Y.astype('uint8'))
            tk_img4 = ImageTk.PhotoImage(pimg)  
            label_img4.configure(image=tk_img4)   
            label_img4.image = tk_img4
            names = globals()
            for i in range(18): 
                if len(dc_text)>i:
                    names['var%s'%i].set("V"+str(i)+':' +"{:.2f}".format(dc_text[i]))    
                else:
                    names['var%s'%i].set("")
            varavg.set("AVG:"+"{:.2f}".format(sum(dc_text)/len(dc_text)))
            vartotal.set("TOTAL:"+"{:.2f}".format(dctotal))
            #varavg.set("AVG:"+"{:.2f}".format(dcavg))
            '''img = Image.fromarray(p_Y.astype('uint8'))
            tk_img3 = ImageTk.PhotoImage(img)  
            label_img3.configure(image=tk_img3)   
            label_img3.image = tk_img3 '''
        

# button
b1 = tk.Button(window, text='Select Image', bg='blue',font=('Rockwell', 12), fg = "white", width=17, height=2, command=open_file)
b1.place(x=20,y=20)
b2 = tk.Button(window, text='Select Model', bg='blue',font=('Rockwell', 12), fg = "white", width=17, height=2, command=choose_model)
b2.place(x=220,y=20)
b3 = tk.Button(window, text='Run', bg='blue',font=('Rockwell', 12), fg = "white", width=17, height=2, command=run)
b3.place(x=420,y=20)

# label
l1 = tk.Label(window, text='Source', font=('Rockwell', 12))
l1.place(x=70,y=486)
l2 = tk.Label(window, text='Ground Truth', font=('Rockwell', 12))
l2.place(x=240,y=486)
l3 = tk.Label(window, text='Result', font=('Rockwell', 12))
l3.place(x=670,y=486)
l4 = tk.Label(window, text='Predict', font=('Rockwell', 12))
l4.place(x=470,y=486)

# label
dc = tk.Label(window, text='DC:', font=('Rockwell', 12))
dc.place(x=800,y=100)
name2 = globals()
for i in range(9): 
    name2['d%s'%(i)] = tk.Label(window, textvariable=name2['var%s'%(i)], font=('Rockwell', 12))
    name2['d%s'%(i)].place(x=800,y=130+i*30)
    name2['d%s'%(i+9)] = tk.Label(window, textvariable=name2['var%s'%(i+9)], font=('Rockwell', 12))
    name2['d%s'%(i+9)].place(x=900,y=130+i*30)
davg = tk.Label(window, textvariable=varavg, font=('Rockwell', 12))
davg.place(x=800,y=430)
dtotal = tk.Label(window, textvariable=vartotal, font=('Rockwell', 12))
dtotal.place(x=800,y=460)
'''
d0 = tk.Label(window, textvariable=var0, font=('Rockwell', 12))
d0.place(x=600,y=130)
d1 = tk.Label(window, textvariable=var1, font=('Rockwell', 12))
d1.place(x=600,y=160)
d2 = tk.Label(window, textvariable=var2, font=('Rockwell', 12))
d2.place(x=600,y=190)
d3 = tk.Label(window, textvariable=var3, font=('Rockwell', 12))
d3.place(x=600,y=220)
d4 = tk.Label(window, textvariable=var4, font=('Rockwell', 12))
d4.place(x=600,y=250)
d5 = tk.Label(window, textvariable=var5, font=('Rockwell', 12))
d5.place(x=600,y=280)
d6 = tk.Label(window, textvariable=var6, font=('Rockwell', 12))
d6.place(x=600,y=310)
d7 = tk.Label(window, textvariable=var7, font=('Rockwell', 12))
d7.place(x=600,y=340)
d8 = tk.Label(window, textvariable=var8, font=('Rockwell', 12))
d8.place(x=700,y=130)
d9 = tk.Label(window, textvariable=var9, font=('Rockwell', 12))
d9.place(x=700,y=160)
d10 = tk.Label(window, textvariable=var10, font=('Rockwell', 12))
d10.place(x=700,y=190)
d11 = tk.Label(window, textvariable=var11, font=('Rockwell', 12))
d11.place(x=700,y=220)
d12 = tk.Label(window, textvariable=var12, font=('Rockwell', 12))
d12.place(x=700,y=250)
d13 = tk.Label(window, textvariable=var13, font=('Rockwell', 12))
d13.place(x=700,y=280)
d14 = tk.Label(window, textvariable=var14, font=('Rockwell', 12))
d14.place(x=700,y=310)
d15 = tk.Label(window, textvariable=var15, font=('Rockwell', 12))
d15.place(x=700,y=340)
d16 = tk.Label(window, textvariable=var15, font=('Rockwell', 12))
d16.place(x=700,y=340)
d17 = tk.Label(window, textvariable=var15, font=('Rockwell', 12))
d17.place(x=700,y=340)
d16 = tk.Label(window, textvariable=varavg, font=('Rockwell', 12))
d16.place(x=600,y=400)'''
window.mainloop()