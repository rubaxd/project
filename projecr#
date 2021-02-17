
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from cv2 import *


import glob
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import time
import threading
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from sklearn.model_selection import train_test_split
from keras import callbacks

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


root=tk.Tk()
root.configure(background="#83b4c0")


##########################################################
# tarining part # 
def save_model():
    model.save_weights("save")
    
def fireTransferLearning():
    textftfl.set("starting fireTransferLearning ")
    datagen = ImageDataGenerator()
    from keras.applications.vgg16 import VGG16
    baseModel = VGG16 (weights= "imagenet", include_top=False,input_shape=(imgsize,imgsize,3))
    headModel = baseModel.output
    
    headModel = Flatten(name='flatten')(headModel)
    headModel = Dense(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(25, activation='sigmoid')(headModel)
    model = Model(inputs = baseModel.input, outputs= headModel)
    
    for layer in baseModel.layers:
        layer.trainable = False
        
    model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
   
    history = model.fit_generator
    datagen.flow(X_train, y_train, batch_size=32) 
    validation_data=(X_test, y_test) 
    epochs=20
    textftfl.set("Complete fireTransferLearning ")
                  
    
def fireCNN():
 
    textcnn.set("starting CNN")   
    global model
    model = Sequential()
    model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(imgsize,imgsize,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) 
    model.add(Dense(25,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    history = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),verbose=0)
    textcnn.set("Complete CNN")
    
    
def Check():
    if var.get() == 1:
        fireCNN()
    elif var.get() == 2:
        fireTransferLearning();
    else:
        return
    
   

        
def open_cv():
    filename = filedialog.askopenfilename(initialdir="C:/", title="select file",filetypes=(("CSV Files","*.csv"), ("all files", "*.*")))
    global df
    try:
        df = pd.read_csv(filename)
    except:
       print("cant open the file")
       return
    print(df)
    
def open_file_training():
        choose=filedialog.askdirectory()
        global imgsize
        imgsize = 120
        global X
        X = []
        global y
        global path
        global img
        #tqdm
        textvar.set("starting...")
        length = (range(df.shape[0]))
        old = 0
        new = 0
        for i in length:
            new = (i * 100) / int(df.shape[0])
            if int(new) > int(old):
                old = new;
                progress['value'] +=1

            path = choose+'/'+df['Id'][i]+'.jpg'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img= cv2.resize(img,(imgsize,imgsize))
            X.append(img)
            tarinning.update_idletasks()
        X = np.array(X)
        y = df.drop(['Id','Genre'],axis=1)
        y = y.to_numpy()
        global X_train, X_test, y_train, y_test
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
        textvar.set("finished.")

def trainning():
    
    r=IntVar()
    global tarinning
    global B1,B2
    tarinning=Toplevel(root)
    tarinning.geometry("500x500")
    tarinning.title("Trainning")
    tarinning.configure(background="#83b4c0")
    l=Label(tarinning,text="MOVIE GENRE DETECTION SYSTEM ",fg="black", bg="#83b4c0")
    l.place(relx = 0.5, rely = 0.13, anchor=CENTER)
    l=Label(tarinning,text="______________________________________________________________________________________________",fg="black", bg="#83b4c0")
    l.place(relx = 0.5, rely = 0.16, anchor=CENTER)
    selectlabel=Label(tarinning, text="Select notation files",fg="black", bg="#83b4c0")
    selectlabel.place(relx = 0.25, rely = 0.25, anchor=E)
    
    selectlabel=Label(tarinning, text="Select images files",fg="black", bg="#83b4c0")
    selectlabel.place(relx = 0.25, rely = 0.35, anchor=E)
   

    selectbut=Button(tarinning,text="Select", command= open_cv  ,width=8,height=1)
    selectbut=Button(tarinning,text="Select" ,command= open_file_training  ,width=8,height=1)
    
    B1.place(relx = 0.45, rely = 0.25, anchor=E)
    B2.place(relx = 0.45, rely = 0.35, anchor=E)  
    
    global progress
    progress=ttk.Progressbar(tarinning,orient=HORIZONTAL,length=120, mode='determinate')
    progress.place(relx = 0.75, rely=0.35, anchor=E)
    
    global textvar 
    textvar = StringVar();
    textlable = Label(tarinning, textvariable=textvar, fg="black", bg="#83b4c0")
    textlable.place(relx=0.92, rely=0.35, anchor=E)
    
    techlabel=Label(tarinning, text="Deep Learning Techniqes",fg="black", bg="#83b4c0")
    techlabel.place(relx = 0.3, rely = 0.5, anchor=E)


    global var
    var = tk.IntVar()
    rad1=Radiobutton(tarinning,text="CNN",variable=var,value=1,fg="black", bg="#83b4c0")
    rad2=Radiobutton(tarinning,text="Transfer learning",variable=var,value=2,fg="black", bg="#83b4c0")
    
    
    save=Button(tarinning, text="Save", command=save_model,width=8,height=1)
    Start=Button(tarinning, text="Start", command=Check,width=8,height=1)
    rad1.place(relx = 0.11, rely = 0.6, anchor=E)
    rad2.place(relx = 0.24, rely = 0.7, anchor=E)
    Start.place(relx = 0.3, rely = 0.8, anchor=E)
    save.place(relx = 0.5, rely = 0.8, anchor=E)
    
    global textcnn
    textcnn = StringVar();
    textlabcnn = Label(tarinning, textvariable=textcnn, fg="black", bg="#83b4c0")
    textlabcnn.place(relx=0.92, rely=0.7, anchor=E)
    
    global textftfl
    textftfl = StringVar();
    textlabftfl= Label(tarinning, textvariable=textftfl, fg="black", bg="#83b4c0")
    textlabftfl.place(relx=0.92, rely=0.7, anchor=E)

#####################################################################
# testing part # 
def open_file_test():
        choose=filedialog.askdirectory()
        global imgsize
        imgsize = 120
        global X
        X = []
        global y
        global path
        global img
        #tqdm
        textvar.set("starting...")
        length = (range(df.shape[0]))
        old = 0
        new = 0
        for i in length:
            new = (i * 100) / int(df.shape[0])
            if int(new) > int(old):
                old = new;
                progress['value'] +=1

            path = choose+'/'+df['Id'][i]+'.jpg'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img= cv2.resize(img,(imgsize,imgsize))
            X.append(img)
            test.update_idletasks()
        X = np.array(X)
        y = df.drop(['Id','Genre'],axis=1)
        y = y.to_numpy()
        global X_train, X_test, y_train, y_test
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
        textvar.set("finished.") 
        
        
def upload_model():
    
    model1.load_weights("save")
    
def Test_Model():
    global er,ac
    er,ac=model.evaluate(x_test,y_test)
    
def Show_Accuracy():
    print(er,ac)
        
    
def testing():
    global test
    test=Toplevel(root)
    test.geometry("500x500")
    test.title("Testing")
    test.configure(background="#83b4c0")
    l=Label(test,text="MOVIE GENRE DETECTION SYSTEM ",fg="black", bg="#83b4c0")
    l.place(relx = 0.5, rely = 0.13, anchor=CENTER)
    l=Label(test,text="______________________________________________________________________________________________",fg="black", bg="#83b4c0")
    l.place(relx = 0.5, rely = 0.16, anchor=CENTER)
    selectlabel=Label(test, text="Select testing data:",fg="black", bg="#83b4c0")
    selectlabel.place(relx = 0.3, rely = 0.25, anchor=E)
    choose=Button(test, text="Choose Notation File",command= open_cv,width=15,height=2)
    choose.place(relx = 0.3, rely = 0.35, anchor=E)
    choose1=Button(test, text="Choose Image File",command= open_file_test,width=15,height=2)
    choose1.place(relx = 0.3, rely = 0.45, anchor=E)
    
    global progress
    progress=ttk.Progressbar(test,orient=HORIZONTAL,length=120, mode='determinate')
    progress.place(relx = 0.75, rely=0.35, anchor=E)
    
    global textvar 
    textvar = StringVar();
    textlable = Label(test, textvariable=textvar, fg="black", bg="#83b4c0")
    textlable.place(relx=0.92, rely=0.35, anchor=E)
    
    load=Button(test, text="Upload Model",command=upload_model,width=15,height=2)
    load.place(relx = 0.3, rely = 0.55, anchor=E)
    testbut=Button(test, text="Test",width=8,height=1)
    testbut.place(relx = 0.4, rely = 0.75, anchor=E)
    acc=Button(test, text="Show Accuracy",width=12,height=1)
    acc.place(relx = 0.8, rely = 0.75, anchor=E)
    
##################################################################   
def predcting():
    predct=Toplevel(root)
    predct.geometry("500x500")
    predct.title("Testing")
    predct.configure(background="#83b4c0")
    l=Label(predct,text="MOVIE GENRE DETECTION SYSTEM ",fg="black", bg="#83b4c0")
    l.place(relx = 0.5, rely = 0.13, anchor=CENTER)
    l=Label(predct,text="______________________________________________________________________________________________",fg="black", bg="#83b4c0")
    l.place(relx = 0.5, rely = 0.16, anchor=CENTER)
    selectlabel=Label(predct, text="Select predcting data:",fg="black", bg="#83b4c0")
    selectlabel.place(relx = 0.3, rely = 0.25, anchor=E)
    choose=Button(predct, text="Choose pooster")
    choose.place(relx = 0.30, rely = 0.3, anchor=E)
    predctbut=Button(predct, text="Predct")
    predctbut.place(relx = 0.50, rely = 0.3, anchor=E)
    img=Label(predct, text="Image:",fg="black", bg="#83b4c0")
    img.place(relx = 0.2, rely = 0.35, anchor=E)
    pre=Label(predct, text="Predct:",fg="black", bg="#83b4c0")
    pre.place(relx = 0.5, rely = 0.35, anchor=E)
#####################################################################
# main page #   
l=Label(root,text="MOVIE GENRE DETECTION SYSTEM ",fg="black", bg="#83b4c0")
l.place(relx = 0.5, rely = 0.13, anchor=CENTER)
l=Label(root,text="______________________________________________________________________________________________",fg="black", bg="#83b4c0")
l.place(relx = 0.5, rely = 0.16, anchor=CENTER)
l=Label(root,text="Deep Learning ",fg="black", bg="#83b4c0")
l.place(relx = 0.5, rely = 0.2, anchor=CENTER)
trinning=Button(root,text="Trinning", command=trainning,width=10,height=2,fg="black", bg="light grey")
trinning.place(relx = 0.2, rely = 0.3, anchor=CENTER)
testing=Button(root,text="Testing",command=testing,width=10,height=2,fg="black", bg="light grey")
testing.place(relx = 0.5, rely = 0.3, anchor = CENTER)
predcting=Button(root,text="Predcting",command=predcting ,width=10,height=2,fg="black", bg="light grey")
predcting.place(relx = 0.8, rely = 0.3, anchor = CENTER)
root.geometry("500x500")
root.title("MOVIE GENRE DETECTION SYSTEM")
root.mainloop()
