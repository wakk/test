# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:45:01 2020

@author: Administrator
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


"""
bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
camera = cv2.VideoCapture(r'E:\car.mp4')
while True:
    ret, frame = camera.read()
    if ret==False:
        break
    fgmask = bs.apply(frame)
    fg2 = fgmask.copy()
    th = cv2.threshold(fg2,100,255,cv2.THRESH_BINARY)[1]
    th=cv2.erode(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=2)
    dilated = cv2.dilate(th,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 2)
    contours, hier = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 50:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            
    cv2.imshow("mog",th)
    cv2.imshow("thresh",dilated)
    cv2.imshow("detection",frame)
    if cv2.waitKey(24) & 0xff == 27:
        break
camera.release()
cv2.destroyAllWindows()
"""
#识别车道范围
cap = cv2.VideoCapture(r'E:/car.mp4')

"""
cap = cv2.VideoCapture(r'E:/car.mp4')
kernel1 = np.ones((3,3),np.uint8)
kernel2 = np.ones((9,9),np.uint8)
kernel3 = np.ones((17,17),np.uint8)
fgbg = cv2.createBackgroundSubtractorKNN()
while(1):
    ret, frame = cap.read()
    if ret==False:
        break
    gray=cv2.GaussianBlur(frame,(31,31), 0)#高斯模糊
    fgmask = fgbg.apply(gray)#生成蒙版
    ret,imBin=cv2.threshold(fgmask,125, 255, cv2.THRESH_BINARY)#二值化
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernel2)#开
    mask2 =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernel3)#膨胀
    
    contours, hier = cv2.findContours(mask2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #画矩形
    for c in contours:
        if 200<cv2.contourArea(c):
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    cv2.imshow('mask',mask)
    cv2.imshow('imBin',imBin)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
"""





















