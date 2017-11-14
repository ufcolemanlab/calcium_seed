# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 12:01:23 2017

@author: jesse
"""

import cv2 as cv2
from skimage import io
import tkFileDialog
import Tkinter as tk
import tifffile
import numpy as np
from matplotlib import pyplot as plt
#import numpy as np

corners = [[0,0],[0,0]]
zoom_coords = [[0,0],[0,0]]
default_zoom = None
frame = None
frame_title = "calGUI"
#video trackbar
def on_slider_move(event):
    index = cv2.getTrackbarPos('frame', "calGUI")
    frame = im[index]#[corners[0][0]:corners[0][0]][corners[1][0]:corners[1][1]]
    cv2.imshow('calGUI', frame)

def selectCorners(event,x,y,flags,param):
    global corners, zoom_coords
    
    if len(corners) == 2:
        corners = []
    
    if event == cv2.EVENT_LBUTTONDOWN and len(corners)<2:
        corners.append([x,y])
        
    elif event == cv2.EVENT_LBUTTONUP and len(corners)<2:
        corners.append([x,y])
        zoom_coords = [ [ zoom_coords[0][0] + corners[0][0], zoom_coords[0][1] + corners[0][1] ] , [ zoom_coords[0][0] + x, zoom_coords[0][1] + y] ]
        print "corners: " + str(corners)
        if corners[1][0] - corners[0][0] > 10 and corners[1][1] - corners[0][1] > 10:
            cv2.imshow(frame_title,frame[zoom_coords[0][0]:zoom_coords[1][0],zoom_coords[0][1]:zoom_coords[1][1]])
    
    elif event == cv2.EVENT_RBUTTONUP:
        zoom_coords = [[0,0],[0,0]]
        
    pass

def plotSeries(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #plt.plot(im[:,x,y])
        pass
        

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    plot_fig = plt.figure()
    ax = plot_fig.add_subplot(111)
    #ax.plot(im[:,int(event.xdata),int(event.ydata)])
    seed = im[:,int(event.ydata),int(event.xdata)]
    correlations = np.zeros((1024,1024))
    for i in range(1024):
        for j in range(1024):
            target = im[:, i, j]
            correlations[i][j] = np.corrcoef(seed, target)[0][1]
    
    ax.imshow(correlations)


if __name__ == "__main__":
    #read in image
    root = tk.Tk()
    root.withdraw()
    filename = tkFileDialog.askopenfilename()
    root.destroy()
    
    im = io.imread(filename)
    print im.shape
    print im.dtype
    
    #create the window and trackbar
    cv2.namedWindow('calGUI', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('calGUI', 512,512)
    cv2.createTrackbar('frame', 'calGUI', 0, im.shape[0] - 1, on_slider_move)
#    cv2.namedWindow('avg')
    
    frame = im[0]
    cv2.imshow('calGUI', frame)
    avg = im[:,:,:].mean(axis = 0)
#    cv2.imshow('avg', avg)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(avg, cmap = 'gray')
    
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    cv2.setMouseCallback(frame_title,plotSeries)
#    
    corners = [[0,0],[frame.shape[0],frame.shape[1]]]
    
    
    #main loop
    while(1):
        #cv2.imshow('img', im[0])
        k = cv2.waitKey(33)
        if k == 27:
            break
        elif k == -1:
            continue

    #close openCV
    cv2.destroyAllWindows()