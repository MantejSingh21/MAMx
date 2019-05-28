import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import sys
import glob
import PIL


filelist = ['C_0001_1.RIGHT_MLO.LJPEG.1_highpass.jpg']
for imagefile in filelist:
    im=Image.open(imagefile)
    box=(240, 240, 300, 300)
    im_crop=im.crop(box)
    im_crop.show()

A='C_0001_1.RIGHT_MLO.LJPEG.1_highpass.jpg'
B='C_0001_1.LEFT_MLO.LJPEG.1_highpass.jpg'
C='C_0001_1.RIGHT_CC.LJPEG.1_highpass.jpg'
D='C_0001_1.LEFT_CC.LJPEG.1_highpass.jpg'


Image.open('C_0001_1.RIGHT_MLO.LJPEG.1_highpass.gif').convert('RGB').save(A)
Image.open('C_0002_1.LEFT_MLO.LJPEG.1_highpass.gif').convert('RGB').save(B)
Image.open('C_0003_1.RIGHT_CC.LJPEG.1_highpass.gif').convert('RGB').save(C)
Image.open('C_0004_1.LEFT_CC.LJPEG.1_highpass.gif').convert('RGB').save(D)

#Crop and resize image
img1 = cv2.imread(A)
resized_img1 = cv2.resize(img1, (240, 300))

img2 = cv2.imread(B)
resized_img2 = cv2.resize(img2, (230, 300))

img3 = cv2.imread(C)
resized_img3 = cv2.resize(img3, (230, 300))

img4 = cv2.imread(D)
resized_img4 = cv2.resize(img4, (230, 300))

# Renaming the images
img_L_CC = cv2.imread(D, cv2.IMREAD_GRAYSCALE)
img_L_CC = cv2.resize(img_L_CC, (240, 300))
img_L_MLO = cv2.imread(B, cv2.IMREAD_GRAYSCALE)
img_L_MLO = cv2.resize(img_L_MLO, (240, 300))
img_R_CC = cv2.imread(C, cv2.IMREAD_GRAYSCALE)
img_R_CC= cv2.resize(img_R_CC, (240, 300))
img_R_MLO = cv2.imread(D, cv2.IMREAD_GRAYSCALE)
img_R_MLO = cv2.resize(img_R_MLO, (240, 300))

# Calculating mean of non-black pixels, taking into account 1.2x for cancer markers
# need to figure out how to only calculate the intensity using non black pixels
average_color_per_row_L_CC = np.average(img_L_CC, axis=0) 
#print (average_color_per_row_L_CC)
average_color_per_row_L_CC = np.average(average_color_per_row_L_CC, axis=0)
average_color_L_CC = np.uint8(average_color_per_row_L_CC)
#print (average_color_L_CC)
cancer_marker_intensity = average_color_L_CC * 1.2
print(cancer_marker_intensity)

average_color_per_row_R_CC = np.average(img_R_CC, axis=0) 
#print (average_color_per_row_R_CC)
average_color_per_row_R_CC = np.average(average_color_per_row_R_CC, axis=0)
average_color_R_CC = np.uint8(average_color_per_row_R_CC)
#print (average_color_R_CC)
cancer_marker_intensity = average_color_R_CC * 1.2
print(cancer_marker_intensity)

average_color_per_row_L_MLO = np.average(img_L_MLO, axis=0) 
#print (average_color_per_row_L_MLO)
average_color_per_row_L_MLO = np.average(average_color_per_row_L_MLO, axis=0)
average_color_L_MLO = np.uint8(average_color_per_row_L_MLO)
#print (average_color_L_MLO)
cancer_marker_intensity = average_color_L_MLO * 1.2
print(cancer_marker_intensity)

average_color_per_row_R_MLO = np.average(img_R_MLO, axis=0) 
#print (average_color_per_row_R_MLO)
average_color_per_row_R_MLO = np.average(average_color_per_row_R_MLO, axis=0)
average_color_R_MLO = np.uint8(average_color_per_row_R_MLO)
#print (average_color_R_MLO)
cancer_marker_intensity = average_color_R_MLO * 1.2
print(cancer_marker_intensity)

#Begin blob.detection() parameters
kernel = np.ones((5,5),np.uint8)

dilation_L_CC = cv2.dilate(img_L_CC,kernel,iterations = 500)
erosion_L_CC = cv2.erode(img_L_CC,kernel,iterations = 500)
closing_L_CC = cv2.morphologyEx(img_L_CC, cv2.MORPH_CLOSE, kernel)

dilation_L_MLO = cv2.dilate(img_L_MLO,kernel,iterations = 500)
erosion_L_MLO = cv2.erode(img_L_MLO,kernel,iterations = 500)
closing_L_MLO = cv2.morphologyEx(img_L_MLO, cv2.MORPH_CLOSE, kernel)

dilation_R_CC = cv2.dilate(img_R_CC,kernel,iterations = 500)
erosion_R_CC = cv2.erode(img_R_CC,kernel,iterations = 500)
closing_R_CC = cv2.morphologyEx(img_R_CC, cv2.MORPH_CLOSE, kernel)

dilation_R_MLO = cv2.dilate(img_R_MLO,kernel,iterations = 500)
erosion_R_MLO = cv2.erode(img_R_MLO,kernel,iterations = 500)
closing_R_MLO = cv2.morphologyEx(img_R_MLO, cv2.MORPH_CLOSE, kernel)

retval_L_CC, threshold_L_CC = cv2.threshold(img_L_CC, 180, 255, cv2.THRESH_BINARY_INV) 
params_L_CC = cv2.SimpleBlobDetector_Params()

retval_L_MLO, threshold_L_MLO = cv2.threshold(img_L_MLO, 180, 255, cv2.THRESH_BINARY_INV)
params_L_MLO = cv2.SimpleBlobDetector_Params()

retval_R_CC, threshold_R_CC = cv2.threshold(img_R_CC, 180, 255, cv2.THRESH_BINARY_INV)
params_R_CC = cv2.SimpleBlobDetector_Params()

retval_R_MLO, threshold_R_MLO = cv2.threshold(img_R_MLO, 180, 255, cv2.THRESH_BINARY_INV)
params_R_MLO = cv2.SimpleBlobDetector_Params()

params_L_CC.minThreshold = 200;
params_L_CC.maxThreshold = 255;

params_L_MLO.minThreshold = 200;
params_L_MLO.maxThreshold = 255;

params_R_CC.minThreshold = 200;
params_R_CC.maxThreshold = 255;

params_R_MLO.minThreshold = 200;
params_R_MLO.maxThreshold = 255;

blur_L_CC = cv2.GaussianBlur(img_L_CC,(5,5),0)
blur_L_MLO = cv2.GaussianBlur(img_L_MLO,(5,5),0)
blur_R_CC = cv2.GaussianBlur(img_R_CC,(5,5),0)
blur_R_MLO = cv2.GaussianBlur(img_R_MLO,(5,5),0)

params_L_CC.filterByCircularity = False
#params.minCircularity = 0.1
params_L_MLO.filterByCircularity = False
#params.minCircularity = 0.1
params_R_CC.filterByCircularity = False
#params.minCircularity = 0.1
params_R_MLO.filterByCircularity = False
#params.minCircularity = 0.1

params_L_CC.filterByArea = True;
params_L_CC.minArea = 300;
params_L_MLO.filterByArea = True;
params_L_MLO.minArea = 300;
params_R_CC.filterByArea = True;
params_R_CC.minArea = 300;
params_R_MLO.filterByArea = True;
params_R_MLO.minArea = 300;

params_L_CC.filterByConvexity = False
#params.minConvexity = 0.01
params_L_MLO.filterByConvexity = False
#params.minConvexity = 0.01
params_R_CC.filterByConvexity = False
#params.minConvexity = 0.01
params_R_MLO.filterByConvexity = False
#params.minConvexity = 0.01


params_L_CC.filterByInertia = False
#params.minInertiaRatio = 0.50
params_L_MLO.filterByInertia = False
#params.minInertiaRatio = 0.50
params_R_CC.filterByInertia = False
#params.minInertiaRatio = 0.50
params_R_MLO.filterByInertia = False
#params.minInertiaRatio = 0.50

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector_L_CC = cv2.SimpleBlobDetector(params_L_CC)
    detector_L_MLO = cv2.SimpleBlobDetector(params_L_MLO)
    detector_R_CC = cv2.SimpleBlobDetector(params_R_CC)
    detector_R_MLO = cv2.SimpleBlobDetector(params_R_MLO)
else :
    detector_L_CC = cv2.SimpleBlobDetector_create(params_L_CC)
    detector_L_MLO = cv2.SimpleBlobDetector_create(params_L_MLO)
    detector_R_CC = cv2.SimpleBlobDetector_create(params_R_CC)
    detector_R_MLO = cv2.SimpleBlobDetector_create(params_R_MLO)

keypoints_L_CC = detector_L_CC.detect(threshold_L_CC)
keypoints_L_MLO = detector_L_MLO.detect(threshold_L_MLO)
keypoints_R_CC = detector_R_CC.detect(threshold_R_CC)
keypoints_R_MLO = detector_R_MLO.detect(threshold_R_MLO)


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
img_L_CC_with_keypoints = cv2.drawKeypoints(img_L_CC, keypoints_L_CC, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_L_MLO_with_keypoints = cv2.drawKeypoints(img_L_MLO, keypoints_L_MLO, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_R_CC_with_keypoints = cv2.drawKeypoints(img_R_CC, keypoints_R_CC, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_R_MLO_with_keypoints = cv2.drawKeypoints(img_R_MLO, keypoints_R_MLO, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Show keypoints
cv2.line(img_L_CC_with_keypoints,(120,0),(120,300),(255,0,0),1)
cv2.line(img_L_CC_with_keypoints,(0,150),(240,150),(255,0,0),1)

cv2.line(img_L_MLO_with_keypoints,(120,0),(120,300),(255,0,0),1)
cv2.line(img_L_MLO_with_keypoints,(0,150),(240,150),(255,0,0),1)

cv2.line(img_R_CC_with_keypoints,(120,0),(120,300),(255,0,0),1)
cv2.line(img_R_CC_with_keypoints,(0,150),(240,150),(255,0,0),1)

cv2.line(img_R_MLO_with_keypoints,(120,0),(120,300),(255,0,0),1)
cv2.line(img_R_MLO_with_keypoints,(0,150),(240,150),(255,0,0),1)

#begin pixel counting process
im1 = Image.open('C_0019_1.RIGHT_MLO.LJPEG.1_highpass.jpg')
possible_cancer_marker_1 = 0
normal_1= 0
possible_cancer_marker_1_RMLO = 0
normal_1_MLO_R= 0

im2 = Image.open('C_0019_1.LEFT_MLO.LJPEG.1_highpass.jpg')
possible_cancer_marker_2 = 0
normal_2= 0
possible_cancer_marker_2_LMLO = 0
normal_2_MLO_L= 0

im3 = Image.open('C_0019_1.RIGHT_CC.LJPEG.1_highpass.jpg')
possible_cancer_marker_3 = 0
normal_3= 0
possible_cancer_marker_3_RCC = 0
normal_3_CC_R= 0

im4 = Image.open('C_0019_1.LEFT_CC.LJPEG.1_highpass.jpg')
possible_cancer_marker_4 = 0
normal_4= 0
possible_cancer_marker_4_LCC = 0
normal_4_CC_L= 0

for pixel in im1.getdata():
    if pixel >= (225, 225, 225):
        possible_cancer_marker_1_RMLO += 1

    else:
        normal_1 += 1
print('No. of possible cancer markers=' + str(possible_cancer_marker_1) + " & normal cells=" + str(normal_1))

## This should now take the average values from 1st calculation and then detect pixels accordingly 
"""for pixel in im1.getdata():
    if pixel >= (130, 130, 130) and pixel < (210,210,210):
        possible_cancer_marker_1_MLO_R += 1

    else:
        normal_1_MLO_R += 1
"""
for pixel in im2.getdata():
    if pixel >= (225, 225, 225): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
        possible_cancer_marker_2_LMLO += 1
        
    else:
        normal_2 += 1
print('No. of possible cancer markers=' + str(possible_cancer_marker_2) + " & normal cells=" + str(normal_2))
"""
for pixel in im2.getdata():
    if pixel >= (130, 130, 130) and pixel < (210,210,210):
        possible_cancer_marker_2_MLO_L += 1

    else:
        normal_2_MLO_L += 1
"""
for pixel in im3.getdata():
    if pixel >= (225, 225, 225): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
        possible_cancer_marker_3_RCC += 1
        
    else:
        normal_3 += 1
print('No. of possible cancer markers=' + str(possible_cancer_marker_3) + " & normal cells=" + str(normal_3))
"""
for pixel in im3.getdata():
    if pixel >= (130, 130, 130) and pixel < (210,210,210):
        possible_cancer_marker_3_CC_R += 1

    else:
        normal_3_CC_R += 1

"""
for pixel in im4.getdata():
    if pixel >= (225, 225, 225): # if your image is RGB (if RGBA, (0, 0, 0, 255) or so
        possible_cancer_marker_4_LCC += 1
        
    else:
        normal_4 += 1
print('No. of possible cancer markers=' + str(possible_cancer_marker_4) + " & normal cells=" + str(normal_4))
"""
for pixel in im4.getdata():
    if pixel >= (130, 130, 130) and pixel < (210,210,210):
        possible_cancer_marker_4_CC_L += 1


    else:
        normal_4_CC_L += 1
"""
X = int(possible_cancer_marker_1_RMLO) - int(possible_cancer_marker_3_RCC)
Y = int(possible_cancer_marker_2_LMLO) - int(possible_cancer_marker_4_LCC)
print ("Bleed effect of MLO view "  +str(X))
print ("Bleed effect of CC view "     + str(Y)) 
#Start of filter 1 - Thresholding of white pixels
#Below code is used to write on the images
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(resized_img1,'PCM' + str(possible_cancer_marker_1) + " NC=" + str(normal_1),(0,30), font, 0.6, (200,255,155), 2, cv2.LINE_AA)
cv2.putText(resized_img2,'PCM' + str(possible_cancer_marker_2) + " NC=" +str(normal_2),(0,30), font, 0.6, (200,255,155), 2, cv2.LINE_AA)
cv2.putText(resized_img3,'PCM' + str(possible_cancer_marker_3) + " NC=" + str(normal_3),(0,30), font, 0.6, (200,255,155), 2, cv2.LINE_AA)
cv2.putText(resized_img4,'PCM' + str(possible_cancer_marker_4) + " NC=" + str(normal_4),(0,30), font, 0.6, (200,255,155), 2, cv2.LINE_AA)


#count the number of pixels and if they are >= 1500, state possible cancer detection
if possible_cancer_marker_1 >= (1500):
     cv2.putText(resized_img1,'cancer detected' ,(10,60), font, 0.6, (200,255,155), 2, cv2.LINE_AA)
if possible_cancer_marker_2 >= (1500):
     cv2.putText(resized_img2,'cancer detected' ,(10,60), font, 0.6, (200,255,155), 2, cv2.LINE_AA)
if possible_cancer_marker_3 >= (1500):
     cv2.putText(resized_img3,'cancer detected' ,(10,60), font, 0.6, (200,255,155), 2, cv2.LINE_AA)
if possible_cancer_marker_4 >= (1500):
     cv2.putText(resized_img4,'cancer detected' ,(10,60), font, 0.6, (200,255,155), 2, cv2.LINE_AA)     
image1 = np.zeros((400,400,3), dtype="uint8")
image1[np.where((image1>=[225,225,225]).all(axis=2))] = [0,0,255]
image2 = np.zeros((400,400,3), dtype="uint8")
image3 = np.zeros((400,400,3), dtype="uint8")
image4 = np.zeros((400,400,3), dtype="uint8")


image1[np.where((image1==[225,225,225]).all(axis=2))] = [255,255,255]
image2[np.where((image2==[225,225,225]).all(axis=2))] = [255,255,255]
image3[np.where((image3==[225,225,225]).all(axis=2))] = [255,255,255]
image4[np.where((image4==[225,225,225]).all(axis=2))] = [255,255,255]       

retval1, threshold = cv2.threshold(resized_img1, 225, 255, cv2.THRESH_BINARY)
grayscaled1 = cv2.cvtColor(resized_img1,cv2.COLOR_BGR2GRAY)
retval1_2, threshold1 = cv2.threshold(grayscaled1, 225, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

retval2, threshold = cv2.threshold(resized_img2, 225, 255, cv2.THRESH_BINARY)
grayscaled2 = cv2.cvtColor(resized_img2,cv2.COLOR_BGR2GRAY)
retval2_1, threshold2 = cv2.threshold(grayscaled2, 225, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

retval3, threshold = cv2.threshold(resized_img3, 225, 255, cv2.THRESH_BINARY)
grayscaled3 = cv2.cvtColor(resized_img3,cv2.COLOR_BGR2GRAY)
retval3_1, threshold3 = cv2.threshold(grayscaled3, 225, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

retval4, threshold = cv2.threshold(resized_img4, 225, 255, cv2.THRESH_BINARY)
grayscaled4 = cv2.cvtColor(resized_img4,cv2.COLOR_BGR2GRAY)
retval4_1, threshold4 = cv2.threshold(grayscaled4, 225, 255, cv2.THRESH_BINARY)
gaus = cv2.adaptiveThreshold(grayscaled4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)



#Provides breast boundary detection

retval1_3,otsu = cv2.threshold(grayscaled1,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval2_3,otsu = cv2.threshold(grayscaled2,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval3_3,otsu = cv2.threshold(grayscaled3,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval4_3,otsu = cv2.threshold(grayscaled4,225,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


from tkinter import *
def analysis () :
   mlabel1 = Label (mGui , text = ' BREAST CANCER DETECTED: PLEASE FOLLOW UP WITH DOCTOR ' ) . pack ()
mGui = Tk()
mGui.geometry ( '600x300+400+200')
mGui.title ('MAMX')
mlabel = Label (mGui, text = 'BREAST CANCER DETECTED: REFER FOR FURTHER CHECK UP', font = ('ARIAL ',12),).pack()
#mbutton = Button(mGui, text = 'BREAST CANCER DETECTED: REFER FOR FURTHER CHECK UP' , command = analysis , fg = 'red' ) .pack()

cv2.imshow("img_L_CC", img_L_CC_with_keypoints)
cv2.imshow("img_L_MLO", img_L_MLO_with_keypoints)
cv2.imshow("img_R_CC", img_R_CC_with_keypoints)
cv2.imshow("img_R_mlo", img_R_MLO_with_keypoints)




cv2.waitKey(0)
cv2.destroyAllWindows()

#Plotting a histogram to detect the intensity of white pixels
#plt.hist(resized_img.ravel(),256,[0,256]);

#cv2.imshow ('MLO RIGHT 1', resized_img1 )
#cv2.imshow('threshold1',threshold1)
#cv2.imshow ('MLO LEFT 1', resized_img2 )
#cv2.imshow('threshold2',threshold2)
#cv2.imshow ('CC RIGHT 1', resized_img3 )
#cv2.imshow('threshold3',threshold3)
#cv2.imshow ('CC LEFT 1', resized_img4 )
#cv2.imshow('threshold4',threshold4)
#cv2.imshow("img_L_CC", img_L_CC_with_keypoints)
#cv2.imshow("img_L_MLO", img_L_MLO_with_keypoints)
#cv2.imshow("img_R_CC", img_R_CC_with_keypoints)
#cv2.imshow("img_R_mlo", img_R_MLO_with_keypoints) 
#cv2.imshow('threshold',threshold)
#cv2.imshow('gaus',gaus)
#cv2.imshow('otsu',otsu)
#plt.show()
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

#End of filter 1 = Thresholding of 1 white pixels
#Filter 2 - Should be able to count the density of white pixels and determine if the tissue is abnormal of normal. Not diagnosis or prediction yet

A = +1
b = +1
C = +1
D = +1


