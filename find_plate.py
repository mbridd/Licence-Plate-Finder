##crude untuned licence plate finder

''' ideal implementation
    - yellow filtering

    - edge detection

    - though line transform
      tangent to plate edges
    - characters filtered

    - check potential "plates" for chars within 
      hough line transform of plate using contour hierarchy

    - hough line transform tangent to char top and bottom

    - check hough line transform of characters top and bottom
      are parallel to the hough line transform of plate edges

    - calc angle of plate

    - perspective transform plate asif facing
ideal implementation '''

''' actual implimentation
    - yellow filtering
    - edge detection
    - area filter
    - aspect ratio filter
    reflection of results:
    the program needs more "tests" such that the
    characteristics that define a number plate
    such as shape colors proportion relativity etc
    and calibrating as it doesnt fully define all plates and some
    show multiple "plates"
actual implimentation'''

import cv2
import os
import numpy as np
import sys

image_folder = os.getcwd()+"/images/" 
image = os.listdir(image_folder)
#setting hsv lower and upper ranges
lower_yellow = np.array([5, 50, 100])
upper_yellow = np.array([60, 255, 255])

for i in range(1,len(sys.argv)):
    
    car = cv2.imread(sys.argv[i])
    
##initial image conversions
##convert to greyscale
##convert grey scale to the hsv color pallete
##(provides easier color range limiting)
##range limit hsv image
##use hsv range limited as mask on original imag
##convert masked image to greyscale
##initial image conversions

    car_grey = cv2.cvtColor(car,cv2.COLOR_BGR2GRAY)
    car_hsv = cv2.cvtColor(car,cv2.COLOR_BGR2HSV)
    
    mask_hsv = cv2.inRange(car_hsv, lower_yellow, upper_yellow)
    y_filter = cv2.bitwise_and(car,car,mask=mask_hsv)
    y_filter_grey = cv2.cvtColor(y_filter, cv2.COLOR_BGR2GRAY)

    
##sobel edge detect yellow filter
##use sobel function to get sobel x and sobel y
##using a 3x3 kernel
##combine sobelx and sobely for better definition
    sobelx = cv2.Sobel(y_filter_grey,cv2.CV_16S,1,0,3,scale = 1, delta = 0,borderType = cv2.BORDER_DEFAULT)
    sobely = cv2.Sobel(y_filter_grey,cv2.CV_16S,0,1,3,scale = 1, delta = 0,borderType = cv2.BORDER_DEFAULT)
##wasnt able to shrink input arguements
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)

    sobel_filter = cv2.addWeighted(abs_sobelx,0.5,abs_sobely,0.5,0) 
##sobel edge detect yellow filter

##contour outline
##range mask to remove ill defined sections
##then outline defined sections

    ret,thresh = cv2.threshold(sobel_filter,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    ##contour area filter
    ##cycle through the contours
    ##check area and limit remove small sections
    ##convex hull remove imperfections
    ##compare sharper transisioned area with
    ##rough area
    ##using a rectangular boundry limit
    ##aspect ratio
    
    for h,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area>700:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            if solidity>0.9:
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if 2<=aspect_ratio<=15:
                    cv2.drawContours(car,[hull],0,(0,255,0),3)
    cv2.imshow('Car', car)

    cv2.waitKey(0) ##wait for set time didnt work in term

##close windows after each image and end of program
    cv2.destroyAllWindows()

