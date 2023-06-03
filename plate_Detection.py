import cv2 as cv
import numpy as np
import imutils

#import image
car_img =cv.imread("car-plate.jpg")
#make image gray
car_img_gray =cv.cvtColor(car_img, cv.COLOR_RGB2GRAY)
#blur image to decrese noise
blateral_filtered=cv.bilateralFilter(car_img_gray,31,15,15)
#take images edges
edges=cv.Canny(blateral_filtered,30,200)
#grouping images edges and find squares
contours=cv.findContours(edges.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#ignore edges details
contours_refined =imutils.grab_contours(contours)
#order contours
contours_sorted=sorted(contours_refined,key=cv.contourArea,reverse=True)[:4]
#approximate with a polygon
for contours in contours_sorted:
    contour_approx=cv.approxPolyDP(contours,10,True)
#if the shape have  4 sides
    if len(contour_approx)==4:
        plate_location=contour_approx
        break
#create a gray mask
plate_mask0=np.zeros(car_img_gray.shape,np.uint8)
#turn mask to white
plate_mask=cv.drawContours(plate_mask0,[plate_location],0,225,-1)
#fit mask to plate size
plate_img=cv.bitwise_and(car_img,car_img,mask=plate_mask)

#reading cordinate of plate
(x,y)=np.where(plate_mask==225)
#start point of img(left part)
(x1,y1)=(np.min(x),np.min(y))
#end points of img(right part)
(x2,y2)=(np.max(x),np.max(y))
#convert all points
cropped_img=car_img_gray[x1:x2+1,y1:y2+1]

#show img
cv.imshow("plate", cropped_img)
#save img
cv.imwrite("plate_img.jpg",cropped_img)
cv.waitKey(0)

