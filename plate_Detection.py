import cv2 as cv
import numpy as np
import imutils

car_img =cv.imread("car-plate.jpg")
car_img_gray =cv.cvtColor(car_img, cv.COLOR_RGB2GRAY)
blateral_filtered=cv.bilateralFilter(car_img_gray,31,15,15)
edges=cv.Canny(blateral_filtered,30,200)
contours=cv.findContours(edges.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
contours_refined =imutils.grab_contours(contours)
contours_sorted=sorted(contours_refined,key=cv.contourArea,reverse=True)[:4]
for contours in contours_sorted:
    contour_approx=cv.approxPolyDP(contours,10,True)
    if len(contour_approx)==4:
        plate_location=contour_approx
        break
plate_mask0=np.zeros(car_img_gray.shape,np.uint8)
plate_mask=cv.drawContours(plate_mask0,[plate_location],0,225,-1)
plate_img=cv.bitwise_and(car_img,car_img,mask=plate_mask)

(x,y)=np.where(plate_mask==225)
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))
cropped_img=car_img_gray[x1:x2+1,y1:y2+1]

cv.imshow("plate", cropped_img)
cv.imwrite("plate_img.jpg",cropped_img)
cv.waitKey(0)

