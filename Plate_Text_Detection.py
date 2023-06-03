import cv2 as cv
import numpy as np
import easyocr

car_img =cv.imread("car-plate.jpg")
plate_img =cv.imread("plate_img.jpg",0)
#specify language
reader=easyocr.Reader(['fa'])
#read img text
plate_info=reader.readtext(plate_img)
print(plate_info)
#exact detecting and classificaiton plate text
plate_text=plate_info[0][-2]
#show plate img
cv.imshow("plate", plate_img)
cv.waitKey(0)
