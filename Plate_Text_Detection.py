import cv2 as cv
import numpy as np
import easyocr

car_img =cv.imread("car-plate.jpg")
plate_img =cv.imread("plate_img.jpg",0)
reader=easyocr.Reader(['fa'])
plate_info=reader.readtext(plate_img)
print(plate_info)
plate_text=plate_info[0][-2]
cv.imshow("plate", plate_img)
