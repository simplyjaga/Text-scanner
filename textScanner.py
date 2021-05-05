import cv2 as cv 
import numpy as np 
from tkinter.filedialog import askopenfilename
import pytesseract as pyt

filename = askopenfilename()
image = cv.imread(filename)

img = np.copy(image)

# detect the edges
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.GaussianBlur(img_gray,(5,5),0)

#Edge detection
img_thresh = cv.Canny(img_gray,100,200)

#finding the contour
contours, hierarchy = cv.findContours(img_thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

if (len(contours) != 0) :
    areas = [cv.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    #drawing contours on original image
    # way 1
    # cv.drawContours(img, [cnt], -1 ,(0,255,0),3)
    # way 2
    # x,y,w,h = cv.boundingRect(cnt)
    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    # way 3
    epsilon = 0.05 * cv.arcLength(cnt,True)
    polygon = cv.approxPolyDP(cnt,epsilon,True)
    cv.drawContours(img,[polygon],-1,(0,255,0), 3)

    #find the perspective matrix and transform the image
    src = np.float32([polygon [0,0],polygon [1,0],polygon [2,0],polygon [3,0]]) 
    dst = np.float32([[0,0],[0,300],[400,300],[400,0]])

    M = cv.getPerspectiveTransform(src,dst)
    warped = cv.warpPerspective(img,M,(400,300))

    #preprocess the image for ocr
    warped = cv.resize(warped,None,fx=1,fy=1, interpolation = cv.INTER_CUBIC)

    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    warped_gray = cv.GaussianBlur(warped_gray,(5,5),0)

    # bool, warped_thresh = cv.threshold(warped_gray,200,255,cv.THRESH_BINARY_INV)
    
    warped_thresh = cv.adaptiveThreshold(warped_gray,255,
                        cv.ADAPTIVE_THRESH_MEAN_C,
                        cv.THRESH_BINARY,11,2)

cv.imshow("Originalimage",image)
cv.imshow("wrapedimage",warped_thresh)

#reading text
text = pyt.image_to_string(warped_thresh)
print(text)

cv.waitKey(0)
cv.destroyAllWindows()