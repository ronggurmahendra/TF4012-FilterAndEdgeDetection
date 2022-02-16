import cv2
import numpy as np
import math
from clockwise_angle_and_distance import clockwise_angle_and_distance

img = cv2.imread(".\image.jpg", cv2.IMREAD_UNCHANGED)
#img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
print("Resolution : "+ str(len(img))+","+ str(len(img[0])))
cv2.imshow('image', img)
cv2.waitKey(0)

#extracting Red
blue,green,red = cv2.split(img)
cv2.imshow('red', red)
cv2.waitKey(0)


# convert to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_gray', img_gray)
cv2.waitKey(0)

#denoise menggunakan fastNlMeansDenoising
denoise_1 = cv2.fastNlMeansDenoising (img_gray,None,3,3,7) 
cv2.imshow('denoise_1 using fastNlMeansDenoising',denoise_1)
cv2.waitKey(0)

denoise_2 = cv2.fastNlMeansDenoising (img_gray,None,5,5,7) 
cv2.imshow('denoise_2 using fastNlMeansDenoising',denoise_2)
cv2.waitKey(0)

denoise_3 = cv2.fastNlMeansDenoising (img_gray,None,15,15,7)
cv2.imshow('denoise_3 using fastNlMeansDenoising',denoise_3)
cv2.waitKey(0)

# #denoise menggunakan GaussianBlur
denoise_1 = cv2.GaussianBlur (img_gray, (1,1),4)
cv2.imshow('denoise_1 using GaussianBlur',denoise_1)
cv2.waitKey(0)

denoise_2 = cv2.GaussianBlur (img_gray, (3,3),9)
cv2.imshow('denoise_2 using GaussianBlur',denoise_2)
cv2.waitKey(0)

denoise_3 = cv2.GaussianBlur (img_gray, (5,5),10)
cv2.imshow('denoise_3 using GaussianBlur',denoise_3)
cv2.waitKey(0)

# denoise menggunakan GaussianBlur
denoise_1 = cv2.GaussianBlur (img_gray, (1,1),4)
cv2.imshow('denoise_1 using GaussianBlur',denoise_1)
cv2.waitKey(0)

denoise_2 = cv2.GaussianBlur (img_gray, (3,3),9)
cv2.imshow('denoise_2 using GaussianBlur',denoise_2)
cv2.waitKey(0)

denoise_3 = cv2.GaussianBlur (img_gray, (5,5),10)
cv2.imshow('denoise_3 using GaussianBlur',denoise_3)
cv2.waitKey(0)



#blur (untuk menghilangkan bintik-bintik) untuk yang dipake dibawahnya
img_denoise = cv2.GaussianBlur(img_gray, (5,5),10) #gaussian Filter
cv2.imshow('img_denoise',img_denoise)
cv2.waitKey(0)


#testing edge extraraction algorithms
sobelx = cv2.Sobel(src=img_denoise, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge
sobely = cv2.Sobel(src=img_denoise, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge
sobelxy = cv2.Sobel(src=img_denoise, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X

cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)


#Testing threshold
edges = cv2.Canny(image=img_denoise, threshold1=100, threshold2=200) # Canny Edge Detection
# cv2.imshow('Canny Edge Detection', edges)
# cv2.waitKey(0)


edges = cv2.Canny(image=img_denoise, threshold1=100, threshold2=175) # Canny Edge Detection
# cv2.imshow('Canny Edge Detection1', edges)
# cv2.waitKey(0)

edges = cv2.Canny(image=img_denoise, threshold1=100, threshold2=150) # Canny Edge Detection
# cv2.imshow('Canny Edge Detection2', edges)
# cv2.waitKey(0)

edges = cv2.Canny(image=img_denoise, threshold1=100, threshold2=125) # Canny Edge Detection
# cv2.imshow('Canny Edge Detection3', edges)
# cv2.waitKey(0)

edges = cv2.Canny(image=img_denoise, threshold1=10, threshold2=35) # Canny Edge Detection
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

#merging contours reference source at : https://stackoverflow.com/questions/44501723/how-to-merge-contours-in-opencv#:~:text=go%20through%20the%20first%20contour,them%20into%20cv2%20contour%20format 
cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)[-2]
list_of_pts = [] 
for ctr in cnts:
    list_of_pts += [pt[0] for pt in ctr]

center_pt = np.array(list_of_pts).mean(axis = 0) # get origin
clock_ang_dist = clockwise_angle_and_distance(center_pt) # set origin
list_of_pts = sorted(list_of_pts, key=clock_ang_dist) # use to sort
ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
ctr = cv2.convexHull(ctr) # done.

#calculating curcumference and area
arclen = cv2.arcLength(ctr, True)
area = cv2.contourArea(ctr)

print("Circumference: {:.3f}\nArea: {:.3f}".format(arclen, area))

#drawing contour on image
temp = img.copy()

cv2.drawContours(temp, [ctr], -1, (0,255,0), 1, cv2.LINE_AA)
cv2.imshow('Contour', temp)
cv2.waitKey(0)

cv2.destroyAllWindows()

