import cv2


img = cv2.imread(".\image.jpg", cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
cv2.imshow('image', img)
cv2.waitKey(0)

#extracting Red
blue,green,red = cv2.split(img)
cv2.imshow('red', red)
cv2.waitKey(0)


# convert to greyscale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray', img_gray)
# cv2.waitKey(0)

#blur
img_blur = cv2.GaussianBlur(red, (5,5),10)
cv2.imshow('img_blur',img_blur)
cv2.waitKey(0)

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X
# cv2.imshow('Sobel X', sobelx)
# cv2.waitKey(0)
# cv2.imshow('Sobel Y', sobely)
# cv2.waitKey(0)
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)


edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)


edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=175) # Canny Edge Detection
cv2.imshow('Canny Edge Detection1', edges)
cv2.waitKey(0)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=150) # Canny Edge Detection
cv2.imshow('Canny Edge Detection2', edges)
cv2.waitKey(0)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=125) # Canny Edge Detection
cv2.imshow('Canny Edge Detection3', edges)
cv2.waitKey(0)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=50) # Canny Edge Detection
cv2.imshow('Canny Edge Detection4', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()

