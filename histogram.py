import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from matplotlib import cm
from skimage.transform import probabilistic_hough_line
from skimage import io, color

img = cv.imread("image_dataset/Images_ground/m01_s05.png")
#img = np.zeros((200,200), np.uint8)
#cv.rectangle(img, (0, 100), (200, 200), (255), -1)
#cv.rectangle(img, (0, 50), (100, 100), (127), -1)
#b, g, r = cv.split(img)
#cv.imshow("img", img)
#cv.imshow("b", b)
#cv.imshow("g", g)
#cv.imshow("r", r)

#plt.hist(b.ravel(), 256, [0, 256])
#plt.hist(g.ravel(), 256, [0, 256])
#plt.hist(r.ravel(), 256, [0, 256])

#hist = cv.calcHist([img], [0], None, [256], [0, 256])
#plt.plot(hist)
#plt.show()


ksize= (5,5)

b, g, r = cv.split(img)
plt.hist(g.ravel(), 256, [0, 256])
plt.show()
a = ((g>130) *255).astype(np.uint8)
c = ((g<130) *255).astype(np.uint8)
#c = cv.blur(c,ksize)
blurred_img = cv.GaussianBlur(c, (5, 5), 0)
#cv.imshow('cblur',c)
#cv.imshow('dark',c)
cv.imshow('blurred', blurred_img)

cv.imshow('light',a)

kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(blurred_img, cv.MORPH_OPEN, kernel)
#closing = cv.morphologyEx(a, cv.MORPH_CLOSE, kernel)
cv.imshow('open',opening)
#cv.imshow('close',closing)

edges = cv.Canny(c, 50, 180)
lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, img.shape[1]))
ax[2].set_ylim((img.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.show()




cv.waitKey(0)
cv.destroyAllWindows()