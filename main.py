import cv2
import numpy as np
from pycparser.ply.yacc import resultlimit

leaves = cv2.imread('leaves.jpg',cv2.IMREAD_UNCHANGED)
logo = cv2.imread('logo.png',cv2.IMREAD_UNCHANGED)
print(leaves.shape)
print(logo.shape)

sm_logo = cv2.resize(logo,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
print(sm_logo.shape)

logo_bgr = sm_logo[:,:,0:3]
logo_alpha = sm_logo[:,:,3]

cv2.imshow('bgr',logo_bgr)
cv2.waitKey(0)
cv2.imshow('alpha',logo_alpha)
cv2.waitKey(0)

w = leaves.shape[1]
h = leaves.shape[0]

logoh= sm_logo.shape[0]
logow = sm_logo.shape[1]

cx = int(w/2)
cy = int(h/2)
print(cx,cy)

tlx = int(cx-logow/2)
tly = int(cy-logoh/2)
print(tlx,tly)

brx = int(cx+logow/2)
bry = int(cy+logoh/2)
print(brx,bry)

# crop the leaves image as per logo size
roi = leaves[tly:bry,tlx:brx]
print(roi.shape)

cv2.imshow('cropped leaves', roi)
cv2.waitKey(0)

# as leaves img has 3 channel we will need to make our mask 3 channel

inv_sm_logo = cv2.bitwise_not(logo_alpha)

cv2.imshow('invert alpha',inv_sm_logo)
cv2.waitKey(0)

masked_roi = cv2.bitwise_and(roi,roi,mask=inv_sm_logo)

cv2.imshow('masked roi',masked_roi)
cv2.waitKey(0)

masked_logo = cv2.bitwise_and(logo_bgr,logo_bgr,mask=logo_alpha)
cv2.imshow('masked logo',masked_logo)
cv2.waitKey(0)

result = cv2.bitwise_or(masked_roi,masked_logo)
cv2.imshow('reslut',result)
cv2.waitKey(0)


leavescopy = leaves.copy()

# now the final result
leavescopy[tly:bry,tlx:brx] = result
cv2.imshow('final reslut',leavescopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# now we will adjust the watermark opacity we just applied

roicopy = roi.copy()
leavescopy1 = leaves.copy()

logo_mask = cv2.merge([logo_alpha,logo_alpha,logo_alpha])
logo_clean = cv2.bitwise_and(logo_bgr,logo_mask)
# we can use masked alpha also, logo clean and masked alpha re same
watermarked = cv2.addWeighted(roicopy,1,logo_clean,0.9,0)

leavescopy1[tly:bry,tlx:brx] = watermarked

cv2.imshow('watermarked image',leavescopy1)
cv2.waitKey(0)
cv2.destroyAllWindows()