import cv2
import numpy as np

def demosaicing(img_path):
    img=cv2.imread(img_path)
    h,w,d=img.shape[0:3]
    thresh=cv2.inRange(img,np.array([240,240,240]),np.array([255,255,255]))
    kernel=np.ones((3,3),np.uint8)
    hi_mask=cv2.dilate(thresh,kernel)
    specular=cv2.inpaint(img,hi_mask,5,flags=cv2.INPAINT_TELEA)

    cv2.namedWindow('Image',0)
    cv2.resizeWindow('Image',int(w/2),int(h/2))
    cv2.imshow('Image',img)
    cv2.namedWindow('newimage',2)
    cv2.resizeWindow('newimage',int(w/2),int(h/2))
    cv2.imshow('newimage',specular)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    demosaicing('1.jpg')
