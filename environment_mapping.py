import math
import numpy as np
from PIL import Image
import cv2

def main():
    
    fpath='r000.png'
    img=Image.open(fpath).convert('RGB')

    igs=np.array(img)

    h,w,_=igs.shape

    ## step 1. cylindrical projection of image
    igs_warp=np.zeros((h,2*w,3))
    r=int(w/2)
    h2=int(h/2)
    for i in range(r,r+w):
      dh=min(h2,int(math.sqrt(r*r-(w-i)*(w-i))))
      for j in range(h2-dh,h2+dh):
        y=int(h*(j-h2+dh)/float(2*dh))
        if y>=h:
          y=h-1
        igs_warp[j,i]=igs[y,i-r]


    ## step 2. copy paste both side of image
    for i in range(0,r):
      for j in range(0,h):
        igs_warp[j,i]=igs_warp[j,w-i]
    for i in range(r+w,2*w):
      for j in range(0,h):
        igs_warp[j,i]=igs_warp[j,3*w-i]


    ## step 3. fill empty space of image
    pix=np.zeros((2*w,2))

    for i in range(0,2*w):
      if igs_warp[0,i,0]==0 and igs_warp[0,i,1]==0 and igs_warp[0,i,2]==0:
        for j in range(0,h2):
          if igs_warp[j,i,0]!=0 or igs_warp[j,i,1]!=0 or igs_warp[j,i,2]!=0:
            pix[i,0]=j
            break
        for j in range(0,h2):
          if igs_warp[j,i,0]!=0 or igs_warp[j,i,1]!=0 or igs_warp[j,i,2]!=0:
            break
          igs_warp[j,i]=igs_warp[int(pix[i,0]),i]
      else:
        pix[i,0]=0

    for i in range(0,2*w):
      if igs_warp[h-1,i,0]==0 and igs_warp[h-1,i,1]==0 and igs_warp[h-1,i,2]==0:
        for j in range(h-1,h2,-1):
          if igs_warp[j,i,0]!=0 or igs_warp[j,i,1]!=0 or igs_warp[j,i,2]!=0:
            pix[i,1]=j
            break
        for j in range(h-1,h2,-1):
          if igs_warp[j,i,0]!=0 or igs_warp[j,i,1]!=0 or igs_warp[j,i,2]!=0:
            break
          igs_warp[j,i]=igs_warp[int(pix[i,1]),i]
      else:
        pix[i,1]=h-1
    for i in range(0,2*w):
      if igs_warp[0,i,0]==0 and igs_warp[0,i,1]==0 and igs_warp[0,i,2]==0:
        igs_warp[:,i]=igs_warp[:,i-1]
        pix[i,0]=h-1
        pix[i,1]=h
    

    ## step 4. blur filled space
    kernel1d=cv2.getGaussianKernel(5,3)
    kernel2d=np.outer(kernel1d,kernel1d.transpose())
    
    tmp=cv2.filter2D(igs_warp,-1,kernel2d)
    for i in range(0,2*w):
      for j in range(0,int(pix[i,0])):
        igs_warp[j,i]=tmp[j,i]
      for j in range(h-1,int(pix[i,1]),-1):
        igs_warp[j,i]=tmp[j,i]
    

    igs_warp=igs_warp[:,r:w+r]
    img_warp=Image.fromarray(igs_warp.astype(np.uint8))

    img_warp.save('environment_map.png')

if __name__ == '__main__':
    main()