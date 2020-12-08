import math
import numpy as np
from PIL import Image
import cv2
import glob

datadir = './data/dataset'
resultdir='./results'

def main():
    
    for img_path in glob.glob(datadir+'/r*.*'):
      img=Image.open(img_path).convert('RGB')

      igs=np.array(img)
      h,w,_=igs.shape

      ## step 0. make height and width of image even for convenience
      if h%2==1:
        temp=np.zeros((h+1,w,3))
        temp[:h]=igs
        temp[h]=igs[h-1]
        igs=temp
        h+=1
      if w%2==1:
        temp=np.zeros((h,w+1,3))
        temp[:,:w]=igs
        temp[:,w]=igs[:,w-1]
        igs=temp
        w+=1


      ## step 1. make cylindrical projection of image
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
      kernel1d=cv2.getGaussianKernel(15,3)
      kernel2d=np.outer(kernel1d,kernel1d.transpose())
      
      tmp=cv2.filter2D(igs_warp,-1,kernel2d)
      for i in range(0,2*w):
        for j in range(0,int(pix[i,0])):
          igs_warp[j,i]=tmp[j,i]
        for j in range(h-1,int(pix[i,1]),-1):
          igs_warp[j,i]=tmp[j,i]
      
      
      img_warp=Image.fromarray(igs_warp.astype(np.uint8))
      img_warp.save(resultdir+'/environment_map/'+img_path[len(datadir)+1:])
      print(img_path,"DONE")

if __name__ == '__main__':
    main()