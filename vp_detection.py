
import math
import glob
import numpy as np
from PIL import Image, ImageDraw

datadir = './data/dataset'
resultdir='./results'
imagename = '/dataset/r016.PNG' # '/dataset/r*.*'

nLines = 64

sigma = 2
threshold_hough = 0.1
rhoRes = 1
thetaRes = math.pi / 1000

threshold_vp_r_lo = 16
threshold_vp_r_hi = 32
threshold_vp_cnt = 8

DRAW_HL = True
DRAW_VP = True

def ConvFilter(Igs, G):
    
    h, w = Igs.shape
    size = G.shape[0]
    halfsize = (size-1)//2

    Ipad = np.pad(Igs,((halfsize,halfsize),(halfsize,halfsize)))
    for i in range(halfsize):
        Ipad[i] = Ipad[halfsize]
        Ipad[h+halfsize+i] = Ipad[h+halfsize-1]
    for i in range(halfsize):
        Ipad[:,i] = Ipad[:,halfsize]
        Ipad[:,w+halfsize+i] = Ipad[:,w+halfsize-1]

    Iconv = np.zeros(Igs.shape)
    for i in range(size):
        for j in range(size):
            Iconv += Ipad[i:i+h,j:j+w]*G[i][j]

    return Iconv

def EdgeDetection(Igs, sigma=2):
    
    h, w = Igs.shape
    div_error = 1e-7

    size = 5
    halfsize = (size-1) // 2
    G = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            G[i][j] = np.exp(-((i - halfsize)**2 + (j - halfsize)**2) / 2.0 / (sigma ** 2))
    G = G / np.sum(G)

    Ismth = ConvFilter(Igs, G)
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix = ConvFilter(Ismth, Gx)
    Iy = ConvFilter(Ismth, Gy)
    Imth = np.sqrt(Ix * Ix + Iy * Iy)
    Io = np.arctan(Iy / (Ix + div_error))

    Bth = np.ones((h,w))
    Bth[0:h-1] *= (np.abs(Io[0:h-1]) < math.pi/4) | (Imth[0:h-1] >= Imth[1:h])
    Bth[1:h] *= (np.abs(Io[1:h]) < math.pi/4) | (Imth[1:h] > Imth[0:h-1])
    Bth[:,0:w-1] *= (np.abs(Io[:,0:w-1]) >= math.pi/4) | (Imth[:,0:w-1] >= Imth[:,1:w])
    Bth[:,1:w] *= (np.abs(Io[:,1:w]) >= math.pi/4) | (Imth[:,1:w] > Imth[:,0:w-1])
    Im = Imth * Bth

    return Im, Io, Ix, Iy

def HoughTransform(Im,threshold, rhoRes=1, thetaRes=math.pi/1000):
    
    h, w = Im.shape
    y, x = np.where(Im>=threshold)
    N = x.shape[0]

    tn = np.ceil(math.pi/thetaRes).astype(np.int32)
    rn = 2 * np.ceil(np.sqrt(h*h+w*w)/rhoRes).astype(np.int32)
    H = np.zeros((tn, rn))

    for i in range(tn):
        v, c = np.unique(np.floor((x*np.cos(thetaRes*i)+y*np.sin(thetaRes*i))/rhoRes).astype(np.int32),return_counts=True)
        H[i][v] += c

    return H

def HoughLines(H,rhoRes,thetaRes,nLines=32):
    
    rbase = H.shape[1] // 2
    thetasize = 35
    rhosize = 15
    thetahalfsize = (thetasize - 1) // 2
    rhohalfsize = (rhosize - 1) // 2
    Hth = H.copy()

    lRho = np.zeros(nLines)
    lTheta = np.zeros(nLines)
    for i in range(nLines):
        ind = np.unravel_index(np.argmax(Hth, axis=None), Hth.shape)
        if ind[1] >= rbase:
            lRho[i] = (ind[1]-H.shape[1]) * rhoRes
        else:
            lRho[i] = ind[1] * rhoRes
        lTheta[i] = ind[0] * thetaRes
        Hth[max(ind[0]-thetahalfsize,0):min(ind[0]+thetahalfsize+1,H.shape[0]),max(ind[1]-rhohalfsize,0):min(ind[1]+rhohalfsize+1,H.shape[1])] = -1

    return lRho, lTheta

def vp_detection(lRho, lTheta, threshold_r_lo=5, threshold_r_hi=25, threshold_cnt=5):

    vp = np.zeros((3, 2))
    for vp_dim in range(3):
        N = lRho.shape[0]
        vp_i, vp_j = -1, -1
        vp_match_max = -1

        for i in range(N):
            for j in range(i+1,N):
                try:
                    A = np.array([[np.sin(lTheta[i]), np.cos(lTheta[i])], [np.sin(lTheta[j]), np.cos(lTheta[j])]])
                    x, y = np.matmul(np.linalg.inv(A), np.array([[lRho[i]], [lRho[j]]]))
                    vp_rho = x * np.sin(lTheta) + y * np.cos(lTheta)
                    passing = (vp_rho <= lRho + threshold_r_lo) * (vp_rho >= lRho - threshold_r_lo)
                    passing_count = np.sum(passing)
                    if passing_count > vp_match_max:
                        vp_match_max = passing_count
                        vp_i, vp_j = i, j
                except:
                    continue
        
        if vp_match_max < threshold_cnt:
            vp = vp[0:vp_dim]
            break
        A = np.array([[np.sin(lTheta[vp_i]), np.cos(lTheta[vp_i])], [np.sin(lTheta[vp_j]), np.cos(lTheta[vp_j])]])
        x, y = np.matmul(np.linalg.inv(A), np.array([[lRho[vp_i]], [lRho[vp_j]]]))
        vp[vp_dim] = x, y
        vp_rho = x * np.sin(lTheta) + y * np.cos(lTheta)
        not_passing = (vp_rho > lRho + threshold_r_hi) + (vp_rho < lRho - threshold_r_hi)
        ind = np.where(not_passing)[0]
        lRho, lTheta = lRho[ind], lTheta[ind]

    if vp.shape[0] == 2 and np.abs(vp[0][0] - vp[1][0]) < np.abs(vp[0][1] - vp[1][1]) * np.tan(math.pi * 15 / 180):
        vertical = (lTheta > math.pi * 150 / 180) + (lTheta < math.pi * 30 / 180)
        ind = np.where(vertical)[0]
        lRho, lTheta = lRho[ind], lTheta[ind]
        N = lRho.shape[0]
        if N >=2:
            vp_i, vp_j = -1, -1
            vp_dist_min = 2147483647

            for i in range(N):
                for j in range(i+1,N):
                    try:
                        A = np.array([[np.sin(lTheta[i]), np.cos(lTheta[i])], [np.sin(lTheta[j]), np.cos(lTheta[j])]])
                        x, y = np.matmul(np.linalg.inv(A), np.array([[lRho[i]], [lRho[j]]]))
                        vp_rho = x * np.sin(lTheta) + y * np.cos(lTheta)
                        max_dist = np.max(np.abs(lRho - vp_rho))
                        if max_dist < vp_dist_min:
                            vp_dist_min = max_dist
                            vp_i, vp_j = i, j
                    except:
                        continue
            if vp_i >= 0:
                A = np.array([[np.sin(lTheta[vp_i]), np.cos(lTheta[vp_i])], [np.sin(lTheta[vp_j]), np.cos(lTheta[vp_j])]])
                x, y = np.matmul(np.linalg.inv(A), np.array([[lRho[vp_i]], [lRho[vp_j]]]))
                vp = np.concatenate((vp, np.array([x, y]).reshape((1,2))), axis=0)
    print(vp)
    return vp

def camera_info(Igs, vp):
    h, w = Igs.shape
    N = vp.shape[0]


    focal_length = None
    camera_direction = None

    if N != 3:
        return None, None, None
   
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    A[0] = vp[0] - vp[1]
    b[0] = np.sum(A[0] * vp[2])
    A[1] = vp[1] - vp[2]
    b[1] = np.sum(A[1] * vp[0])
    camera_direction = np.matmul(np.linalg.inv(A), b).reshape((2,))
    mid = (vp[0] + vp[1]) / 2
    fl_sq = np.sum((vp[0]-mid)*(vp[0]-mid))-np.sum((camera_direction-mid)*(camera_direction-mid))
    if fl_sq < 0:
        focal_length = 2147483647
    else:
        focal_length = np.sqrt(fl_sq)
    mid = (vp[1] + vp[2]) / 2
    fl_sq = np.sum((vp[1]-mid)*(vp[1]-mid))-np.sum((camera_direction-mid)*(camera_direction-mid))
    if fl_sq < 0 or np.abs(np.sqrt(fl_sq)-focal_length) > 1e-6:
        focal_length = 2147483647
    mid = (vp[2] + vp[0]) / 2
    fl_sq = np.sum((vp[2]-mid)*(vp[2]-mid))-np.sum((camera_direction-mid)*(camera_direction-mid))
    if fl_sq < 0 or np.abs(np.sqrt(fl_sq)-focal_length) > 1e-6:
        focal_length = 2147483647

    sorted_index = np.argsort(vp[:, 0])
    vp_z = None
    if vp[sorted_index[1]][0] - vp[sorted_index[0]][0] > vp[sorted_index[2]][0] - vp[sorted_index[1]][0]:
        vp_z = vp[sorted_index[0]]
    else:
        vp_z = vp[sorted_index[2]]
    
    camera_theta = np.arctan(np.sqrt(np.sum((vp_z - camera_direction)**2)) / focal_length)
    
    return camera_theta, camera_direction, focal_length

def main():

    # read images
    for img_path in glob.glob(datadir+'/r*.*'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        H = HoughTransform(Im,threshold_hough, rhoRes, thetaRes)
        lRho, lTheta = HoughLines(H,rhoRes,thetaRes,nLines)
        vp = vp_detection(lRho, lTheta, threshold_vp_r_lo, threshold_vp_r_hi, threshold_vp_cnt)
        camera_theta, camera_direction, focal_length = camera_info(Igs, vp)
        print(camera_theta, camera_direction, focal_length)

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w, c = image.shape
        vp_min0 = (vp[:, 0].min().astype(np.int32) - 5) if vp.shape[0] > 0 else 2147483647
        vp_max0 = (vp[:, 0].max().astype(np.int32) + 5) if vp.shape[0] > 0 else -2147483648
        vp_min1 = (vp[:, 1].min().astype(np.int32) - 5) if vp.shape[0] > 0 else 2147483647
        vp_max1 = (vp[:, 1].max().astype(np.int32) + 5) if vp.shape[0] > 0 else -2147483648
        x_min, x_max = min(0, vp_min0), max(h - 1, vp_max0)
        y_min, y_max = min(0, vp_min1), max(w - 1, vp_max1)
        result = np.zeros((x_max - x_min + 1, y_max - y_min + 1, c))
        result[-x_min:h-x_min, -y_min:w-y_min] = image
        result = Image.fromarray(result.astype(np.uint8))
        draw = ImageDraw.Draw(result)
        if DRAW_HL:
            N = lRho.shape[0]
            for i in range(N):
                if np.abs(np.cos(lTheta[i])) >= np.abs(np.sin(lTheta[i])):
                    draw.line((np.round((lRho[i]-x_min*np.sin(lTheta[i]))/np.cos(lTheta[i])).astype(np.int32)-y_min,0,np.round((lRho[i]-x_max*np.sin(lTheta[i]))/np.cos(lTheta[i])).astype(np.int32)-y_min,x_max-x_min), fill=(0,255,0), width=2)
                else:
                    draw.line((0,np.round((lRho[i]-y_min*np.cos(lTheta[i]))/np.sin(lTheta[i])).astype(np.int32)-x_min,y_max-y_min,np.round((lRho[i]-y_max*np.cos(lTheta[i]))/np.sin(lTheta[i])).astype(np.int32)-x_min), fill=(0,255,0), width=2)
        if DRAW_VP:
            N = vp.shape[0]
            for i in range(N):
                draw.rectangle((vp[i][1]-y_min-2,vp[i][0]-x_min-2,vp[i][1]-y_min+2+1,vp[i][0]-x_min+2+1), fill=(255,0,0))
        result.save(resultdir+'/vanishing_point/'+img_path[len(datadir)+1:])

        print(img_path,"DONE")

if __name__ == '__main__':
    main()