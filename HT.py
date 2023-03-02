import math
import glob
import numpy as np
from PIL import Image, ImageDraw
import os   # for check results folder and save transformed images


# parameters

datadir = './data'
resultdir='./results'

# you can calibrate these parameters
sigma=2
threshold=0.03
rhoRes=2
thetaRes=math.pi/180
nLines=20   # fix

def Replication_pad(Igs,r): # grey_scale Image, r = math.ceil(3*sigma)
    Igs_pd = np.pad(Igs,((r,r),(r,r)), 'constant', constant_values=0 )

    for i in range(r):
        for j in range(np.shape(Igs)[1]):
            Igs_pd[r-1-i,j+r] = Igs[i,j] # up side pad
            Igs_pd[np.shape(Igs)[0]+r+i,j+r] = Igs[np.shape(Igs)[0]-1-i,j] # down side pad

    for i in range(r):
        for j in range(np.shape(Igs)[0]):
            Igs_pd[j+r,r-1-i] = Igs[j,i] # left side pad
            Igs_pd[j+r,np.shape(Igs)[1]+r+i] = Igs[j,np.shape(Igs)[1]-1-i] # right side pad

    # diagonal pad
    for i in range(r):
        for j in range(r):
            Igs_pd[r-1-i,r-1-j] = Igs[i,j] # left up pad
            Igs_pd[r-1-i,np.shape(Igs)[1]+r+j] = Igs[i,np.shape(Igs)[1]-1-j] # right up pad
            Igs_pd[np.shape(Igs)[0]+r+i,r-1-j] = Igs[np.shape(Igs)[0]-1-i,j] # left down pad
            Igs_pd[np.shape(Igs)[0]+r+i,np.shape(Igs)[1]+r+j] = Igs[np.shape(Igs)[0]-1-i,np.shape(Igs)[1]-1-j] # right down pad
    
    return Igs_pd

def get_gaussian(sigma):
    
    r = math.ceil(3*sigma)
    k = 2*r+1
    kernel = np.zeros((k, k))
    sum = 0

    for x in range(-r,r+1):
        for y in range(-r,r+1):
                g = (1/(2 * math.pi * (sigma**2)))*math.exp(-(x**2 + y**2)/(2* sigma**2))
                kernel[x+r, y+r] = g
                sum += kernel[x+r, y+r]

    # Normalize the kernel to prevent of the image become darker depending on the sigma
    for x in range(-r,r+1):
        for y in range(-r,r+1):
            kernel[x+r, y+r] /= sum

    
    return kernel

def ConvFilter(Igs, G):
    
    k = len(G)
    r = k//2

    Igs_pd = Replication_pad(Igs, r)

    Iconv_shape = (len(Igs), len(Igs[0]))
    Iconv = np.zeros(Iconv_shape)  # initialize output image
    fS = np.zeros((k, k))  # initialize flipped kernel

    # kernel filp twice
    for x in range(k):
        for y in range(k):
            xp = k - (x+1)
            yp = k - (y+1)
            fS[x, y] = G[xp, yp]

    for row in range (Iconv_shape[0]):        # stride = 1
        for column in range (Iconv_shape[1]):
            current_img = Igs_pd[row:row+k, column:column+k]    
            multiplication = current_img*fS
            Iconv[row][column] = sum(sum(multiplication))

    return Iconv

def EdgeDetection(Igs, sigma):
    # TODO ...
    G = get_gaussian(sigma)
    Iconv = ConvFilter(Igs, G)

    sobelx = np.array([[-1.,0.,1.], [-2.,0.,2.], [-1.,0.,1.]]) 
    sobely = np.array([[1.,2.,1.], [-1.,-2.,-1.], [0.,0.,0.]]) 

    Ix = ConvFilter(Iconv, sobelx)
    Iy = ConvFilter(Iconv, sobely)

    Im = np.sqrt(Ix**2 + Iy**2)
    Io = np.arctan2(Iy, Ix)

    # NMS
    H, W = Im.shape
    Im_NMS = np.zeros((H, W), dtype=np.float32)     # initialize edge magnitue image
       
    # rotate and calculation in (0, 45, 90, 135) degree direction
    for y in range(1, H-1):
        for x in range(1, W-1):
            angle = Io[y, x]
            if angle < 0:
                angle += np.pi

            # 0 degree
            if (0 <= angle < np.pi/8) or (np.pi*7/8 <= angle <= np.pi):
                dy1, dx1, dy2, dx2 = 0, -1, 0, 1
            
            # 45 degree
            elif (np.pi/8 <= angle < np.pi*3/8):
                dy1, dx1, dy2, dx2 = 1, -1, -1, 1
            
            # 90 degree
            elif (np.pi*3/4 <= angle < np.pi*5/8):
                dy1, dx1, dy2, dx2 = -1, 0, 1, 0
            
            # 135 degree
            else:
                dy1, dx1, dy2, dx2 = -1, -1, 1, 1
            
            if Im[y, x] > Im[y+dy1, x+dx1] and Im[y, x] > Im[y+dy2, x+dx2]:
                Im_NMS[y, x] = Im[y, x]

    return Im_NMS, Io, Ix, Iy

def HoughTransform(Im,threshold, rhoRes, thetaRes):
    # TODO ...
    R, C = Im.shape
    Ith = np.where(np.array(Im) > threshold, 1, 0)
    H = np.zeros((int(math.sqrt(R**2 + C**2)/rhoRes), int(2*math.pi/thetaRes)))
        
    for y in range(0, R):
        for x in range(0, C):
            if Ith[y, x] == 1:
                # per (x, y) in Ith, vote line in rho and theta axes
                for i in range(0, int(2*math.pi/thetaRes)): 
                    theta = i*thetaRes
                    rho = x*math.cos(theta) + y*math.sin(theta)
                    rho_idx = int(rho/rhoRes)
                    rho_idx = np.where(rho_idx < 0, 0, rho_idx) # if rho_idx < 0, set it to 0
                    H[rho_idx, i] += 1
    H[0,:] = 0
    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...
    R, C = H.shape
    H_pad = np.pad(H, ((1, 1),(1, 1)), 'constant', constant_values=0) 
    lRho = np.zeros(nLines)
    lTheta = np.zeros(nLines)
    
    # NMS for all neighbors of pixel
    for i in range(0, R):
        for j in range(0, C):
            if np.amax(H_pad[i:i+3,j:j+3]) != H[i][j]:
                H[i][j] = 0

    # calculate line parameters
    for i in range(0, nLines):
        rho_idx = np.argmax(H) // C
        theta_idx = np.argmax(H) % C
        H[rho_idx][theta_idx] = 0 
        lRho[i] = rho_idx * rhoRes
        lTheta[i] = theta_idx * thetaRes

    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im, threshold):
    # TODO ...
    R, C = Im.shape
    Ith = np.where(Im > threshold, 1, 0)
    Ith = np.pad(Ith, ((1, 1),(1, 1)), 'constant', constant_values=0)

    l = {}
    for i in range(0, lRho.size):
        rho = lRho[i]
        theta = lTheta[i]
        temp = {}
        longest = {}
        longest_len = 0
        ending = 2

        if -1 <= np.tan(theta) <= 1: 
            for y in range(0, R):
                x = (- np.tan(theta) * y + rho / np.cos(theta)).astype(int)
                if 0 <= x < C:  # check if x is in the image
                    if np.max(Ith[y:y+3, x:x+3]) == 1 and ending == 2:  # start point
                        ending = 1
                        temp['start'] = (x, y)
                        temp['end'] = (x, y)

                    elif np.max(Ith[y:y+3, x:x+3]) == 0 and 0 < ending < 2:  # end point
                        ending -= 1
                        temp['end'] = (x, y)

                    elif ending == 0 or (y == R - 1 and ending != 2): # end of line
                        ending = 2
                        temp['end'] = (x, y)
                        line_len = abs(temp['end'][1] - temp['start'][1])
                        # longest line
                        if longest_len < line_len:
                            longest_len = line_len
                            longest = temp
                        temp = {}

        else:
            for x in range(0, C):
                y = (- 1 / np.tan(theta) * x + rho / np.sin(theta)).astype(int)
                if 0 <= y < R:  # check if y is in the image
                    if np.max(Ith[y:y+3, x:x+3]) == 1 and ending == 2:  # start point
                        ending = 1
                        temp['start'] = (x, y)
                        temp['end'] = (x, y)
                    
                    elif np.max(Ith[y:y+3, x:x+3]) == 0 and 0 < ending < 2:  # end point
                        ending -= 1
                        temp['end'] = (x, y)
                    
                    elif ending == 0 or (x == C - 1 and ending != 2): # end of line
                        ending = 2
                        temp['end'] = (x, y)
                        line_len = abs(temp['end'][0] - temp['start'][0])
                        # longest line
                        if longest_len < line_len:
                            longest_len = line_len
                            longest = temp
                        temp = {}

        l[i] = longest
    return l

# Draw HoughLines on the image
def DrawLine(Irgb, lRho, lTheta):
    R, C, channel = Irgb.shape

    for i in range(0, lRho.size):
        rho = lRho[i]
        theta = lTheta[i]

        if -1 <= np.tan(theta) <= 1:
            for y in range(0, R):
                x = (- np.tan(theta) * y + rho / np.cos(theta)).astype(int)
                if 0<= x < C:
                    Irgb[y][x] = [255,0,0]
        else:
            for x in range(0, C):
                y = (- 1 / np.tan(theta) * x + rho / np.sin(theta)).astype(int)
                if 0<= y < R:
                    Irgb[y][x] = [255,0,0]
    Img = Image.fromarray(Irgb)
    return Img

def DrawSegments(Irgb, l):

    for j in range(0, len(l)):
        shape = [ l[j]['start'] , l[j]['end'] ]
        img1 = ImageDraw.Draw(Irgb)
        img1.line(shape, fill="red", width = 2)
    
    return Irgb

def main():

    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        I = Image.open(img_path)
        Irgb = I.convert('RGB')
        Igs = np.array(I.convert('L'))
        Igs = Igs / 255.

        img_name = img_path.split('/')[-1][:-4]

        if not os.path.exists(resultdir):
            os.makedirs(resultdir) 

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments

        # EdgeDetection
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        img = Image.fromarray(Im * 255).convert("L")
        img.save(resultdir+'/'+img_name+"_Im.jpg")
        img = Image.fromarray(Io * 255).convert("L")
        img.save(resultdir+'/'+img_name+"_Io.jpg")
        img = Image.fromarray(Ix * 255).convert("L")
        img.save(resultdir+'/'+img_name+"_Ix.jpg")
        img = Image.fromarray(Iy * 255).convert("L")
        img.save(resultdir+'/'+img_name+"_Iy.jpg")

        # HoughTransform
        H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        Himg = H / np.amax(H)
        img = Image.fromarray(Himg * 255).convert("L")
        img.save(resultdir+'/'+img_name+"_HT.jpg")


        # HoughLines
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        img = DrawLine(np.array(Irgb), lRho, lTheta)
        img.save(resultdir+'/'+img_name+"_HL.jpg")

        # HoughLineSegments
        l = HoughLineSegments(lRho, lTheta, Im, threshold)
        img = DrawSegments(Irgb, l)
        img.save(resultdir+'/'+img_name+"_HLS.jpg")  

if __name__ == '__main__':
    main()
