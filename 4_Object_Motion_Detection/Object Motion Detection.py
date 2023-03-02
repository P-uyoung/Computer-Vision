import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
from tqdm import tqdm

def lucas_kanade_affine(T, I):
    # These codes are for calculating image gradient by using sobel filter
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this
    
    p = np.zeros(6) # initializer p
  
    ### START CODE HERE ###
    # [Caution] You should use only numpy and RectBivariateSpline functions
    # Never use opencv
    width = T.shape[1]
    height = T.shape[0]
    M = np.array([[1+p[0], p[2], p[4]],[p[1], 1+p[3], p[5]],[0, 0, 1]])
    
    x_array = np.arange(0, 1, 1/width) #normalize
    y_array = np.arange(0, 1, 1/height) #normalize
    I_Spline = RectBivariateSpline(y_array, x_array, I)
    Ix_Spline = RectBivariateSpline(y_array, x_array, Ix)
    Iy_Spline = RectBivariateSpline(y_array, x_array, Iy)

    H = np.zeros((6,6))
    Hdp = np.zeros((6))

    for x in range(0, width, 3):
        for y in range(0, height, 3):
            x_warped, y_warped, _ = np.dot(M, np.array([x,y,1]))
            
            I_warped = I_Spline.ev(y_warped/height, x_warped/width)
            Ix_warped = Ix_Spline.ev(y_warped/height, x_warped/width)
            Iy_warped = Iy_Spline.ev(y_warped/height, x_warped/width)
               
            error = T[y][x] - I_warped
            I_gradinet = np.array([Ix_warped * width, Iy_warped * height])
            J = np.array([[x/width, 0., y/height, 0., 1., 0.], 
                          [0., x/width, 0., y/height, 0., 1.]])
               
            H += np.outer(np.transpose(np.dot(I_gradinet, J)), np.dot(I_gradinet, J))
            Hdp += np.transpose(np.dot(I_gradinet, J)) * error

    dp = np.dot(np.linalg.inv(H), Hdp)
    
    ### END CODE HERE ###
    return dp
    
def subtract_dominant_motion(It, It1):
    
    ### START CODE HERE ###
    # [Caution] You should use only numpy and RectBivariateSpline functions
    # Never use opencv
    p = np.zeros(6)
    dp = np.zeros(6)
    i = 0
    # iterate dp
    for j in range(0,5):
        dp = lucas_kanade_affine(It, It1)
        p += dp
    maxdp = 1
    while np.amax(np.abs(dp)) > 0.00005 and i < 100:
        if np.amax(np.abs(dp)) > maxdp:
            break
        maxdp = np.amax(np.abs(dp))
        dp = lucas_kanade_affine(It, It1)
        p += dp 
        i += 1

    M = np.array([[1.+p[0], p[2], p[4]],[p[1], 1.+p[3], p[5]],[0., 0., 1.]])
    M_inv = np.linalg.inv(M)
    width = It.shape[1]
    height = It.shape[0]    
    x_array = np.arange(0, width)
    y_array = np.arange(0, height)

    It_Spline = RectBivariateSpline(y_array, x_array, It)
    It_warped = np.zeros((height,width))
    It_warped += It1

    for x in range(width):
        for y in range(height):
            x_warped, y_warped, _ = np.dot(M_inv, np.array([x,y,1]))
            if 0 <= x_warped < width and 0 <= y_warped < height:
                It_warped[y][x] = It_Spline.ev(y_warped,x_warped)

    motion_image = np.abs(It1 - It_warped)
    
    ### START CODE HERE ###
    
    th_hi = 0.25 * 256 # you can modify this
    th_lo = 0.10 * 256 # you can modify this
    
    mask = apply_hysteresis_threshold(motion_image, th_lo, th_hi)
    
    return mask

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    data_dir = 'data/motion'
    video_path = 'results/motion_best.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
    img_path = os.path.join(data_dir, "{}.jpg".format(0))
    It = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    for i in tqdm(range(1, 61)):
        img_path = os.path.join(data_dir, "{}.jpg".format(i))
        It1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        It_clone = It.copy()
        mask = subtract_dominant_motion(It, It1)
        It_clone = cv2.cvtColor(It_clone, cv2.COLOR_GRAY2BGR)
        It_clone[mask, 2] = 255 # set color red for masked pixels
        out.write(It_clone)
        It = It1
    out.release()