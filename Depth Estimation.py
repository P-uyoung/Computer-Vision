import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

def SSD(t, x, r, c, w):
    w0 = w // 2
    
    ### START CODE HERE ###
    ssd = np.sum((t - x[r-w0:r+w0+1, c-w0:c+w0+1])**2)
    
    ### END CODE HERE ###
    return ssd

def disparity_SSD_c(I_L, I_R):
    assert I_L.shape == I_R.shape
    
    w = 11                  # you can change this
    assert(w % 2 == 1)
    w0 = w // 2
    
    ### START CODE HERE ###
    D = np.zeros_like(I_L) # you can remove this
    H, W = I_L.shape
    
    for row in range(w0, H-w0):
        dsi = np.zeros((W, W)) # disparity search image
        for col in range(w0, W-w0):
            template = I_L[row-w0:row+w0+1, col-w0:col+w0+1]
            for i in range(w0, W-w0):
                dsi[col, i] = SSD(template, I_R, row, i, w)
            

        ## By DSI, find disparity map (Dynamic Programming) ##
        occlusionCost = 100*w*w
        C = np.zeros((W,W)) # minimum cost to each pixel
        M = np.zeros((W,W)) # direction from each pixel
        
        # Since dsi was calculated in units of window size, C set with the w0 axes for H,W
        for x in range(W):
            C[x,w0] = x*occlusionCost
            C[w0,x] = x*occlusionCost
            
        # M set with direction
        for x in range(w0+1,W):
            for y in range(w0+1,W):
                min1=C[x-1,y-1]+abs(dsi[x,y])
                min2=C[x-1,y]+occlusionCost
                min3=C[x,y-1]+occlusionCost
                C[x,y]=cmin=min([min1,min2,min3])
                if(min1==cmin):
                    M[x,y]=1
                if(min2==cmin):
                    M[x,y]=2
                if(min3==cmin):
                    M[x,y]=3
        
        # At M array, the index to go up from the bottom right to the top left
        # Strating point is [W-w0, W-w0]
        x=W-w0-1
        y=W-w0-1
        
        # break when the index reaches the top left
        while(x!=w0 and y!=w0):
            if M[x,y]==1:
                D[row,x]=abs(x-y)
                x-=1
                y-=1
            elif M[x,y]==2:
                x-=1
            elif M[x,y]==3:
                y-=1

    ## Consider occlusion ##
    for x in range(H):
        for y in range(W):
            # Detect occlusion
            if D[x,y] == 0:
                # Find (x,y) with values on the samle line
                for z in range(y, W):
                    if D[x,z] != 0:
                        D[x,y] = (D[x,y-1] + D[x,z])//2 # the average of the occlusion's left and found values 
                        break
                    # If there is no value on the same line, set the value to the left
                    D[x,y] = D[x,y-1] # occlusion's left value

    ### END CODE HERE ###
    return D

def disparity_SSD_d(I_L, I_R):
    assert I_L.shape == I_R.shape
    
    w = 11                  # you can change this
    assert(w % 2 == 1)
    w0 = w // 2
    max_disparity = 20      # you can change this
    
    ### START CODE HERE ###
    D = np.zeros_like(I_L) # you can remove this

    H, W = I_L.shape
    
    for row in range(w0, H-w0):
        dsi = np.zeros((W, W)) # disparity search image
        for col in range(w0, W-w0):
            template = I_L[row-w0:row+w0+1, col-w0:col+w0+1]
            min_templ, max_templ = np.min(template), np.max(template)
            template = (template - min_templ) / (max_templ - min_templ)

            for i in range(w0, W-w0):
                target = I_R[row-w0:row+w0+1, i-w0:i+w0+1]
                min_target, max_target = np.min(target), np.max(target)
                target = (target - min_target) / (max_target - min_target)
                dsi[col, i] = np.sum((template - target)**2)
            

        # By DSI, find disparity map (Dynamic Programming) ##
        occlusionCost = 0.4*w*w
        C = np.zeros((W,W)) # minimum cost to each pixel
        M = np.zeros((W,W)) # direction from each pixel
        
        # Since dsi was calculated in units of window size, C set with the w0 axes for H,W
        for x in range(W):
            C[x,w0] = x*occlusionCost
            C[w0,x] = x*occlusionCost
            
        # M set with direction
        for x in range(w0+1,W):
            for y in range(w0+1,W):
                min1=C[x-1,y-1]+abs(dsi[x,y])
                min2=C[x-1,y]+occlusionCost
                min3=C[x,y-1]+occlusionCost
                C[x,y]=cmin=min([min1,min2,min3])
                if(min1==cmin):
                    M[x,y]=1
                if(min2==cmin):
                    M[x,y]=2
                if(min3==cmin):
                    M[x,y]=3
        
        # At M array, the index to go up from the bottom right to the top left
        # Strating point is [W-w0, W-w0]
        x=W-w0-1
        y=W-w0-1
        
        # break when the index reaches the top left
        while(x!=w0 and y!=w0):
            if M[x,y]==1:
                D[row,x]=abs(x-y)
                x-=1
                y-=1
            elif M[x,y]==2:
                x-=1
            elif M[x,y]==3:
                y-=1


    ## Consider occlusion ##
    for x in range(H):
        for y in range(W):
            # Detect occlusion
            if D[x,y] == 0:
                # Find (x,y) with values on the samle line
                for z in range(y, W):
                    if D[x,z] != 0:
                        D[x,y] = (D[x,y-1] + D[x,z])//2 # the average of the occlusion's left and found values 
                        break
                    # If there is no value on the same line, set the value to the left
                    D[x,y] = D[x,y-1] # occlusion's left value
    
    ### END CODE HERE ###
    
    return D


# Do not modify this
def save_disparity_map(filename, D):
    D = (np.clip(D / 64 * 255.0, 0, 255)).astype(np.uint8)
    cv2.imwrite(filename, D)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('wrong arguments')
        exit(0)
        
    problem = sys.argv[1] # c or d
    l_image = sys.argv[2] # L1
    r_image = sys.argv[3] # R1 or R2
    
    # Read images
    data_dir = './data/stereo'
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    I_L = cv2.cvtColor(cv2.imread(os.path.join(data_dir, f'{l_image}.png')), cv2.COLOR_BGR2GRAY)
    I_R = cv2.cvtColor(cv2.imread(os.path.join(data_dir, f'{r_image}.png')), cv2.COLOR_BGR2GRAY)

    if problem == 'c':
        D = disparity_SSD_c(I_L, I_R)
    elif problem == 'd':
        D = disparity_SSD_d(I_L, I_R)
    else:
        raise ValueError
    
    save_disparity_map(os.path.join(save_dir, f'disparity_{problem}_{l_image}{r_image}.png'), D)