import math
from tkinter import N
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # TODO ...
    # p_1 = H * p_2
    n = p1.shape[0]
    A = np.zeros((2*n, 9)).astype(float)
    
    for i in range(n):
        A[i*2][0] = p2[i][0]
        A[i*2][1] = p2[i][1]
        A[i*2][2] = 1.
        A[i*2][6] = - p2[i][0] * p1[i][0]
        A[i*2][7] = - p1[i][0] * p2[i][1]
        A[i*2][8] = - p1[i][0]
        A[i*2+1][3] = p2[i][0]
        A[i*2+1][4] = p2[i][1]
        A[i*2+1][5] = 1.
        A[i*2+1][6] = - p2[i][0] * p1[i][1]
        A[i*2+1][7] = - p2[i][1] * p1[i][1]
        A[i*2+1][8] = - p1[i][1]
    
    u, s, v = np.linalg.svd(A)
    H = v[s.shape[0]-1].reshape((3,3))
    return H

# Normalization for improving numerical stability 
def compute_h_norm(p1, p2):
    p1 = p1.astype(float)
    p2 = p2.astype(float)

    n = p1.shape[0]         # keypoint 개수

    avg_x1 = np.mean(p1[:,0])
    avg_x2 = np.mean(p2[:,0])
    avg_y1 = np.mean(p1[:,1])
    avg_y2 = np.mean(p2[:,1])
    
    sum1 = 0.
    sum2 = 0.
    # sum of square of distance from each point to the average point
    for i in range(0, n):
        sum1 += (p1[i,0] - avg_x1) ** 2 + (p1[i,1] - avg_y1) ** 2   
        sum2 += (p2[i,0] - avg_x2) ** 2 + (p2[i,1] - avg_y2) ** 2
    s1 = math.sqrt(2) * n / math.sqrt(sum1)
    s2 = math.sqrt(2) * n / math.sqrt(sum2)

    T1 = s1 * np.array([[1, 0, -avg_x1],[0, 1, -avg_y1],[0, 0, 1/s1]])
    T2 = s2 * np.array([[1, 0, -avg_x2],[0, 1, -avg_y2],[0, 0, 1/s2]])

    p1 = np.pad(p1, ((0,0),(0,1)), 'constant', constant_values = 1)
    p2 = np.pad(p2, ((0,0),(0,1)), 'constant', constant_values = 1)

    p1 = np.dot(p1, np.transpose(T1[0:2,:]))
    p2 = np.dot(p2, np.transpose(T2[0:2,:]))

    H = compute_h(p1, p2)

    H = np.dot(np.linalg.inv(T1), H)
    H = np.dot(H, T2)  
    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    # multipling H to 4 corner vertexes of igs_in and divide it to array[2]
    p1 = (np.dot(H, [0,0,1])).astype(float)
    p1 = (p1 / p1[2]).astype(int)

    p2 = (np.dot(H, [igs_in.shape[1]-1, 0, 1])).astype(float)
    p2 = (p2 / p2[2]).astype(int)

    p3 = (np.dot(H, [0, igs_in.shape[0]-1, 1])).astype(float)
    p3 = (p3 / p3[2]).astype(int)

    p4 = (np.dot(H, [igs_in.shape[1]-1, igs_in.shape[0]-1, 1])).astype(float)
    p4 = (p4 / p4[2]).astype(int)
    
    x_arr = [p1[0], p2[0], p3[0], p4[0]]
    y_arr = [p1[1], p2[1], p3[1], p4[1]]
    
    # Using heuristic that the input images are only "wdc1.png", "wdc2.pnd"
    x_n = np.amax(x_arr) + np.amin(x_arr)
    y_n = np.amax(y_arr) - np.amin(y_arr)

    igs_merge = np.zeros((y_n, x_n, 3))
    igs_warp = np.zeros((igs_ref.shape[0], igs_ref.shape[1], 3))
   
    H_inv = np.linalg.inv(H)

    for y in range(0, y_n) :
        for x in range(0, x_n) :
            new = (np.dot(H_inv, np.array([x - np.amin(x_arr), y + np.amin(y_arr), 1.]))).astype(float)
            new = (new / new[2]).astype(int)

            # if each x,y coordinate of new is inside the range of p_in, then mark it to igs_warp     
            if 0 <= new[0] < igs_in.shape[1]  and 0 <= new[1] < igs_in.shape[0] :
                igs_merge[y,x,:] = igs_in[new[1],new[0],:] 
    
    igs_warp[:,:,:] = igs_merge[-np.amin(y_arr) : -np.amin(y_arr) + igs_ref.shape[0], np.amin(x_arr): np.amin(x_arr) + igs_ref.shape[1], :] 
    igs_merge[-np.amin(y_arr) : -np.amin(y_arr) + igs_ref.shape[0], np.amin(x_arr): np.amin(x_arr) + igs_ref.shape[1], :] = igs_ref

    return igs_warp, igs_merge


def rectify(igs, p1, p2):
    # TODO ...
    # p2 = H * p1
    # concatenating the same point coordinate
    row_n, col_n = igs.shape[0], igs.shape[1]
    p1 = np.concatenate( (p1, np.array([[p1[3][0], p1[3][1]]])), axis = 0)
    p2 = np.concatenate( (p2, np.array([[p2[3][0], p2[3][1]]])), axis = 0)
    H = compute_h_norm(p2,p1)
    H_inv = np.linalg.inv(H) 
    igs_rec = np.zeros((row_n, col_n, 3))

    for y in range(0, row_n) :
        for x in range(0, col_n) :
            new = (np.dot(H_inv, np.array([x,y,1.]))).astype(float)
            new = (new / new[2]).astype(int)

            # if each x, y coordinate of new is inside the range of igs, then mark it to igs_rec     
            if 0 <= new[0] < col_n  and 0 <= new[1] < row_n :
                igs_rec[y,x,:] = igs[new[1],new[0],:] 
    
    return igs_rec

def set_cor_mosaic():
    # TODO ...
    # wdc1
    p_in = np.array([[375,98],[358,109],[265,137],[207,139],[146,180],[121,224],[371,250]])     
    # wdc2
    p_ref = np.array([[207,267],[217,232],[281,137],[330,103],[331,50],[307,22],[107,93]])


    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    c_in = np.array([[169,14],[251,22],[251,247],[170,257]])
    x1 = np.sqrt((c_in[0][0] - c_in[1][0]) ** 2 + (c_in[0][1] - c_in[1][1]) ** 2)
    x2 = np.sqrt((c_in[2][0] - c_in[3][0]) ** 2 + (c_in[2][1] - c_in[3][1]) ** 2)
    y1 = np.sqrt((c_in[1][0] - c_in[2][0]) ** 2 + (c_in[1][1] - c_in[2][1]) ** 2)
    y2 = np.sqrt((c_in[3][0] - c_in[0][0]) ** 2 + (c_in[3][1] - c_in[0][1]) ** 2)
    
    # (width/height) ratio of iPhone
    ratio = (x1 + x2) / (y1 + y2)

    hgh = 200
    wth = (hgh * ratio).astype(int)
    
    c_ref = np.array([[180, 30],[180 + wth, 30],[180 + wth, 30 + hgh],[180, 30 + hgh]]) 
    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/wdc1.png').convert('RGB')
    img_ref = Image.open('data/wdc2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # check keypoint
    import matplotlib.pyplot as plt
    plt.subplot(121); plt.imshow(igs_in); plt.axis('off');
    plt.scatter(p_in[:,0], p_ref[:,1], marker='x');
    plt.subplot(122); plt.imshow(igs_ref); plt.axis('off');
    plt.scatter(p_ref[:,0], p_ref[:,1], marker='x');
    plt.savefig('result/keypoint.png', bbox_inches='tight')

    # # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    print(H)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('result/wdc1_warped.png')
    img_merge.save('result/my_mosaic.png')
    
    # Compare my code and openCV
    import cv2
    from skimage.transform import warp
    img_in = plt.imread('data/wdc1.png')
    img_ref = plt.imread('data/wdc2.png')
    my_mosaic = plt.imread('result/my_mosaic.png')
    P = cv2.getPerspectiveTransform(p_in[[0,3,4,6]].astype(np.float32), p_ref[[0,3,4,6]].astype(np.float32)) # 특징점 4개 입력받아 투시변환 행렬 반환
    f_stitched = cv2.warpPerspective(img_in, P, dsize=my_mosaic.shape[:2])
    plt.imsave('result/mosaic.png', f_stitched)
    M, N = img_ref.shape[:2]
    print(f_stitched.shape)
    f_stitched[0:M, 0:N, :] = img_ref
    print(f_stitched.shape)
    plt.imshow(f_stitched)
    plt.show()
    plt.imsave('result/cv2_mosaic.png', f_stitched)
    
    ##############
    # step 2: rectification
    ##############

    # img_rec = Image.open('data/iphone.png').convert('RGB')
    # igs_rec = np.array(img_rec)

    # c_in, c_ref = set_cor_rec()

    # igs_rec = rectify(igs_rec, c_in, c_ref)

    # img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    # img_rec.save('result/iphone_rectified.png')

if __name__ == '__main__':
    main()
