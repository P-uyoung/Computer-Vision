import sys
import numpy as np
from scipy import misc
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy import fftpack as fft
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
from matplotlib import pyplot as plt

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class PhotometricStereo():
    def __init__(self):
        self.images = []
        self.light_coords = []
        self.image_num = 0

    def loadImgNCoord(self, paths, light_coords):
        for i in file_list:
            path = os.path.join(dir_name, i)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.images.append(img)
            self.image_num = self.image_num + 1

        self.light_coords = light_coords
        self.img_size = self.images[0].shape

    def calNormal(self):
        self.normal_map = np.zeros((self.img_size[0], self.img_size[1], 3))
        self.rou_map = np.zeros(self.img_size)

        light_coords = np.array(self.light_coords)
        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                intensities = []

                for img_num in range(self.image_num):
                    intensities.append(self.images[img_num][i][j])

                intensities = np.array(intensities)
                intensities = intensities / light_intensities
                normal = np.linalg.inv(light_coords.transpose().dot(light_coords)).dot(light_coords.transpose()).dot(
                    intensities)
                rou = np.linalg.norm(normal)
                normal = normal / rou
                self.rou_map[i][j] = rou
                self.normal_map[i][j] = normal


def sanitise_image(image):
    return (image / 255).flatten()

def get_surface_normal(images, L):
    original_size = images[0].shape[:2]

    # Normalise all images and convert to row vectors (each image is one row)
    images = np.vstack(map(sanitise_image, images))

    # Make sure that lighting vectors are normalised
    L = L / np.linalg.norm(L, ord=2, axis=1, keepdims=True)

    # Solve for G = N'*rho using ordinary least squares
    # (L^T L) \ L^T
    norm_sln = np.linalg.pinv(L.T.dot(L)).dot(L.T)

    # For a single pixel (3x1 column) we can trivially calculate G: norm_sum * px
    # norm_sln is 3x3, images is 3xn (where n i num pixels)
    # It's slow to iterate this, but the einsum method lets us broadcast the multiplication over the array
    G = np.einsum("ij,il", norm_sln, images)

    # The albedo is just the column-wise L2 norm (magnitude) of G...
    rho = np.linalg.norm(G, axis=0)

    # The normal map is simply each column of G divided the equivalent column in the albedo
    N = np.divide(G, np.vstack([rho] * 3))

    # Reshape back to image
    rho = rho.reshape(512,612)

    # We need to transpose the normal list before we reshape
    N = N.T.reshape(512, 612, 3)

    return N, rho

def unbiased_integrate(n1, n2, n3, mask, angle_threshold):
    nrows, ncols = mask.shape
    # Mapping
    objectPixels = np.where(mask > 0)
    numPixels = len(objectPixels[0])

    index = np.zeros((nrows, ncols)).astype(int)

    for i in range(0, numPixels):
        pRow = objectPixels[0][i]
        pCol = objectPixels[1][i]
        index[pRow, pCol] = i

    M = np.zeros((4*numPixels, numPixels))
    b = np.zeros((4*numPixels, 1))

    for i in tqdm(range(0, numPixels)):
        pRow = objectPixels[0][i]
        pCol = objectPixels[1][i]
        nx = n1[pRow, pCol]
        ny = n2[pRow, pCol]
        nz = n3[pRow, pCol]
        # 1 qt /  x > 0
        if (index[pRow, pCol+1] > 0):
            near_pixel_norm = [n1[pRow,pCol+1], n2[pRow,pCol+1], n3[pRow,pCol+1]]
            if(angle([nx,ny,nz],near_pixel_norm) < angle_threshold):
                M[4 * i, index[pRow, pCol]] = -1
                M[4 * i, index[pRow, pCol+1]] = 1
                b[4 * i, 0] = - nx / nz
        # 2 qt / y > 0
        if (index[pRow-1, pCol]>0):
            near_pixel_norm = [n1[pRow-1, pCol], n2[pRow-1, pCol], n3[pRow-1, pCol]]
            if (angle([nx, ny, nz], near_pixel_norm) < angle_threshold):
                M[4 * i + 1, index[pRow, pCol]] = -1
                M[4 * i + 1, index[pRow - 1, pCol]] = 1
                b[4 * i + 1, 0] = - ny / nz
        # 3 qt / x < 0
        if (index[pRow, pCol-1] > 0):
            near_pixel_norm = [n1[pRow, pCol-1], n2[pRow, pCol-1], n3[pRow, pCol-1]]
            if (angle([nx, ny, nz], near_pixel_norm) < angle_threshold):
                M[4 * i + 2, index[pRow, pCol]] = -1
                M[4 * i + 2, index[pRow, pCol - 1]] = 1
                b[4 * i + 2, 0] = nx / nz
        # 4 qt / y < 0
        if (index[pRow+1, pCol]>0):
            near_pixel_norm = [n1[pRow+1, pCol], n2[pRow+1, pCol], n3[pRow+1, pCol]]
            if (angle([nx, ny, nz], near_pixel_norm) < angle_threshold):
                M[4 * i + 3, index[pRow, pCol]] = -1
                M[4 * i + 3, index[pRow + 1, pCol]] = 1
                b[4 * i + 3, 0] = ny / nz

    sparse_M = sp.csr_matrix(M)
    print("done sparse matrix")
    print(sparse_M.shape)
    print(b.shape)
    x = sp.linalg.lsqr(sparse_M,b)[0]
    print("done least square f")
    print(np.min(x), np.max(x), np.mean(x))
    x = x - np.min(x)
    tempShape = np.zeros((nrows, ncols))

    for i in range(0, numPixels):
        pRow = objectPixels[0][i]
        pCol = objectPixels[1][i]
        tempShape[pRow, pCol] = x[i]


    return tempShape

def angle(v1, v2):
    v = np.inner(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    theta = np.arccos(v)
    return theta

def display_depth_matplotlib(z, target, title):
    from matplotlib.colors import LightSource

    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ls = LightSource(azdeg=0, altdeg=65)
    greyvals = ls.shade(z, plt.cm.Greys)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False, facecolors=greyvals)

    for azim in np.linspace(0,90,2):
        for elev in np.linspace(0, 90, 4):
            ax.azim = azim
            ax.elev = elev
            plt.savefig(target + "/" + title +"_azim_"+str(int(azim))+"_elev_"+str(int(elev))+".png")


def read_data_file(filename):
    data = loadmat(filename)
    norm_gt = data['Normal_gt']

    return norm_gt

def remove_edge_mask(z, mask):
    objectPixels = np.where(mask > 0)
    new_mask = mask

    for i in range(0, len(objectPixels)):
        pRow = objectPixels[0][i]
        pCol = objectPixels[1][i]

        if mask[pRow-1,pCol] == 0 or mask[pRow+1,pCol] == 0 or mask[pRow,pCol-1] == 0 or mask[pRow,pCol+1] == 0:
            new_mask[pRow,pCol] = 0
        else:
            new_mask[pRow,pCol] = 1

    return z * new_mask, new_mask

if __name__ == "__main__":
    # # Get Light Direction
    # L = np.loadtxt("./view_01/light_directions.txt").T
    #
    # # Create a list from the input images
    # images = []
    # start = 2
    # end = 4
    # for i in range(start, end+1):
    #     # Important: Note that we flatten the images to greyscale here
    #     image = np.array(Image.open("./view_01/00" + str(i) + ".png").convert('L')).flatten()
    #     images.append(image)
    #
    # Get Mask
    mask = np.array(Image.open("./view_01/mask.png").convert('L'))
    #
    # # Run the Photometric Stereo
    # N, rho = get_surface_normal(images, L[:, start - 1:end])
    # N[:, :, 0] = N[:, :, 0] * mask
    # N[:, :, 1] = N[:, :, 1] * mask
    # N[:, :, 2] = N[:, :, 2] * mask

    # 우영님 photometric stereo normal code
    # photometricStereo = PhotometricStereo()
    # dir_name = "./view_01"
    # file_list = os.listdir(dir_name)
    # file_list = file_list[:96]
    #
    # light_coords = np.loadtxt('./view_01/light_directions.txt')
    # light_intensities = np.loadtxt('./view_01/light_intensities.txt')
    # light_intensities = light_intensities[::, 0] * 0.299 + light_intensities[::, 1] * 0.587 + light_intensities[::,
    #                                                                                           2] * 0.114
    #
    # photometricStereo.loadImgNCoord(file_list, light_coords)
    # photometricStereo.calNormal()
    #
    # norm_stereo = photometricStereo.normal_map * 255

    # get 3D surface form normal
    norm_gt = read_data_file('./view_01/Normal_gt.mat')

    # plt.imshow(norm_stereo)
    # plt.show()
    plt.imshow(norm_gt)
    plt.show()

    # gt integration
    z = unbiased_integrate(norm_gt[:,:,0], norm_gt[:,:,1], norm_gt[:,:,2], mask, angle_threshold=np.pi/9)

    display_depth_matplotlib(z,"bear","naive_bear_gt")

    edge_removed_z, new_mask = remove_edge_mask(z, mask)

    display_depth_matplotlib(edge_removed_z,"bear","edge_rm_bear_gt")

    #
    # # stereo integration
    # z_st = unbiased_integrate(norm_gt[:, :, 0], norm_gt[:, :, 1], norm_gt[:, :, 2], mask, angle_threshold=np.pi / 9)
    #
    # display_depth_matplotlib(z_st, "bear", "naive_bear_stereo")
    #
    # edge_removed_z_st, new_mask_st = remove_edge_mask(z_st, mask)
    #
    # display_depth_matplotlib(edge_removed_z_st, "bear", "edge_rm_bear_stereo")
    #
