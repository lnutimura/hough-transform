# author: Luan Nunes Utimura
# Processamento de Imagens Digitais - PPGCC 2018

import cv2
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from scipy import ndimage

# Função para a aplicação do Sobel
# em uma determinada imagem (img);
def sobelFilter(img):
    sobelFilter1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelFilter2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    conv1 = cv2.filter2D(img, -1, sobelFilter1)
    conv2 = cv2.filter2D(img, -1, sobelFilter2)
    
    filteredImg = np.sqrt(np.power(conv1, 2.0) + np.power(conv2, 2.0))
    filteredImg = filteredImg.astype(int)
    
    return filteredImg

# Função para a aplicação da binarização
# em uma determinada imagem (img);
def thresholdFilter(img, threshold = 150):
    img = np.where(img > threshold, 255, 0)

    return img

# Função auxiliar para verificar se um determinado ponto
# está dentro de uma região válida da imagem;
def inImgBoundaries(x, y, height, width):
    return x >= 0 and x < height and y >= 0 and y < width

# Função que realiza a transformada de Hough (circular);
def circleHoughTransform(img, threshold = 5):
    height, width = img.shape
    print(height, width)

    radius = np.arange(58, 64)
    len_radius = len(radius)

    thetas = np.deg2rad(np.arange(0, 360.0, 1.0))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    len_thetas = len(thetas)

    accumulator = np.zeros((height, width), dtype=np.uint8)

    are_edges = img > threshold
    x_indexes, y_indexes = np.nonzero(are_edges)

    for i in tqdm(range(len(x_indexes))):
        x = x_indexes[i]
        y = y_indexes[i]
        
        for r in range(len_radius):
            for t in range(len_thetas):
                a = x - radius[r] * cos_thetas[t]
                b = y - radius[r] * sin_thetas[t]
                if a >= 0 and a < height and b >= 0 and b < width:
                    accumulator[int(a), int(b)] += 1

    return accumulator, thetas, radius

# Função que desenha e encontra os círculos a partir
# da imagem do espaço de Hough;
def displayHoughTransformCircles(img, radius, grayscaleReference, houghReference):
    height, width = grayscaleReference.shape

    controlMatrix = np.zeros((height, width), dtype=np.uint8)

    outputImg1 = np.zeros((height, width), dtype=np.uint8)
    outputImg2 = grayscaleReference.copy()

    x, y = np.nonzero(img)
    r = int((radius[0] + radius[-1]) / 2)
    
    candidatesList = []
    for i in range(len(x)):
        if controlMatrix[int(x[i]), int(y[i])] == 0:
            window = houghReference[(int(x[i]) - 10):(int(x[i]) + 10 + 1), (int(y[i]) - 10):(int(y[i]) + 10 + 1)]
            
            centerOfMass = ndimage.measurements.center_of_mass(window)
            com_x = int(centerOfMass[0] + x[i] - 10)
            com_y = int(centerOfMass[1] + y[i] - 10)
            
            controlMatrix[(int(x[i]) - 10):(int(x[i]) + 10 + 1), (int(y[i]) - 10):(int(y[i]) + 10 + 1)] = 1
            
            cv2.circle(outputImg1, (com_y, com_x), 2, 255, -1)
            cv2.circle(outputImg2, (com_y, com_x), 2, 255, -1)

            cv2.circle(outputImg1, (com_y, com_x), r, 255, 2)
            cv2.circle(outputImg2, (com_y, com_x), r, 255, 2)
            
            candidatesList.append([com_x, com_y, r])

    # Para cada círculo candidato, desenha ele sobre a imagem grayscale;
    print('circle\t(x, y)\tr')
    for i, candidate in enumerate(candidatesList):
        centerX, centerY, centerR = candidate
        print('{}\t{}\t{}'.format(i, (centerX, centerY), centerR))

    return outputImg1, outputImg2


if __name__ == '__main__':
    # Segunda Parte (Circular Hough Transform);
    
    # Carrega a imagem;
    img = cv2.imread('circles.png')
    # Converte de BGR (padrão OpenCV) p/ RGB;
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Converte de RGB p/ Escala de Cinza;
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplica todos os filtros/métodos necessários
    # para as plotagens de cada etapa;
    filtered_img = cv2.Canny(grayscale_img, 100, 200)

    plt.figure(1)
    # Imagem original (colorida);
    plt.subplot(231)
    plt.imshow(rgb_img)
    # Imagem em escala de cinza;
    plt.subplot(232)
    plt.imshow(grayscale_img, cmap='gray')
    # Imagem após a aplicação do filtro Sobel;
    plt.subplot(233)
    plt.imshow(filtered_img, cmap='gray')
    
    plt.figure(2)
    # Imagem após a aplicação da Transformada de Hough;
    plt.subplot(121)
    hough_img, thetas, radius = circleHoughTransform(filtered_img)
    plt.imshow(hough_img, cmap='jet')
    # Imagem após a aplicação da Binarização na Transformada de Hough;;
    plt.subplot(122)
    thresholded_hough_img = thresholdFilter(hough_img, 240)
    plt.imshow(thresholded_hough_img, cmap='gray')
   
    plt.figure(1)
    # Imagem após a identificação dos círculos correspondentes aos picos;
    plt.subplot(234)
    houghCircles_img1, houghCircles_img2 = displayHoughTransformCircles(thresholded_hough_img, radius, grayscale_img, hough_img)
    plt.imshow(houghCircles_img1, cmap='gray') 
    plt.subplot(235)
    plt.imshow(houghCircles_img2, cmap='gray')

    plt.show()
