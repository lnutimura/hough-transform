# @author: Luan Nunes Utimura
# Processamento de Imagens Digitais - PPGCC 2018

import cv2
import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm

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

# Função que realiza a transformada de Hough (linear);
def lineHoughTransform(img, threshold = 5):
    height, width = img.shape
    diagonalLength = int(round(np.sqrt(height * height + width * width)))
    
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    rhos = np.linspace(-diagonalLength, diagonalLength, 2 * diagonalLength)
    
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    len_thetas = len(thetas)

    accumulator = np.zeros((2 * diagonalLength, len_thetas), dtype=np.uint8)

    are_edges = img > threshold
    x_indexes, y_indexes = np.nonzero(are_edges)

    for i in tqdm(range(len(x_indexes))):
        x = x_indexes[i]
        y = y_indexes[i]

        for t in range(len_thetas):
            rho = diagonalLength + int(round(y * cos_thetas[t] + x * sin_thetas[t]))
            accumulator[rho, t] += 1
    
    return accumulator, thetas, rhos

# Função que desenha e encontra os segmentos de reta a partir
# da imagem do espaço de Hough;
def displayHoughTransformLines(img, thetas, rhos, reference, reference_edges):
    height, width = reference.shape

    outputImg1 = np.zeros((height, width), dtype=np.uint8)
    outputImg2 = reference.copy()

    diagonalLength = int(round(np.sqrt(height * height + width * width)))
    
    r, t = np.nonzero(img)
    linesList = []

    for i in range(len(r)):
        a = np.cos(thetas[t[i]])
        b = np.sin(thetas[t[i]])

        x0 = int(round(a * rhos[r[i]]))
        y0 = int(round(b * rhos[r[i]]))
        
        y = np.linspace(int(x0 + 1000 * (-b)), int(x0 - 1000 * (-b)), 2000 + 1)
        x = np.linspace(int(y0 + 1000 * a), int(y0 - 1000 * a), 2000 + 1)
        
        linesList.append([x, y])

        cv2.line(outputImg1, (int(y[0]), int(x[0])), (int(y[-1]), int(x[-1])), 255, 1)
   

    candidatesList = []

    # Para cada linha;
    for line in linesList:
        x, y = line
        
        
        minLength = 50
        currentLength = 0
        
        maxTolerance = 8
        currentTolerance = 0

        currentlyDrawing = False
        
        tmpX, tmpY = 0, 0

        # Para cada ponto da linha;
        for i in range(2001):
            # Se está desenhando;
            if currentlyDrawing:
                # Se é um ponto válido e realmente existe uma borda,
                # incrementa o comprimento da linha;
                if inImgBoundaries(int(x[i]), int(y[i]), height, width) and reference_edges[int(x[i]), int(y[i])] == 255:
                    currentLength += 1
                # Se é um ponto válido e não existe uma borda,
                # verifica se há tolerância para continuar incrementando o comprimento da linha;
                elif inImgBoundaries(int(x[i]), int(y[i]), height, width) and reference_edges[int(x[i]), int(y[i])] != 255:
                    # Se a tolerância atual for menor do que a tolerância máxima,
                    # incrementa o comprimento da linha;
                    if currentTolerance < maxTolerance:
                        currentLength += 1
                        currentTolerance += 1
                    # Senão, considera que terminou o desenho desta linha.
                    # Todavia, ela pode ser um candidato;
                    else:
                        if currentLength >= minLength:
                            candidatesList.append([tmpX, tmpY, x[i], y[i]])
                        currentlyDrawing = False
                        currentLength = 0
                        currentTolerance = 0
                # Se não é um ponto válido (passou das extremidades da imagem);
                else:
                    if currentLength >= minLength:
                        candidatesList.append([tmpX, tmpY, x[i], y[i]])
                    currentlyDrawing = False
                    currentLength = 0
                    currentTolerance = 0
            # Se não está desenhando;
            else:
                # Se é um ponto válido e realmente existe uma borda,
                # começa a desenhar e registra o primeiro ponto desta linha;
                if inImgBoundaries(int(x[i]), int(y[i]), height, width) and reference_edges[int(x[i]), int(y[i])] == 255:
                    currentlyDrawing = True
                    currentLength += 1

                    tmpX, tmpY = x[i], y[i]

    # Para cada linha candidata, desenha ela sobre a imagem grayscale;
    print('line\tfrom (r, c)\tto (r, c)')
    for i, candidate in enumerate(candidatesList):
        iniX, iniY, endX, endY = candidate

        if int(iniX) == -1: iniX = 0
        if int(iniY) == -1: iniY = 0
        if int(endX) == -1: endX = 0
        if int(endY) == -1: endY = 0

        print('{}\t{}\t{}'.format(i, (int(iniX), int(iniY)), (int(endX), int(endY))))
        cv2.line(outputImg2, (int(iniY), int(iniX)), (int(endY), int(endX)), 255, 2)

    return outputImg1, outputImg2


if __name__ == '__main__':
    # Primeira Parte (Linear Hough Transform);
    
    # Carrega a imagem;
    img = cv2.imread('sudoku.jpg')
    # Converte de BGR (padrão OpenCV) p/ RGB;
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Converte de RGB p/ Escala de Cinza;
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplica todos os filtros/métodos necessários
    # para as plotagens de cada etapa;
    filtered_img = sobelFilter(grayscale_img)
    thresholded_img = thresholdFilter(filtered_img)

    plt.figure(1)
    # Imagem original (colorida);
    plt.subplot(321)
    plt.imshow(rgb_img)
    # Imagem em escala de cinza;
    plt.subplot(322)
    plt.imshow(grayscale_img, cmap='gray')
    # Imagem após a aplicação do filtro Sobel;
    plt.subplot(323)
    plt.imshow(filtered_img, cmap='gray')
    # Imagem após a aplicação da binarização;
    plt.subplot(324)
    plt.imshow(thresholded_img, cmap='gray')
    dilated_thresholded_img = cv2.dilate(thresholded_img.astype(np.float32), np.ones((3, 3)), iterations=1).astype(np.uint8)

    plt.figure(2)
    # Imagem após a aplicação da Transformada de Hough;
    plt.subplot(121)
    hough_img, thetas, rhos = lineHoughTransform(thresholded_img)
    plt.imshow(hough_img, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]], aspect='auto')
    # Imagem após a aplicação da Binarização na Transformada de Hough;;
    plt.subplot(122)
    thresholded_hough_img = thresholdFilter(hough_img, 200)
    plt.imshow(thresholded_hough_img, cmap='gray', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]], aspect='auto')
   
    plt.figure(1)
    # Imagem após a identificação das retas correspondentes aos picos;
    plt.subplot(325)
    houghLines_img1, houghLines_img2 = displayHoughTransformLines(thresholded_hough_img, thetas, rhos, grayscale_img, dilated_thresholded_img)
    plt.imshow(houghLines_img1, cmap='gray') 
    plt.subplot(326)
    plt.imshow(houghLines_img2, cmap='gray')

    plt.show()
