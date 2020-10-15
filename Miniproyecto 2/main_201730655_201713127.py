import os
import glob
import requests
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy as sp
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters.rank import percentile 
from scipy.io import loadmat
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import mean_squared_error
import nibabel as nib


# daisy = io.imread(os.path.join("lab_analisis\mini_proy_2", "daisy.jpg"))
# daisy = rgb2gray(daisy)

# noisy_daisy = io.imread(os.path.join("lab_analisis\mini_proy_2", "noisy_daisy.jpg"))
# noisy_daisy = rgb2gray(noisy_daisy)

# # funcion otorgada en la guia para crear el kernel gaussiano
# def gaussian_kernel(size, sigma):
#     size = int(size)//2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1/(2.0 * np.pi * sigma**2)
#     g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2))) * normal
#     return g

# creo las matrices de cada filtro (3.a,b,c,d)
one = np.ones((5,5))
tf = 1/25 * one
hf = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
vf = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]

def MyCorrelation_201730655_201713127(image, kernel, boundary_condition):
    correlated_image = np.zeros(image.shape[0]+(2*((kernel.shape[0]+1)/2)),image.shape[1]+(2*((kernel.shape[1]+1)/2)))

# ahora a hacer el correlate2d artesanal 
# def MyCorrelation_201730655_201713127(image, kernel): 
    
#     # se va a separar la función con las condiciones de frontera de la parte algorítmica únicamente de la correlación 
#     def algor_corr(imag, kerne):
#     # size = imag.shape # x -> 1, y -> 0 
#         size = [len(imag), len(imag[0])]
    
        
#     # nueva imagen con todos los valores en 1
#         new_img = np.ones((size[1], size[0]))

#     # coordenada para las filas
#         j = 0
#         while j < (size[1] - (len(kerne) - 1)):
#             for i in range((size[0] - (len(kerne)-1))):
#                 suma = 0
#                 for k in range(len(kerne)):
#                     for l in range(len(kerne)):
#                     # se iguala el valor del pixel de la imagen nueva a la multiplicación de los pixeles en esas coordenadas 
#                     # de la imagen base y el kernel para todos los puntos del kernel
#                         suma += imag[i+k][j+l] * kerne[k][l]
#                 new_img[i+len(kerne)//2][j+len(kerne)//2] = suma
#             j+=1

#         return new_img

#     im_final = []


#     # se pasa la imagen a formato de lista (este paso se realizará en todos los casos de fronteras)
#     a = image.tolist()

#     size_m = len(a)
#     ker = kernel
#     len_ker_1 = len(ker)-1

#     # los siguientes ciclos añadirán 2 ceros en todas las direcciones de los bordes de la matriz
#     for i in range(len_ker_1):
#         a.append(np.zeros((size_m,), dtype='int').tolist())
#         a.insert(0, np.zeros((size_m,), dtype='int').tolist())

#     nsize = len(a)

#     for i in range(nsize):
#         for j in range(len_ker_1):
#             a[i].append(0)
#             a[i].insert(0,0)
        
#     # se iguala la imagen nueva y se ajusta el tamaño
#     im_final = algor_corr(a, kernel)[1:-1, 1:-1]
#     return im_final 


# def MyCorrelation_201730655_201713127(image, kernel, boundary_condition):

#     # se va a separar la función con las condiciones de frontera de la parte algorítmica únicamente de la correlación 
#     def algor_corr(imag, kerne):
#     # size = imag.shape # x -> 1, y -> 0 
#         size = [len(imag), len(imag[0])]
#     # nueva imagen con todos los valores en 1
#         new_img = np.ones((size[1], size[0]))

#     # coordenada para las filas
#         j = 0
#         while j < (size[0] - (len(kerne) - 1)):
#             for i in range((size[1] - (len(kerne)-1))):
#                 suma = 0
#                 for k in range(len(kerne)):
#                     for l in range(len(kerne)):
#                     # se iguala el valor del pixel de la imagen nueva a la multiplicación de los pixeles en esas coordenadas 
#                     # de la imagen base y el kernel para todos los puntos del kernel
#                         suma += imag[i+k][j+l] * kerne[k][l]
#                 new_img[i+len(kerne)//2][j+len(kerne)//2] = suma
#             j+=1

#         return new_img

#     im_final = []

#     # condición para el valor de frontera 'fill'
#     if(boundary_condition == 'fill'):
#         # se pasa la imagen a formato de lista (este paso se realizará en todos los casos de fronteras)
#         a = image.tolist()

#         size_m = len(a)
#         ker = kernel
#         len_ker_1 = len(ker)-1

#         # los siguientes ciclos añadirán 2 ceros en todas las direcciones de los bordes de la matriz
#         for i in range(len_ker_1):
#             a.append(np.zeros((size_m,), dtype='int').tolist())
#             a.insert(0, np.zeros((size_m,), dtype='int').tolist())

#         nsize = len(a)

#         for i in range(nsize):
#             for j in range(len_ker_1):
#                 a[i].append(0)
#                 a[i].insert(0,0)
        
#         # se iguala la imagen nueva y se ajusta el tamaño
#         im_final = algor_corr(a, kernel)[1:-1, 1:-1]

#     # condición para el valor de frontera 'wrap'
#     elif (boundary_condition == 'wrap'):

#         a = image.tolist()

#         # se procede a reflejar las dos primeras y últimas filas y columnas como condiciones de frontera (como si siguiera la imagen infinitamente)
#         a.insert(0, a[-1])
#         a.insert(0, a[-2])
#         a.append(a[2])
#         a.append(a[3])

#         for i in range(len(a[0])):
#             a[i].insert(0, a[i][-1])
#             a[i].insert(0, a[i][-2])
#             a[i].append(a[i][2])
#             a[i].append(a[i][3])
#         # se obtiene la imagen final y se ajusta el tamaño
#         im_final = algor_corr(a, kernel)[1:-1, 1:-1]

#     # condición para el valor de frontera 'valid'
#     elif (boundary_condition == 'valid'):
#         a = image.tolist()

#         size_ms= len(a)
#         # se repiten la primera y última fila y columna
#         a.insert(-1, a[-1])
#         a.insert(0, a[0])

#         for i in range(len(a)):
#             a[i].insert(0, a[i][0])
#             a[i].append(a[i][-1])
#         # se obtiene la imagen final y se ajusta el tamaño
#         im_final = algor_corr(a, kernel)[2:-2, 2:-2]

#     return im_final



# # se carga la matriz groundtruth
# mat = os.path.join("lab_analisis\mini_proy_2", "ccorrelation_test.mat")
# anot = loadmat(mat)
# matrix = anot['matrix']
# mat_kernel = anot['kernel']


# # se prueban los filtros con las diferentes condiciones de frontera para el kernel dado de dimensiones 5x5
# filt_mat = MyCorrelation_201730655_201713127(matrix, mat_kernel, 'fill')
# cor2_mat = sp.signal.correlate2d(matrix, mat_kernel, boundary='fill')

# filt_mat_w = MyCorrelation_201730655_201713127(matrix, mat_kernel, 'wrap')
# cor2_mat_w = sp.signal.correlate2d(matrix, mat_kernel, boundary='wrap')

# cor2_mat_s = sp.signal.correlate2d(matrix, mat_kernel, mode='valid')
# filt_mat_s = MyCorrelation_201730655_201713127(matrix, mat_kernel, 'valid')


# # se genera el subplot comparando los filtros realizados con MyCorrelate con la funcion correlate2d
# figs, axs = plt.subplots(2,4, figsize=(15, 8))
# figs.tight_layout(pad=3.0)

# axs[0][0].imshow(matrix, cmap='gray')
# axs[0][0].set_title('matriz original')

# figs.delaxes(axs[1,0])

# axs[0][1].imshow(filt_mat, cmap='gray')
# axs[0][1].set_title('Fill MyCorrelation')

# axs[1][1].imshow(cor2_mat, cmap='gray')
# axs[1][1].set_title('Fill correlate2d')

# axs[0, 2].imshow(filt_mat_w, cmap='gray')
# axs[0, 2].set_title('Wrap MyCorrelation')

# axs[1,2].imshow(cor2_mat_w, cmap='gray')
# axs[1,2].set_title('Wrap correlate2d')

# axs[0, 3].imshow(filt_mat_s, cmap='gray')
# axs[0, 3].set_title('Valid MyCorrelation')

# axs[1,3].imshow(cor2_mat_s, cmap='gray')
# axs[1,3].set_title('Valid correlate2d')

# plt.savefig("correlate2dvsMyCorrelate.jpg")
# plt.show()

# input('press enter to continue...')

# # filtro con Mycorrelation (se eligió el filtro 3c) para la imagen 'daisy.jpg'
# my_hf = MyCorrelation_201730655_201713127(daisy, hf, 'fill')
# corr2_hf = sp.signal.correlate2d(daisy, hf, boundary='fill')

# # se genera el subplot para la comparación de la implementación de MyCorrelation vs correlate2d
# figs, axs = plt.subplots(1, 3, figsize=(15, 8))
# figs.tight_layout(pad=3.0)

# axs[0].imshow(daisy, cmap='gray')
# axs[0].set_title('imagen daisy original')
# axs[0].axis('off')

# axs[1].imshow(my_hf, cmap='gray')
# axs[1].set_title('Fill 3c daisy MyCorrelation')
# axs[1].axis('off')

# axs[2].imshow(corr2_hf, cmap='gray')
# axs[2].set_title('Fill 3c daisy correlate2d')
# axs[2].axis('off')

# plt.savefig("filtro3cComparacion.jpg")
# plt.show()

# input('press enter to continue...')

# # error cuadratico medio para el filtro anterior 
# print(mean_squared_error(my_hf, corr2_hf))



# # filtro con Mycorrelation (con los filtros 3a, 3b) para la imagen 'noisy_daisy.jpg'
# my_ones = MyCorrelation_201730655_201713127(noisy_daisy, one, 'fill')
# corr2_ones = sp.signal.correlate2d(noisy_daisy, one, boundary='fill')

# my_tf = MyCorrelation_201730655_201713127(noisy_daisy, tf, 'fill')
# corr2_tf = sp.signal.correlate2d(noisy_daisy, tf, boundary='fill')

# # se genera el subplot para la comparación de la implementación de MyCorrelation vs correlate2d para los filtros 3a y 3b
# figs, axs = plt.subplots(2, 3, figsize=(15, 8))
# figs.tight_layout(pad=3.0)

# axs[0,0].imshow(noisy_daisy, cmap='gray')
# axs[0,0].set_title('imagen noisy_daisy original')
# axs[0,0].axis('off')

# figs.delaxes(axs[1,0])

# axs[0,1].imshow(my_ones, cmap='gray')
# axs[0,1].set_title('Fill 3a noisy_daisy MyCorrelation')
# axs[0,1].axis('off')

# axs[1,1].imshow(corr2_ones, cmap='gray')
# axs[1,1].set_title('Fill 3a noisy_daisy correlate2d')
# axs[1,1].axis('off')

# axs[0,2].imshow(my_tf, cmap='gray')
# axs[0,2].set_title('Fill 3b noisy_daisy MyCorrelation')
# axs[0,2].axis('off')

# axs[1,2].imshow(corr2_tf, cmap='gray')
# axs[1,2].set_title('Fill 3b noisy_daisy correlate2d')
# axs[1,2].axis('off')

# plt.savefig("3a3bcompara.jpg")
# plt.show()

# input('press enter to continue...')



# # aplicando kernel gaussiano
# gauss = gaussian_kernel(5, 1)

# my_gauss_1 = MyCorrelation_201730655_201713127(noisy_daisy, gauss, 'fill')
# corr2_gauss_1 = sp.signal.correlate2d(noisy_daisy, gauss, boundary='fill')

# # visualización del kernel gaussiano junto al kernel 3a
# figs, axs = plt.subplots(2, 2, figsize=(10, 8))
# figs.tight_layout(pad=3.0)

# axs[0,0].imshow(my_gauss_1, cmap='gray')
# axs[0,0].set_title('gauss5x5_1 noisy_daisy MyCorrelation')
# axs[0,0].axis('off')

# axs[1,0].imshow(corr2_gauss_1, cmap='gray')
# axs[1,0].set_title('gauss5x5_1 noisy_daisy correlate2d')
# axs[1,0].axis('off')

# axs[0,1].imshow(my_ones, cmap='gray')
# axs[0,1].set_title('Fill 3a noisy_daisy MyCorrelation')
# axs[0,1].axis('off')

# axs[1,1].imshow(corr2_ones, cmap='gray')
# axs[1,1].set_title('Fill 3a noisy_daisy correlate2d')
# axs[1,1].axis('off')

# plt.savefig("gaussvs3a.jpg")
# plt.show()

# input('press enter to continue...')


# # 3 filtros gaussianos con tamaño fijo y sigma variable
# gauss_3 = gaussian_kernel(5, 3)
# gauss_7 = gaussian_kernel(5, 7)
# gauss_11 = gaussian_kernel(5, 11)

# my_gauss_3 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_3, 'fill')
# my_gauss_7 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_7, 'fill')
# my_gauss_11 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_11, 'fill')

# # visualización de los 3 filtros gaussianos con igual tamaño y sigma diferente
# figs, axs = plt.subplots(1, 3, figsize=(15, 6))
# figs.tight_layout(pad=3.0)

# axs[0].imshow(my_gauss_3, cmap='gray')
# axs[0].set_title('gauss5x5_3 noisy_daisy MyCorrelation')
# axs[0].axis('off')

# axs[1].imshow(my_gauss_7, cmap='gray')
# axs[1].set_title('gauss5x5_7 noisy_daisy MyCorrelation')
# axs[1].axis('off')

# axs[2].imshow(my_gauss_11, cmap='gray')
# axs[2].set_title('gauss5x5_7 noisy_daisy MyCorrelation')
# axs[2].axis('off')

# plt.savefig("3gaussmismot.jpg")
# plt.show()

# input('press enter to continue...')




# # 3 filtros gaussianos con sigma fijo y tamaño variable
# gauss_t1 = gaussian_kernel(3, 1)
# gauss_t2 = gaussian_kernel(7, 1)
# gauss_t3 = gaussian_kernel(11, 1)

# my_gauss_t1 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_t1, 'fill')
# my_gauss_t2 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_t2, 'fill')
# my_gauss_t3 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_t3, 'fill')

# # visualización de los 3 filtros gaussianos con igual tamaño y sigma diferente
# figs, axs = plt.subplots(1, 3, figsize=(15, 6))
# figs.tight_layout(pad=3.0)

# axs[0].imshow(my_gauss_t1, cmap='gray')
# axs[0].set_title('gauss3x3_1 noisy_daisy MyCorrelation')
# axs[0].axis('off')

# axs[1].imshow(my_gauss_t2, cmap='gray')
# axs[1].set_title('gauss7x7_1 noisy_daisy MyCorrelation')
# axs[1].axis('off')

# axs[2].imshow(my_gauss_t3, cmap='gray')
# axs[2].set_title('gauss11x11_1 noisy_daisy MyCorrelation')
# axs[2].axis('off')

# plt.savefig("3gaussmismos.jpg")
# plt.show()

# input('press enter to continue...')



# # reproducción de la figura 4 con los filtros 3c y 3d 
# #3d
# my_vf = MyCorrelation_201730655_201713127(daisy, vf, 'fill')
# # en un punto anterior se realizó la cross-correlación con el filtro 3c

# # se genera el subplot para la reproducción de la figura 4
# figs, axs = plt.subplots(1, 2, figsize=(10, 8))
# figs.tight_layout(pad=3.0)

# axs[0].imshow(my_vf, cmap='gray')
# axs[0].set_title('Fill 3d daisy MyCorrelation')
# axs[0].axis('off')

# axs[1].imshow(my_hf, cmap='gray')
# axs[1].set_title('Fill 3c daisy MyCorrelation')
# axs[1].axis('off')

# plt.savefig("figura4repro.jpg")
# plt.show()

# input('press enter to continue...')


# # se procede a reproducir las imágenes de la figura 5 únicamente con filtro gaussiano y con los kernels 3c y 3d
# gauss_4 = gaussian_kernel(5, 4)
# my_gauss_4 = MyCorrelation_201730655_201713127(noisy_daisy, gauss_4, 'fill')
# f5_a = my_hf * my_vf

# plt.imshow(f5_a, cmap='gray')

# plt.show()



# # REFERENCIAS 

# # --> correlate2d: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html



#PROBLEMA BIOMEDICO
#Ruta de la imagen
df_Train = glob.glob(os.path.join('Train','*.png'))
print(df_Train)

imagenes = []
for i in range(0,len(df_Train)):
    imagenes.append(rgb2gray(io.imread(df_Train[i])))

for i in range(0,len(imagenes)):
    imagenes[i] = 255*imagenes[i]
    imagenes[i] = imagenes[i].astype(np.uint8) 

#Equalización de histograma
def MyHistEq_201730655_201713127(hist):
    
    histeq = hist
    comparacion = np.arange(256)
    cant_num = 0
    cant_pixel = []
    pixeles = []
    proba = []
    Sk =[]
    table = {}

    #Cuenta de los pixeles con la misma intensidad
    for i in range(0,len(comparacion)):
        cant_num = np.count_nonzero(hist == comparacion[i])
        if cant_num != 0:
            cant_pixel.append(cant_num)
            pixeles.append(comparacion[i])

    suma = 0
    aux = 0
    #Calculo de Sk
    for i in range(0,len(cant_pixel)):
        proba.append(cant_pixel[i]/len(hist))
        suma = cant_pixel[i] + aux
        Sk.append(np.round(((256-1)/len(hist))*suma))
        aux = suma
    
    #Nuevo histograma
    for i in range(0,len(histeq)):
        for j in range(0,len(pixeles)):
            if histeq[i]==pixeles[j]:
                histeq[i] = Sk[j]
                break
    
    
    for i in range(0,len(hist)):
        table[str(hist[i])] = str(histeq[i]) 
    
    return histeq, table


#Histograma target
hist_img1_target = MyHistEq_201730655_201713127(imagenes[0].flatten())[0]
for i in range(0,len(hist_img1_target)):
    if hist_img1_target[i] < 170:
        hist_img1_target[i] = 170
    elif hist_img1_target[i] > 230:
        hist_img1_target[i] = 230

        

#Especificacion
def MyHistEsp_201730655_201713127(image, target_hist):
    eq_target_hist = MyHistEq_201730655_201713127(target_hist)[0]
    eq_image = MyHistEq_201730655_201713127(image)[0]
    Sk =[]
    Vq = []
    pixeles_th = []
    pixeles_eqim = []
    new_pixeles = []
    cant_num = 0
    comparacion = np.arange(256)
    new_image = eq_image
    for i in range(0,len(comparacion)):
        cant_num = np.count_nonzero(eq_target_hist == comparacion[i])
        if cant_num != 0:
            pixeles_th.append(comparacion[i]) 

    cant_num = 0   
    for i in range(0,len(comparacion)):
        cant_num = np.count_nonzero(eq_image == comparacion[i])
        if cant_num != 0:
            pixeles_eqim.append(comparacion[i])
            new_pixeles.append(comparacion[i])
    

    for i in  range(0,len(pixeles_eqim)):
        for j in range(0,len(pixeles_th)):
            if pixeles_eqim[i] <= pixeles_th[j]:
                new_pixeles[i] = pixeles_th[j]
                break
    
    for i in range(0,len(new_image)):
        for j in range(0,len(new_pixeles)):
            if new_image[i]==pixeles_eqim[j]:
                new_image[i] = new_pixeles[j]
                break
    
    return new_image

imagen1_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[0].flatten())[0],(imagenes[0].shape[0],imagenes[0].shape[1]))
imagen1_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes[0].flatten(),hist_img1_target),(imagenes[0].shape[0],imagenes[0].shape[1]))


# #Gráfica
# input("Press Enter to continue...")
# figx, ax1 = plt.subplots(4, 2, figsize=(10, 12), gridspec_kw={'wspace':0.5,'hspace':0.7})
# ax1[0][0].set_title('Target hist')
# ax1[0][0].hist(hist_img1_target,bins=255)
# ax1[1][0].set_title('Imágen original')
# ax1[1][0].imshow(imagenes[0], cmap='gray')
# ax1[1][0].axis('off')
# ax1[2][0].set_title('Imagen equalizada')
# ax1[2][0].imshow(imagen1_eq, cmap = 'gray')
# ax1[2][0].axis('off')
# ax1[3][0].set_title('Imagen especificada')
# ax1[3][0].imshow(imagen1_esp,cmap='gray')
# ax1[3][0].axis('off')
# ax1[0][1].axis('off')
# ax1[1][1].set_title('Histograma')
# ax1[1][1].hist(imagenes[0].flatten(), bins=255)
# ax1[2][1].set_title('Histograma equalizado')
# ax1[2][1].hist(MyHistEq_201730655_201713127(imagenes[0].flatten())[0],bins=255)
# ax1[3][1].set_title('Histograma especificado')
# ax1[3][1].hist(imagen1_esp.flatten(),bins=255)
# plt.show()

# # #Gráfica2
# # imagen2_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[1].flatten())[0],(163,136))
# # input("Press Enter to continue...")
# # fig2, ax2 = plt.subplots(4, 2, figsize=(10, 12), gridspec_kw={'wspace':0.5,'hspace':0.7})
# # ax2[0][0].set_title('Target hist')
# # ax2[0][0].hist(tar_hist,bins=255)
# # ax2[1][0].set_title('Imágen original')
# # ax2[1][0].imshow(imagenes[1], cmap='gray')
# # ax2[1][0].axis('off')
# # ax2[2][0].set_title('Imagen equalizada')
# # ax2[2][0].imshow(imagen2_eq, cmap = 'gray')
# # ax2[2][0].axis('off')
# # ax2[3][0].set_title('Imagen especificada')
# # # ax1[3][0].imshow(imagen_eq)
# # ax2[3][0].axis('off')
# # # ax1[0][1].set_title('Histograma equalizado')
# # # ax1[0][1].hist()
# # ax2[0][1].axis('off')
# # ax2[1][1].set_title('Histograma')
# # ax2[1][1].hist(imagenes[1].flatten(), bins=255)
# # ax2[2][1].set_title('Histograma equalizado')
# # ax2[2][1].hist(MyHistEq_201730655_201713127(imagenes[1].flatten())[0],bins=255)
# # ax2[3][1].set_title('Histograma especificado')
# # # ax1[3][1].hist()
# # ax2[3][1].axis('off')
# # plt.show()

#PARTE 2 
# imagen1_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[0].flatten())[0],(imagenes[0].shape[0],imagenes[0].shape[1]))
# imagen2_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[1].flatten())[0],(imagenes[1].shape[0],imagenes[1].shape[1]))

# # imagen1_cor = sp.signal.correlate2d(imagen1_eq, hf, boundary='fill')
# # imagen2_cor = sp.signal.correlate2d(imagen2_eq, hf, boundary='fill')
# # plt.imshow(imagen1_cor,cmap='gray')
# # plt.show()
# # plt.imshow(imagen2_cor,cmap='gray')
# # plt.show()
# imagen1_corart = MyCorrelation_201730655_201713127(imagen1_eq, hf)
# imagen2_corart = MyCorrelation_201730655_201713127(imagen2_eq, hf)
# plt.imshow(imagen1_corart,cmap='gray')
# plt.show()
# plt.imshow(imagen2_corart,cmap='gray')
# plt.show()