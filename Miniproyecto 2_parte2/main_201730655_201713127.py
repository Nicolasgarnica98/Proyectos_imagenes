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
from scipy.signal import correlate2d

# # se carga la imagen noisy_1 y se pone en escala de grises
# noisy_1 = io.imread(os.path.join("lab_analisis\mini_proy_2\imgs_p6", "noisy_1.jpg"))
# noisy_1 = rgb2gray(noisy_1)

# # se carga la imagen noisy_2 y se pone en escala de grises
# noisy_2 = io.imread(os.path.join("lab_analisis\mini_proy_2\imgs_p6", "noisy_2.jpg"))
# noisy_2 = rgb2gray(noisy_2)

# # definición de la función para el filtro de mediana adaptativo
# def MyAdaptMedian_201730655_201713127(image, window_size, max_window_size):

#     # se crean variables para el tamaño de la imagen, tamaño de la ventana y máximo tamaño de la ventana
#     size = image.shape
#     ws = window_size
#     mws = max_window_size
#     # kernel: matriz de 1s con el tamaño especificado en los parametros
#     kernel = np.ones((window_size, window_size))

#     # se crea una nueva imagen con todos los valores en 1
#     new_img = np.ones((size[0], size[1]))

#     # se inicia el ciclo 
#     j = 0
#     while j < (size[1] - (ws - 1)):

#         for i in range((size[0] - (ws - 1))):
#             # se inicializa un array vacío del que después se sacarán los datos para calcular los 
#             # valores necesarios para las condiciones del filtro
#             array = []
#             ws = window_size
#             kernel = np.ones((ws, ws))
            
#             for k in range(len(kernel)):
#                 for l in range(len(kernel)):
#                     # se llena un array con todos los datos de los pixeles que estén dentro del kernel
#                     # para luego sacar la mediana de estos y escoger ese valor para el pixel central
#                     if(i+k >= size[0] or j+l >= size[1]):
#                         break
#                     array.append(image[i+k][j+l])
            
#             # los valores booleanos se usarán para correr el ciclo en el que se probarán las condiciones
#             etapaa = False
#             etapab = False

#             # empieza el ciclo de condiciones
#             while (etapaa == False):
                
#                 # del array obtenido anteriormente se sacan los valores mínimo, máximo y la mediana
#                 z_med = int(np.median(array))
#                 z_min = int(np.min(array))
#                 z_max = int(np.max(array))
                
#                 # se obtiene el pixel central, y se condiciona para que la ventana no salga de los bordes de la imagen
#                 if (i + ws >= size[0] or j + ws >= size[1]):
#                     z_xy = int(image[i+(ws//2)-ws][j+(ws//2)-ws])
#                 else: 
#                     z_xy = int(image[i+len(kernel)//2][j+len(kernel)//2])

#                 # se sacan los límites para saber si el valor de la ventana y el pixel central es ruido o no
#                 A1 = int((z_med - z_min))
#                 A2 = int(int(z_med - z_max))
#                 B1 = int(z_xy) - z_min
#                 B2 = int(z_xy) - z_max
                
#                 # si el valor de la ventana no es de ruido, se procede a la etapa B
#                 if (A1 > 0 and A2 < 0):
#                     etapaa = True
#                     etapab = True
        
#                 else:
#                     # se aumenta el tamaño de la ventana, por lo que se tiene que construir nuevamente el array de 
#                     # los pixeles pertenecientes a esta
#                     ws += 1
#                     kernel = np.ones((ws, ws))
#                     array = []

#                     for k in range(len(kernel)):
#                         for l in range(len(kernel)):    
    
#                             if(i+k >= size[0] or j+l >= size[1]):
#                                 break
#                             array.append(image[i+k][j+l])

#                     # si el tamaño de la ventana no excede el tamaño máximo establecido, continúa la función
#                     if (ws <= mws):
#                         etapaa = False
#                         etapab = False
#                         continue

#                     # si el tamaño máximo de la ventana excede el tamaño máximo establecido, el valor en i,j será la mediana del array
#                     if (ws > mws):
#                         etapaa = True
#                         etapab = False
#                         new_img[i][j] = z_med

    
#                 if (etapab == True):
                    
#                     # si el valor del pixel central no es un valor de ruido se usará este valor para el pixel en i,j
#                     if(B1 > 0 and B2 < 0):
#                         etapaa = True
#                         new_img[i][j] = z_xy

#                     # si el valor del pixel central es un valor de ruido, se usará el valor de la mediana para el pixel i,j
#                     else:
#                         etapaa = True
#                         new_img[i][j] = z_med

#                     # se reestablecen los valores booleanos de la etapa A y la etapa B para que salgan del loop y se siga
#                     # con el siguiente valor en la imagen
#                     etapaa = True
#                     etapab = False
            
#         j+=1
    
#     return new_img
    
# # funcion otorgada en la guia para crear el kernel gaussiano
# def gaussian_kernel(size, sigma):
#     size = int(size)//2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1/(2.0 * np.pi * sigma**2)
#     g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2))) * normal
#     return g

# gauss_1 = gaussian_kernel(3, 11)
# gauss_2 = gaussian_kernel(7, 2)

# filt_n1_g1 = correlate2d(noisy_1, gauss_1, boundary='fill')
# filt_n1_g2 = correlate2d(noisy_1, gauss_2, boundary='fill')

# filt_n2_g1 = correlate2d(noisy_2, gauss_1, boundary='fill')
# filt_n2_g2 = correlate2d(noisy_2, gauss_2, boundary='fill')

# filt_n_1_3x7 = MyAdaptMedian_201730655_201713127(noisy_1, 3, 7)
# filt_n_1_5x9 = MyAdaptMedian_201730655_201713127(noisy_1, 5, 9)
# filt_n_1_5x11 = MyAdaptMedian_201730655_201713127(noisy_1, 5, 11)

# filt_n_2_3x7 = MyAdaptMedian_201730655_201713127(noisy_2, 3, 7)
# filt_n_2_5x9 = MyAdaptMedian_201730655_201713127(noisy_2, 5, 9)
# filt_n_2_5x11 = MyAdaptMedian_201730655_201713127(noisy_2, 5, 11)

# # inicio de subplot de las diferentes imagenes de MyAdaptMedian
# figs, axs = plt.subplots(2, 4, figsize=(30, 8))
# figs.tight_layout(pad=3.0)

# axs[0,0].imshow(noisy_1, cmap='gray')
# axs[0,0].set_title('noisy_1 original')
# axs[0,0].axis('off')

# axs[0,1].imshow(filt_n_1_3x7, cmap='gray')
# axs[0,1].set_title('noisy_1 con filtro 3_max_7')
# axs[0,1].axis('off')

# axs[0,2].imshow(filt_n_1_5x9, cmap='gray')
# axs[0,2].set_title('noisy_1 con filtro 5_max_9')
# axs[0,2].axis('off')

# axs[0,3].imshow(filt_n_1_5x11, cmap='gray')
# axs[0,3].set_title('noisy_1 con filtro 5_max_11')
# axs[0,3].axis('off')

# axs[1,0].imshow(noisy_2, cmap='gray')
# axs[1,0].set_title('noisy_2 original')
# axs[1,0].axis('off')

# axs[1,1].imshow(filt_n_2_3x7, cmap='gray')
# axs[1,1].set_title('noisy_2 con filtro 3_max_7')
# axs[1,1].axis('off')

# axs[1,2].imshow(filt_n_2_5x9, cmap='gray')
# axs[1,2].set_title('noisy_2 con filtro 5_max_9')
# axs[1,2].axis('off')

# axs[1,3].imshow(filt_n_2_5x11, cmap='gray')
# axs[1,3].set_title('noisy_2 con filtro 5_max_11')
# axs[1,3].axis('off')

# plt.show()


# # inicio de subplot para las imagenes filtradas por método de Gauss
# figs, axs = plt.subplots(2, 3, figsize=(30, 8))
# figs.tight_layout(pad=3.0)

# axs[0,0].imshow(noisy_1, cmap='gray')
# axs[0,0].set_title('noisy_1 original')
# axs[0,0].axis('off')

# axs[0,1].imshow(filt_n1_g1, cmap='gray')
# axs[0,1].set_title('gauss noisy_1 t3_s11')
# axs[0,1].axis('off')

# axs[0,2].imshow(filt_n1_g2, cmap='gray')
# axs[0,2].set_title('gauss noisy_1 t7_s2')
# axs[0,2].axis('off')

# axs[1,0].imshow(noisy_2, cmap='gray')
# axs[1,0].set_title('noisy_2 original')
# axs[1,0].axis('off')

# axs[1,1].imshow(filt_n2_g1, cmap='gray')
# axs[1,1].set_title('gauss noisy_2 t3_s11')
# axs[1,1].axis('off')

# axs[1,2].imshow(filt_n2_g2, cmap='gray')
# axs[1,2].set_title('gauss noisy_2 t7_s2')
# axs[1,2].axis('off')

# plt.show()

# # REFERENCIAS

# # Salt and pepper noise -> https://www.sciencedirect.com/topics/engineering/pepper-noise
# # Gaussian noise -> https://www.sciencedirect.com/topics/computer-science/gaussian-noise
# # Uniform noise -> https://aishack.in/tutorials/generating-uniform-noise/







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


# creo las matrices de cada filtro (3.a,b,c,d)
one = np.ones((3,3))
tf = 1/9 * one
hf = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
vf = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
new_filter1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
new_filter2 = np.array([[7, 14, 7], [1, 1, 1], [-7, -14, -7]])

#Creo la funcion de correlacion
def MyCorrelation_201730655_201713127(image, kernel, boundary_condition):
    image_flat= image.flatten()
    kernel_flat = kernel.flatten()
    large_kernel = []
    if boundary_condition == 'fill':
        correlated_image = np.zeros((int(image.shape[0]+(((kernel.shape[0]+1)/2))),int(image.shape[1]+(((kernel.shape[1]+1)/2)))))
        for i in range(1,int(correlated_image.shape[0]-1)):
            for j in range(1,int(correlated_image.shape[1]-1)):
               correlated_image[i][j] = image[i-1][j-1]
        
        m_original = []
        operaciones = []
        for j in range(0,int(kernel.shape[0])):
                m_original.append([0,0,0])

        for i in range(0,int(image.shape[0])):
            for j in range(0,int(image.shape[1])):
                            m_original = [[correlated_image[i][j],correlated_image[i][j+1],correlated_image[i][j+2]],[correlated_image[i+1][j],
                            correlated_image[i+1][j+1],correlated_image[i+1][j+2]],[correlated_image[i+2][j],correlated_image[i+2][j+1],correlated_image[i+2][j+2]]]
                            operaciones.append(m_original)
        
        operaciones = np.array(operaciones)
        mult = operaciones
        for i in range(0,len(image_flat)):
            large_kernel.append(kernel)

        for i in range(0,len(image_flat)):
            for j in range(0,len(kernel)):
                for k in range(0,len(kernel)):
                    mult[i][j][k] =  large_kernel[i][j][k]*operaciones[i][j][k]
        suma = []
        for i in range(0,len(image_flat)):
            suma.append(np.sum(mult[i].flatten()))

                   
         

    return np.reshape(suma,(int(image.shape[0]),int(image.shape[1])))


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


#PARTE 2 
imagen1_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[0].flatten())[0],(imagenes[0].shape[0],imagenes[0].shape[1]))
imagen2_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[1].flatten())[0],(imagenes[1].shape[0],imagenes[1].shape[1]))
imagen1_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes[0].flatten(),hist_img1_target),(imagenes[0].shape[0],imagenes[0].shape[1]))
imagen2_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes[1].flatten(),hist_img1_target),(imagenes[1].shape[0],imagenes[1].shape[1]))
imagen1 = imagenes[0]
imagen2 = imagenes[1]

imagen1_cor_eqf1 = MyCorrelation_201730655_201713127(imagenes[0],new_filter1,'fill')
imagen2_cor_eqf1 = MyCorrelation_201730655_201713127(imagenes[0],new_filter1,'fill')
imagen1_cor_espf1 = MyCorrelation_201730655_201713127(imagen1_esp,new_filter1,'fill')
imagen2_cor_espf1 = MyCorrelation_201730655_201713127(imagen2_esp,new_filter1,'fill')
imagen1_corrf1 = MyCorrelation_201730655_201713127(imagen1,new_filter1,'fill')
imagen2_corrf1 = MyCorrelation_201730655_201713127(imagen2,new_filter1,'fill')

imagen1_cor_eqf2 = MyCorrelation_201730655_201713127(imagenes[0],new_filter2,'fill')
imagen2_cor_eqf2 = MyCorrelation_201730655_201713127(imagenes[0],new_filter2,'fill')
imagen1_cor_espf2 = MyCorrelation_201730655_201713127(imagen1_esp,new_filter2,'fill')
imagen2_cor_espf2 = MyCorrelation_201730655_201713127(imagen2_esp,new_filter2,'fill')
imagen1_corrf2 = MyCorrelation_201730655_201713127(imagen1,new_filter2,'fill')
imagen2_corrf2 = MyCorrelation_201730655_201713127(imagen2,new_filter2,'fill')

#Cargo imageenes d la carpeta Test
df_Test = glob.glob(os.path.join('Test','*.png'))
print(df_Test)

#Las convierto a grises y cambio su tipo de dato a uint8
imagenes_test = []
for i in range(0,len(df_Test)):
    imagenes_test.append(rgb2gray(io.imread(df_Test[i])))

for i in range(0,len(imagenes_test)):
    imagenes_test[i] = 255*imagenes_test[i]
    imagenes_test[i] = imagenes_test[i].astype(np.uint8) 

#Aplico pre proecesamientos de forma aleatoria a cada imagen de la carpeta test
test1_eq = np.reshape(MyHistEq_201730655_201713127(imagenes_test[0].flatten())[0],(imagenes_test[0].shape[0],imagenes_test[0].shape[1]))
test2_eq = np.reshape(MyHistEq_201730655_201713127(imagenes_test[1].flatten())[0],(imagenes_test[1].shape[0],imagenes_test[1].shape[1]))
test3_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes_test[2].flatten(),hist_img1_target),(imagenes_test[2].shape[0],imagenes_test[2].shape[1]))
test4 = imagenes_test[3]
test5_eq = np.reshape(MyHistEq_201730655_201713127(imagenes_test[4].flatten())[0],(imagenes_test[4].shape[0],imagenes_test[4].shape[1]))
test6_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes_test[5].flatten(),hist_img1_target),(imagenes_test[5].shape[0],imagenes_test[5].shape[1]))
test7 = imagenes_test[6]
test8 = imagenes_test[7]
test9_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes_test[8].flatten(),hist_img1_target),(imagenes_test[8].shape[0],imagenes_test[8].shape[1]))
test10_esp = np.reshape(MyHistEsp_201730655_201713127(imagenes_test[9].flatten(),hist_img1_target),(imagenes_test[9].shape[0],imagenes_test[9].shape[1]))

#Aplico la funcion de correlación con los kernels definidos, a las imagenes de la carpeta Test
corr_test1 = MyCorrelation_201730655_201713127(test1_eq,new_filter1,'fill')
corr_test2 = MyCorrelation_201730655_201713127(test2_eq,new_filter2,'fill')
corr_test3 = MyCorrelation_201730655_201713127(test3_esp,new_filter1,'fill')
corr_test4 = MyCorrelation_201730655_201713127(test4,new_filter2,'fill')
corr_test5 = MyCorrelation_201730655_201713127(test5_eq,new_filter1,'fill')
corr_test6 = MyCorrelation_201730655_201713127(test6_esp,new_filter2,'fill')
corr_test7 = MyCorrelation_201730655_201713127(test7,new_filter1,'fill')
corr_test8 = MyCorrelation_201730655_201713127(test8,new_filter2,'fill')
corr_test9 = MyCorrelation_201730655_201713127(test9_esp,new_filter2,'fill')
corr_test10 = MyCorrelation_201730655_201713127(test10_esp,new_filter1,'fill')

#Grafico
input('press enter to continue...')
fig, ax = plt.subplots(5,2)
ax[0][0].imshow(corr_test1,cmap='gray')
ax[0][0].axis('off')
ax[0][0].set_title('1')
ax[1][0].imshow(corr_test2,cmap='gray')
ax[1][0].axis('off')
ax[1][0].set_title('10')
ax[2][0].imshow(corr_test3,cmap='gray')
ax[2][0].axis('off')
ax[2][0].set_title('2')
ax[3][0].imshow(corr_test4,cmap='gray')
ax[3][0].axis('off')
ax[3][0].set_title('3')
ax[4][0].imshow(corr_test5,cmap='gray')
ax[4][0].axis('off')
ax[4][0].set_title('4')
ax[0][1].imshow(corr_test6,cmap='gray')
ax[0][1].axis('off')
ax[0][1].set_title('5')
ax[1][1].imshow(corr_test7,cmap='gray')
ax[1][1].axis('off')
ax[1][1].set_title('6')
ax[2][1].imshow(corr_test8,cmap='gray')
ax[2][1].axis('off')
ax[2][1].set_title('7')
ax[3][1].imshow(corr_test9,cmap='gray')
ax[3][1].axis('off')
ax[3][1].set_title('8')
ax[4][1].imshow(corr_test10,cmap='gray')
ax[4][1].axis('off')
ax[4][1].set_title('9')

plt.show()

#Métrica
#hallo los maximos de las imagenes de salida 
medias_test = [np.amax(corr_test1.flatten()),np.amax(corr_test2.flatten()),np.amax(corr_test3.flatten()),np.amax(corr_test4.flatten()),np.amax(corr_test5.flatten()),
np.amax(corr_test6.flatten()),np.amax(corr_test7.flatten()),np.amax(corr_test8.flatten()),np.amax(corr_test9.flatten()),np.amax(corr_test10.flatten())]

#hallo los maximos de los templates
media_moldes = [np.amax(imagen1_cor_eqf1.flatten()),np.amax(imagen2_cor_eqf1.flatten()),np.amax(imagen1_cor_espf1.flatten()),np.amax(imagen2_cor_espf1.flatten()),np.amax(imagen1_corrf1.flatten()),
np.amax(imagen2_corrf1.flatten()),np.amax(imagen1_cor_eqf2.flatten()),np.amax(imagen2_cor_eqf2.flatten()),np.amax(imagen1_cor_espf2.flatten()),np.amax(imagen2_cor_espf2.flatten()),
np.amax(imagen1_corrf2.flatten()),np.amax(imagen2_corrf2.flatten())]

#divido los maximos de los templates sobre los maximos de las imagenes de salida, como corresponda.
metrica = [media_moldes[0]/medias_test[0], media_moldes[7]/medias_test[1], media_moldes[3]/medias_test[2], media_moldes[4]/medias_test[3], media_moldes[0]/medias_test[4]
, media_moldes[8]/medias_test[5], media_moldes[5]/medias_test[6], media_moldes[10]/medias_test[7], media_moldes[9]/medias_test[8], media_moldes[3]/medias_test[9]]

print(metrica)