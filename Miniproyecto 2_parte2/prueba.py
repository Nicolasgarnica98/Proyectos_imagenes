import os
import glob
import requests
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy as sp
from skimage.color import rgb2hsv
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.filters.rank import percentile
from scipy.io import loadmat
from sklearn.metrics import jaccard_score as jaccard
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate2d



df_Train = glob.glob(os.path.join('Train','*.png'))
print(df_Train)

img_train = []
for i in range(0,len(df_Train)):
    img_train.append(io.imread(df_Train[i]))

#Funcion de concatenacion de histogramas
def MyColorHist_Código1_Código2(image, espacio):

    comparacion = np.arange(256)
    conc_hist = []
    image_actual = image

    #Condicion para el espacio RGB
    if espacio == 'RGB':
        #Conversion de la imagen al espacio de color
        image_actual = image_actual
        #obtencion de los arrays de las 3 dimensiones de la imagen
        hist1 = cv2.calcHist([image_actual],[0],None,[765],[float(np.min(image_actual[:,:,0])),float(np.max(image_actual[:,:,0]))])
        #sumar los desfaces para el histograma concatenado
        hist2 = cv2.calcHist([image_actual],[1],None,[765],[float(np.min(image_actual[:,:,1])),float(np.max(image_actual[:,:,1]))])
        hist3 = cv2.calcHist([image_actual],[2],None,[765],[float(np.min(image_actual[:,:,2])),float(np.max(image_actual[:,:,2]))])
        #union de los arrays desfasados
        #Calculo del histograma
        conc_hist = hist1+hist2+hist3
        #Normalizacion
        conc_hist = cv2.normalize(conc_hist,conc_hist,0,1,cv2.NORM_MINMAX)        

    #Condicion para el espacio HSV
    elif espacio == 'HSV':
        #Conversion de la imagen al espacio de color
        image_actual = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([image_actual],[0],None,[765],[float(np.min(image_actual[:,:,0])),float(np.max(image_actual[:,:,0]))])
        #sumar los desfaces para el histograma concatenado
        hist2 = cv2.calcHist([image_actual],[1],None,[765],[float(np.min(image_actual[:,:,1])),float(np.max(image_actual[:,:,1]))])
        hist3 = cv2.calcHist([image_actual],[2],None,[765],[float(np.min(image_actual[:,:,2])),float(np.max(image_actual[:,:,2]))])
        #union de los arrays desfasados
        #Calculo del histograma
        conc_hist = hist1+hist2+hist3
        #Normalizacion
        conc_hist = cv2.normalize(conc_hist,conc_hist,0,1,cv2.NORM_MINMAX)   

    #Condicion para el espacio Lab
    elif espacio == 'Lab':
        #Conversion de la imagen al espacio de color
        image_actual = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
        hist1 = cv2.calcHist([image_actual],[0],None,[765],[float(np.min(image_actual[:,:,0])),float(np.max(image_actual[:,:,0]))])
        #sumar los desfaces para el histograma concatenado
        hist2 = cv2.calcHist([image_actual],[1],None,[765],[float(np.min(image_actual[:,:,1])),float(np.max(image_actual[:,:,1]))])
        hist3 = cv2.calcHist([image_actual],[2],None,[765],[float(np.min(image_actual[:,:,2])),float(np.max(image_actual[:,:,2]))])
        #union de los arrays desfasados
        #Calculo del histograma
        conc_hist = hist1+hist2+hist3
        #Normalizacion
        conc_hist = cv2.normalize(conc_hist,conc_hist,0,1,cv2.NORM_MINMAX)   



    return conc_hist

plt.plot(MyColorHist_Código1_Código2(img_train[1],'HSV'))
plt.plot(MyColorHist_Código1_Código2(img_train[0],'HSV'))
plt.show()

# #Funcion de concatenacion de histogramas
# def MyColorHist_Código1_Código2(image, espacio):

#     comparacion = np.arange(256)
#     conc_hist = []
#     image_actual = image

#     #Condicion para el espacio RGB
#     if espacio == 'RGB':
#         #Conversion de la imagen al espacio de color
#         image_actual = image_actual
#         #obtencion de los arrays de las 3 dimensiones de la imagen
#         hist1 = image_actual[:,:,0].flatten()
#         #sumar los desfaces para el histograma concatenado
#         hist2 = image_actual[:,:,1].flatten() + 256
#         hist3 = image_actual[:,:,2].flatten() + 511
#         #union de los arrays desfasados
#         conc_hist=np.concatenate((hist1,hist2,hist3))
#         #Calculo del histograma
#         conc_hist = conc_hist.astype('float32')
#         conc_hist = cv2.calcHist([conc_hist],[0],None,[765],[float(np.min(conc_hist)),float(np.max(conc_hist))],conc_hist)
#         #Normalizacion
#         conc_hist = cv2.normalize(conc_hist,conc_hist,0,1,cv2.NORM_MINMAX)        

#     #Condicion para el espacio HSV
#     elif espacio == 'HSV':
#         #Conversion de la imagen al espacio de color
#         image_actual = rgb2hsv(image_actual)
#         image_actual = image_actual*255
#         #obtencion de los arrays de las 3 dimensiones de la imagen
#         hist1 = image_actual[:,:,0].flatten()
#         #sumar los desfaces para el histograma concatenado
#         hist2 = image_actual[:,:,1].flatten()+1+(1/255)
#         hist2 = hist2*255
#         hist3 = image_actual[:,:,2].flatten()+2+(1/255)
#         hist3 = hist3*255
#         conc_hist=np.concatenate((hist1,hist2,hist3))
#         #Calculo del histograma
#         conc_hist = conc_hist.astype('float32')
#         conc_hist = cv2.calcHist([conc_hist],[0],None,[765],[float(np.min(conc_hist)),float(np.max(conc_hist))],conc_hist)
#         #Normalizacion
#         conc_hist = cv2.normalize(conc_hist,conc_hist,0,1,cv2.NORM_MINMAX)

#     #Condicion para el espacio Lab
#     elif espacio == 'Lab':
#         #Conversion de la imagen al espacio de color
#         image_actual = rgb2lab(image_actual)
#         #sumar los desfaces para el histograma concatenado, dado que pueden existir valores negativos, estos son sumados como desface tambien
#         hist1 = image_actual[:,:,0].flatten()
#         hist2 = image_actual[:,:,1].flatten()
#         hist2 = hist2 + np.min(hist2)+ np.max(hist1)+1
#         hist3 = image_actual[:,:,2].flatten()
#         hist3 = hist3 + np.min(hist3)+ np.max(hist2)+1
#         conc_hist=np.concatenate((hist1,hist2,hist3))
#         #Calculo del histograma
#         conc_hist = conc_hist.astype('float32')
#         conc_hist = cv2.calcHist([conc_hist],[0],None,[765],[float(np.min(conc_hist)),float(np.max(conc_hist))],conc_hist)
#         #Normalizacion
#         conc_hist = cv2.normalize(conc_hist,conc_hist,0,1,cv2.NORM_MINMAX)



#     return conc_hist