import os
import glob
import requests
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.io import loadmat
from sklearn.metrics import jaccard_score
import nibabel as nb

# #Ruta de la imagen
# img = 'https://cdn.shopify.com/s/files/1/1280/3657/products/GM-BA-SET01_1_e5745bf6-253a-4a83-a8d1-5d55ee691e5d_1024x1024.jpg?v=1548113518'

# #Obtener imagen de la ruta URL
# r = requests.get(img)
# with open('monedas.jpg', 'wb') as f:
#           f.write(r.content)

# #Conversion de la imagen a grises
# imagen_monedas = io.imread(os.path.join('monedas.jpg'))
# imagen_monedas_grises = rgb2gray(imagen_monedas)

# #Creacion de la mascara otsu
# otsu = threshold_otsu(imagen_monedas_grises)
# mascara_otsu = imagen_monedas_grises < otsu

# #Creacion de la mascara con umbral en el percentil que acumula el 80% de intensidad.
# percentil_80 = np.percentile(imagen_monedas_grises.flatten(),80)
# mascara_perc80 = imagen_monedas_grises < percentil_80

# #Creacion de la mascara con umbral en el percentil que acumula el 30% de intensidad.
# percentil_30 = np.percentile(imagen_monedas_grises.flatten(),30)
# mascara_umb = imagen_monedas_grises < percentil_30

# # #Creacion de la mascara con umbral superior en el valor de otsu y el inferior en el percentil que acumula el 30%.
# percentil_30 = np.percentile(imagen_monedas_grises.flatten(),30)
# umbrales = [otsu,percentil_30]
# mascara_umbrales = np.digitize(imagen_monedas_grises, bins=umbrales)

# dt = loadmat(os.path.join('groundtruth.mat'))
# imag_anotacion = rgb2gray(dt['gt'])

# def jaccard(anotacion, mascara):
#     interseccion = 0
#     union = len(mascara.flatten()) + len(anotacion.flatten())
#     for i in range(0,len(anotacion)):
#         for j in range(0, len(anotacion)):
#             if anotacion[i][j] == mascara[i][j]:
#                 interseccion = interseccion + 1
    
#     union = len(mascara.flatten()) + len(anotacion.flatten()) - interseccion
    
#     return interseccion/union

# jaccard_otsu = jaccard(imag_anotacion,mascara_otsu)
# jaccard_p80 = jaccard(imag_anotacion,mascara_perc80)
# jaccard_umb = jaccard(imag_anotacion,mascara_umb)
# jaccard_umbrales = jaccard(imag_anotacion,mascara_umbrales)

# print(jaccard_otsu,jaccard_p80,jaccard_umb,jaccard_umbrales)


#Segunda parte biomédica
df_data = glob.glob(os.path.join('Liver_Slices\Data','*.nii.gz'))
df_anotaciones = glob.glob(os.path.join('Liver_Slices\GroundTruth','*.nii.gz'))

print(df_anotaciones[0])

