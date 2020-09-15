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

#Ruta de la imagen
img = 'https://cdn.shopify.com/s/files/1/1280/3657/products/GM-BA-SET01_1_e5745bf6-253a-4a83-a8d1-5d55ee691e5d_1024x1024.jpg?v=1548113518'

#Obtener imagen de la ruta URL
r = requests.get(img)
with open('monedas.jpg', 'wb') as f:
          f.write(r.content)

#Conversion de la imagen a grises
imagen_monedas = io.imread(os.path.join('monedas.jpg'))
imagen_monedas_grises = rgb2gray(imagen_monedas)

#Creacion de la mascara otsu
otsu = threshold_otsu(imagen_monedas_grises)
mascara_otsu = imagen_monedas_grises < otsu

#Creacion de la mascara con umbral en el percentil que acumula el 80% de intensidad.
percentil_80 = np.percentile(imagen_monedas_grises.flatten(),80)
mascara_perc80 = imagen_monedas_grises < percentil_80

#Creacion de la mascara con umbral en el percentil que acumula el 30% de intensidad.
percentil_30 = np.percentile(imagen_monedas_grises.flatten(),30)
mascara_umb = imagen_monedas_grises < percentil_30

# #Creacion de la mascara con umbral superior en el valor de otsu y el inferior en el percentil que acumula el 30%.
percentil_30 = np.percentile(imagen_monedas_grises.flatten(),30)
umbrales = [percentil_30,otsu]
mascara_umbrales = np.digitize(imagen_monedas_grises, bins=umbrales)

#Se cargan los datos ed la anotacion para las monedas
dt = loadmat(os.path.join('groundtruth.mat'))
imag_anotacion = rgb2gray(dt['gt'])

#Se crea la funcion de Jaccard que cuenta la cantidad de datos en interseccion y la divide sobre la cantidad de elementos de la unión
#de ambos grupos (Anotacion y mascara)
def jaccard(anotacion, mascara):
    interseccion = np.logical_and(anotacion,mascara)
    union = np.logical_or(anotacion,mascara)
    return interseccion.sum()/union.sum()

#Aplico la funcion para comparar las mascaras creadas con las anotaciones
jaccard_otsu = jaccard(imag_anotacion,mascara_otsu)
jaccard_p80 = jaccard(imag_anotacion,mascara_perc80)
jaccard_umb = jaccard(imag_anotacion,mascara_umb)
jaccard_umbrales = jaccard(imag_anotacion,mascara_umbrales)

#Indices de Jaccard para cada segmentación
print('Jaccar Otsu: ',jaccard_otsu)
print('Jaccard percetil 80: ',jaccard_p80)
print('Jaccard umbral aleatorio: ',jaccard_umb)
print('Jaccard umbrales: ',jaccard_umbrales)







#SEGUNDA PARTE PROBLEMA BIOMÉDICO
input("Presionar enter para continuar con el problema biomédico")
df_data = glob.glob(os.path.join('Liver_Slices\Data','*.nii.gz'))
df_anotaciones = glob.glob(os.path.join('Liver_Slices\GroundTruth','*.nii.gz'))

#Adquirir y organizar datos de los volumenes de la entrega pasada y los volumenes nuevos de las anotaciones
#Se utilizan los mismos metodos de la entrega pasada.
def organizar_paciente(df):
  aux1 = []
  aux2 = []
  aux3 = []  
  for i in range(0,len(df)):
     x = nb.load(df[i])
     if str(x.header['intent_name']) == "b'Patient 1'":
         aux1.append(x)
     elif str(x.header['intent_name']) == "b'Patient 25'":
         aux2.append(x)
     else:
         aux3.append(x)
  
  return aux1,aux2,aux3

pre_vol1 = organizar_paciente(df_data)[0]
pre_vol2 = organizar_paciente(df_data)[1]
pre_vol3 = organizar_paciente(df_data)[2]
pre_vol1_anotaciones = organizar_paciente(df_anotaciones)[0]
pre_vol2_anotaciones = organizar_paciente(df_anotaciones)[1]
pre_vol3_anotaciones = organizar_paciente(df_anotaciones)[2]

def organizar_volumenes(aux):
 saux = []
 for i in range(0,len(aux)):
     saux.append(0)

 for i in range(0,len(saux)):
     num_slide = int(str(aux[i].header['descrip'])[8:-1])
     saux[int(str(aux[i].header['descrip'])[8:-1])-1] = aux[i]

 return saux

vol1 = organizar_volumenes(pre_vol1)
vol2 = organizar_volumenes(pre_vol2)
vol3 = organizar_volumenes(pre_vol3)
vol1_anotaciones = organizar_volumenes(pre_vol1_anotaciones)
vol2_anotaciones = organizar_volumenes(pre_vol2_anotaciones)
vol3_anotaciones = organizar_volumenes(pre_vol3_anotaciones)

#Paciente 1, se escogieron los cortes 32, 54 y 21
#Se carga el corte seleccionado
img1_p1 = vol1[32].get_fdata()
#Se crean las mascaras para dicho corte
mascara_img1_p1_otsu = img1_p1 > threshold_otsu(img1_p1)
mascara_img1_p1_p80 = img1_p1 > np.percentile(img1_p1.flatten(),80)
mascara_img1_p1_umb = img1_p1 > np.percentile(img1_p1.flatten(),30)
mascara_img1_p1_umbrales = np.digitize(img1_p1, bins=umbrales)
#Se aplican las mascaras para obtener las segmentaciones
seg_img1_p1_otsu = img1_p1*mascara_img1_p1_otsu
seg_img1_p1_p80 = img1_p1*mascara_img1_p1_p80
seg_img1_p1_umb = img1_p1*mascara_img1_p1_umb
seg_img1_p1_umbrales = img1_p1*mascara_img1_p1_umbrales
#Se obtiene la anotacion correspondiente al corte seleccionado
anot1_p1 = rgb2gray(vol1_anotaciones[32].get_fdata())
#Se utiliza la funcion de Jaccard creada anteriormente para obtener los valores de comparacion entre cada mascara y la anotacion 
jacc_img1_p1_otsu = jaccard(anot1_p1,mascara_img1_p1_otsu)
jacc_img1_p1_p80 = jaccard(anot1_p1,mascara_img1_p1_p80)
jacc_img1_p1_umb = jaccard(anot1_p1,mascara_img1_p1_umb)
jacc_img1_p1_umbrales = jaccard(anot1_p1,mascara_img1_p1_umbrales)
#Se imprimen los valores del indice de jaccard
print(jacc_img1_p1_otsu,jacc_img1_p1_p80,jacc_img1_p1_umb, jacc_img1_p1_umbrales)
#Se repite este proceso para las 3 imagenes de los 3 pacientes.

img2_p1 = vol1[54].get_fdata()
mascara_img2_p1_otsu = img2_p1 > threshold_otsu(img2_p1)
mascara_img2_p1_p80 = img2_p1 > np.percentile(img2_p1.flatten(),80)
mascara_img2_p1_umb = img2_p1 > np.percentile(img2_p1.flatten(),30)
mascara_img2_p1_umbrales = np.digitize(img2_p1, bins=umbrales)
seg_img2_p1_otsu = img2_p1*mascara_img2_p1_otsu
seg_img2_p1_p80 = img2_p1*mascara_img2_p1_p80
seg_img2_p1_umb = img2_p1*mascara_img2_p1_umb
seg_img2_p1_umbrales = img2_p1*mascara_img2_p1_umbrales
anot2_p1 = rgb2gray(vol1_anotaciones[54].get_fdata())
jacc_img2_p1_otsu = jaccard(anot2_p1,mascara_img2_p1_otsu)
jacc_img2_p1_p80 = jaccard(anot2_p1,mascara_img2_p1_p80)
jacc_img2_p1_umb = jaccard(anot2_p1,mascara_img2_p1_umb)
jacc_img2_p1_umbrales = jaccard(anot2_p1,mascara_img2_p1_umbrales)
print(jacc_img2_p1_otsu,jacc_img2_p1_p80,jacc_img2_p1_umb, jacc_img2_p1_umbrales)


img3_p1 = vol1[21].get_fdata()
mascara_img3_p1_otsu = img3_p1 > threshold_otsu(img3_p1)
mascara_img3_p1_p80 = img3_p1 > np.percentile(img3_p1.flatten(),80)
mascara_img3_p1_umb = img3_p1 > np.percentile(img3_p1.flatten(),30)
mascara_img3_p1_umbrales = np.digitize(img3_p1, bins=umbrales)
seg_img3_p1_otsu = img3_p1*mascara_img3_p1_otsu
seg_img3_p1_p80 = img3_p1*mascara_img3_p1_p80
seg_img3_p1_umb = img3_p1*mascara_img3_p1_umb
seg_img3_p1_umbrales = img3_p1*mascara_img3_p1_umbrales
anot3_p1 = rgb2gray(vol1_anotaciones[21].get_fdata())
jacc_img3_p1_otsu = jaccard(anot3_p1,mascara_img3_p1_otsu)
jacc_img3_p1_p80 = jaccard(anot3_p1,mascara_img3_p1_p80)
jacc_img3_p1_umb = jaccard(anot3_p1,mascara_img3_p1_umb)
jacc_img3_p1_umbrales = jaccard(anot3_p1,mascara_img3_p1_umbrales)
print(jacc_img3_p1_otsu,jacc_img3_p1_p80,jacc_img3_p1_umb, jacc_img3_p1_umbrales)


#Paciente 25, se escogieron los cortes 32, 54 y 21
img1_p2 = vol2[32].get_fdata()
mascara_img1_p2_otsu = img1_p2 > threshold_otsu(img1_p2)
mascara_img1_p2_p80 = img1_p2 > np.percentile(img1_p2.flatten(),80)
mascara_img1_p2_umb = img1_p2 > np.percentile(img1_p2.flatten(),30)
mascara_img1_p2_umbrales = np.digitize(img1_p2, bins=umbrales)
seg_img1_p2_otsu = img1_p2*mascara_img1_p2_otsu
seg_img1_p2_p80 = img1_p2*mascara_img1_p2_p80
seg_img1_p2_umb = img1_p2*mascara_img1_p2_umb
seg_img1_p2_umbrales = img1_p2*mascara_img1_p2_umbrales
anot1_p2 = rgb2gray(vol2_anotaciones[32].get_fdata())
jacc_img1_p2_otsu = jaccard(anot1_p2,mascara_img1_p2_otsu)
jacc_img1_p2_p80 = jaccard(anot1_p2,mascara_img1_p2_p80)
jacc_img1_p2_umb = jaccard(anot1_p2,mascara_img1_p2_umb)
jacc_img1_p2_umbrales = jaccard(anot1_p2,mascara_img1_p2_umbrales)
print(jacc_img1_p2_otsu,jacc_img1_p2_p80,jacc_img1_p2_umb, jacc_img1_p2_umbrales)


img2_p2 = vol2[54].get_fdata()
mascara_img2_p2_otsu = img2_p2 > threshold_otsu(img2_p2)
mascara_img2_p2_p80 = img2_p2 > np.percentile(img2_p2.flatten(),80)
mascara_img2_p2_umb = img2_p2 > np.percentile(img2_p2.flatten(),30)
mascara_img2_p2_umbrales = np.digitize(img2_p2, bins=umbrales)
seg_img2_p2_otsu = img2_p2*mascara_img2_p2_otsu
seg_img2_p2_p80 = img2_p2*mascara_img2_p2_p80
seg_img2_p2_umb = img2_p2*mascara_img2_p2_umb
seg_img2_p2_umbrales = img2_p2*mascara_img2_p2_umbrales
anot2_p2 = rgb2gray(vol2_anotaciones[54].get_fdata())
jacc_img2_p2_otsu = jaccard(anot2_p2,mascara_img2_p2_otsu)
jacc_img2_p2_p80 = jaccard(anot2_p2,mascara_img2_p2_p80)
jacc_img2_p2_umb = jaccard(anot2_p2,mascara_img2_p2_umb)
jacc_img2_p2_umbrales = jaccard(anot2_p2,mascara_img2_p2_umbrales)
print(jacc_img2_p2_otsu,jacc_img2_p2_p80,jacc_img2_p2_umb, jacc_img2_p2_umbrales)


img3_p2 = vol2[21].get_fdata()
mascara_img3_p2_otsu = img3_p2 > threshold_otsu(img3_p2)
mascara_img3_p2_p80 = img3_p2 > np.percentile(img3_p2.flatten(),80)
mascara_img3_p2_umb = img3_p2 > np.percentile(img3_p2.flatten(),30)
mascara_img3_p2_umbrales = np.digitize(img3_p2, bins=umbrales)
seg_img3_p2_otsu = img3_p2*mascara_img3_p2_otsu
seg_img3_p2_p80 = img3_p2*mascara_img3_p2_p80
seg_img3_p2_umb = img3_p2*mascara_img3_p2_umb
seg_img3_p2_umbrales = img3_p2*mascara_img3_p2_umbrales
anot3_p2 = rgb2gray(vol2_anotaciones[21].get_fdata())
jacc_img3_p2_otsu = jaccard(anot3_p2,mascara_img3_p2_otsu)
jacc_img3_p2_p80 = jaccard(anot3_p2,mascara_img3_p2_p80)
jacc_img3_p2_umb = jaccard(anot3_p2,mascara_img3_p2_umb)
jacc_img3_p2_umbrales = jaccard(anot3_p2,mascara_img3_p2_umbrales)
print(jacc_img3_p2_otsu,jacc_img3_p2_p80,jacc_img3_p2_umb, jacc_img3_p2_umbrales)


#Paciente 44, se escogieron los cortes 32, 54 y 21
img1_p3 = vol3[32].get_fdata()
mascara_img1_p3_otsu = img1_p3 > threshold_otsu(img1_p3)
mascara_img1_p3_p80 = img1_p3 > np.percentile(img1_p3.flatten(),80)
mascara_img1_p3_umb = img1_p3 > np.percentile(img1_p3.flatten(),30)
mascara_img1_p3_umbrales = np.digitize(img1_p3, bins=umbrales)
seg_img1_p3_otsu = img1_p3*mascara_img1_p3_otsu
seg_img1_p3_p80 = img1_p3*mascara_img1_p3_p80
seg_img1_p3_umb = img1_p3*mascara_img1_p3_umb
seg_img1_p3_umbrales = img1_p3*mascara_img1_p3_umbrales
anot1_p3 = rgb2gray(vol3_anotaciones[32].get_fdata())
jacc_img1_p3_otsu = jaccard(anot1_p3,mascara_img1_p3_otsu)
jacc_img1_p3_p80 = jaccard(anot1_p3,mascara_img1_p3_p80)
jacc_img1_p3_umb = jaccard(anot1_p3,mascara_img1_p3_umb)
jacc_img1_p3_umbrales = jaccard(anot1_p3,mascara_img1_p3_umbrales)
print(jacc_img1_p3_otsu,jacc_img1_p3_p80,jacc_img1_p3_umb, jacc_img1_p3_umbrales)


img2_p3 = vol3[54].get_fdata()
mascara_img2_p3_otsu = img2_p3 > threshold_otsu(img2_p3)
mascara_img2_p3_p80 = img2_p3 > np.percentile(img2_p3.flatten(),80)
mascara_img2_p3_umb = img2_p3 > np.percentile(img2_p3.flatten(),30)
mascara_img2_p3_umbrales = np.digitize(img2_p3, bins=umbrales)
seg_img2_p3_otsu = img2_p3*mascara_img2_p3_otsu
seg_img2_p3_p80 = img2_p3*mascara_img2_p3_p80
seg_img2_p3_umb = img2_p3*mascara_img2_p3_umb
seg_img2_p3_umbrales = img2_p3*mascara_img2_p3_umbrales
anot2_p3 = rgb2gray(vol3_anotaciones[54].get_fdata())
jacc_img2_p3_otsu = jaccard(anot2_p3,mascara_img2_p3_otsu)
jacc_img2_p3_p80 = jaccard(anot2_p3,mascara_img2_p3_p80)
jacc_img2_p3_umb = jaccard(anot2_p3,mascara_img2_p3_umb)
jacc_img2_p3_umbrales = jaccard(anot2_p3,mascara_img2_p3_umbrales)
print(jacc_img2_p3_otsu,jacc_img2_p3_p80,jacc_img2_p3_umb, jacc_img2_p3_umbrales)


img3_p3 = vol3[21].get_fdata()
mascara_img3_p3_otsu = img3_p3 > threshold_otsu(img3_p3)
mascara_img3_p3_p80 = img3_p3 > np.percentile(img3_p3.flatten(),80)
mascara_img3_p3_umb = img3_p3 > np.percentile(img3_p3.flatten(),30)
mascara_img3_p3_umbrales = np.digitize(img3_p3, bins=umbrales)
seg_img3_p3_otsu = img3_p3*mascara_img3_p3_otsu
seg_img3_p3_p80 = img3_p3*mascara_img3_p3_p80
seg_img3_p3_umb = img3_p3*mascara_img3_p3_umb
seg_img3_p3_umbrales = img3_p3*mascara_img3_p3_umbrales
anot3_p3 = rgb2gray(vol3_anotaciones[21].get_fdata())
jacc_img3_p3_otsu = jaccard(anot3_p3,mascara_img3_p3_otsu)
jacc_img3_p3_p80 = jaccard(anot3_p3,mascara_img3_p3_p80)
jacc_img3_p3_umb = jaccard(anot3_p3,mascara_img3_p3_umb)
jacc_img3_p3_umbrales = jaccard(anot3_p3,mascara_img3_p3_umbrales)
print(jacc_img3_p3_otsu,jacc_img3_p3_p80,jacc_img3_p3_umb, jacc_img3_p3_umbrales)


#Gráfico de las imagenes seleccionadas: corte 32 del paciente 1 y 44 con sus respectivas segmentaciones con los metodos
#propuestos.
input("Press Enter to continue...")
figx, ax1 = plt.subplots(2, 5, figsize=(20, 6))
figx.suptitle('Segmentacion del corte 32 de los pacientes 1 y 44')

ax1[0][0].set_title('Corte 32 P1')
ax1[0][0].imshow(img1_p1, cmap='gray')
ax1[0][0].axis('off')
ax1[0][1].set_title('Seg. Otsu P1')
ax1[0][1].imshow(seg_img1_p1_otsu, cmap = 'gray')
ax1[0][1].axis('off')
ax1[0][2].set_title('Seg. percentil 80 P1')
ax1[0][2].imshow(seg_img1_p1_p80, cmap='gray')
ax1[0][2].axis('off')
ax1[0][3].set_title('Seg. umbral arbitrario P1')
ax1[0][3].imshow(seg_img1_p1_umb, cmap = 'gray')
ax1[0][3].axis('off')
ax1[0][4].set_title('Seg. umbrales arbitrarios P1')
ax1[0][4].imshow(seg_img1_p1_umbrales, cmap='gray')
ax1[0][4].axis('off')
ax1[1][0].set_title('Corte 32 P44')
ax1[1][0].imshow(img1_p3, cmap = 'gray')
ax1[1][0].axis('off')
ax1[1][1].set_title('Seg. Otsu P44')
ax1[1][1].imshow(seg_img1_p3_otsu, cmap='gray')
ax1[1][1].axis('off')
ax1[1][2].set_title('Seg. percentil 80 P44')
ax1[1][2].imshow(seg_img1_p3_p80, cmap = 'gray')
ax1[1][2].axis('off')
ax1[1][3].set_title('Seg. umbral arbitrario P44')
ax1[1][3].imshow(seg_img1_p3_umb, cmap='gray')
ax1[1][3].axis('off')
ax1[1][4].set_title('Seg. umbrales arbitrarios P44')
ax1[1][4].imshow(seg_img1_p3_umbrales, cmap='gray')
ax1[1][4].axis('off')
plt.show()

