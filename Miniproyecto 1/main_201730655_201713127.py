import os
import glob
import requests
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.io import loadmat
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


#Grafico
input("Press Enter to continue...")
fig, a = plt.subplots(1, 3, figsize=(14, 6),gridspec_kw={'wspace':0.3})
a[0].set_title('Imagen a color')
a[0].imshow(imagen_monedas)
a[0].axis('off')
a[1].set_title('Imagen en grises')
a[1].imshow(imagen_monedas_grises, cmap ='gray')
a[1].axis('off')
a[2].set_title('Histograma imagen en grises')
a[2].hist(imagen_monedas_grises.flatten(), bins=255)
a[2].set(xlabel='Intensidad',ylabel='Cantidad de pixeles')
plt.show()

#Creacion de la mascara otsu
otsu = threshold_otsu(imagen_monedas_grises)
mascara_otsu = imagen_monedas_grises < otsu
#Segmentacion con la mascara Otsu
seg_Otsu = mascara_otsu*imagen_monedas_grises

#Creacion de la mascara con umbral en el percentil que acumula el 80% de intensidad.
percentil_80 = np.percentile(imagen_monedas_grises.flatten(),80)
mascara_perc80 = imagen_monedas_grises < percentil_80
seg_perc80 = mascara_perc80 * imagen_monedas_grises

#Creacion de la mascara con umbral en el percentil que acumula el 30% de intensidad.
percentil_30 = np.percentile(imagen_monedas_grises.flatten(),30)
mascara_umb = imagen_monedas_grises < percentil_30
seg_umb = mascara_umb * imagen_monedas_grises

# #Creacion de la mascara con umbral superior en el valor de otsu y el inferior en el percentil que acumula el 30%.
percentil_30 = np.percentile(imagen_monedas_grises.flatten(),30)
umbrales = [otsu,percentil_30]
mascara_umbrales = np.digitize(imagen_monedas_grises, bins=umbrales)
seg_umbrales = mascara_umbrales * imagen_monedas_grises


# #Gráfico
input("Press Enter to continue...")

fig1, a2 = plt.subplots(2, 4, figsize=(14, 6))

a2[0][0].set_title('Máscara Otsu')
a2[0][0].imshow(mascara_otsu, cmap='gray')
a2[0][0].axis('off')
a2[0][1].set_title('Máscara percentil 80')
a2[0][1].imshow(mascara_perc80,cmap='gray')
a2[0][1].axis('off')
a2[0][2].set_title('Máscara umbral arbitrario')
a2[0][2].imshow(mascara_umb,cmap='gray')
a2[0][2].axis('off')
a2[0][3].set_title('Máscara umbrales arbitrarios')
a2[0][3].imshow(mascara_umbrales,cmap='gray')
a2[0][3].axis('off')
a2[1][0].set_title('Segmentación Otsu')
a2[1][0].imshow(seg_Otsu,cmap = 'gray')
a2[1][0].axis('off')
a2[1][1].set_title('Segmentación umbral percentil 80')
a2[1][1].imshow(seg_perc80,cmap = 'gray')
a2[1][1].axis('off')
a2[1][2].set_title('Segmentación umbral aleatorio')
a2[1][2].imshow(seg_umb,cmap = 'gray')
a2[1][2].axis('off')
a2[1][3].set_title('Segmentación umbrales aleatorios')
a2[1][3].imshow(seg_umbrales,cmap = 'gray')
a2[1][3].axis('off')
plt.show()

# #Segmentación a color
seg_imagen_monedas_otsu = imagen_monedas
seg_imagen_monedas_p80 = imagen_monedas
seg_imagen_monedas_ua = imagen_monedas
seg_imagen_monedas_du = imagen_monedas

seg_imagen_monedas_otsu[:,:,0] = mascara_otsu * seg_imagen_monedas_otsu[:,:,0]
seg_imagen_monedas_otsu[:,:,1] = mascara_otsu * seg_imagen_monedas_otsu[:,:,1]
seg_imagen_monedas_otsu[:,:,2] = mascara_otsu * seg_imagen_monedas_otsu[:,:,2]


seg_imagen_monedas_p80[:,:,0] = mascara_perc80 * seg_imagen_monedas_p80[:,:,0]
seg_imagen_monedas_p80[:,:,1] = mascara_perc80 * seg_imagen_monedas_p80[:,:,1]
seg_imagen_monedas_p80[:,:,2] = mascara_perc80 * seg_imagen_monedas_p80[:,:,2]


seg_imagen_monedas_ua[:,:,0] = mascara_umb * seg_imagen_monedas_ua[:,:,0]
seg_imagen_monedas_ua[:,:,1] = mascara_umb * seg_imagen_monedas_ua[:,:,1]
seg_imagen_monedas_ua[:,:,2] = mascara_umb * seg_imagen_monedas_ua[:,:,2]


seg_imagen_monedas_du[:,:,0] = mascara_umbrales * seg_imagen_monedas_du[:,:,0]
seg_imagen_monedas_du[:,:,1] = mascara_umbrales * seg_imagen_monedas_du[:,:,1]
seg_imagen_monedas_du[:,:,2] = mascara_umbrales * seg_imagen_monedas_du[:,:,2]


#Gráfica de segmentación a color
input("Press Enter to continue...")
figx, ax1 = plt.subplots(1, 4, figsize=(20, 6))
ax1[0].set_title('Segmentación a color Otsu')
ax1[0].imshow(seg_imagen_monedas_otsu)
ax1[0].axis('off')
ax1[1].set_title('Segmentación a color percentil 80')
ax1[1].imshow(seg_imagen_monedas_p80)
ax1[1].axis('off')
ax1[2].set_title('Segmentación a color umbral arbitrario')
ax1[2].imshow(seg_imagen_monedas_ua)
ax1[2].axis('off')
ax1[3].set_title('Segmentación a color umbrales arbitrarios')
ax1[3].imshow(seg_imagen_monedas_du)
ax1[3].axis('off')
plt.show()


#Problema biomédico
#adquiero los datos en la variable df
df = glob.glob(os.path.join('Liver_Slices\Data','*.nii.gz'))
print(df[0])

#creo 3 arreglos vacios que me ayudaran en la organizacion de las imagenes
aux1 = []
aux2 = []
aux3 = []

#de los header de cada imagen, obtengo el nombre del paciente al que pertenece cada imagen y las organizo agregandolas a los arreglos auxiliares
for i in range(0,len(df)):
    x = nb.load(df[i])
    if str(x.header['intent_name']) == "b'Patient 1'":
        aux1.append(x)
    elif str(x.header['intent_name']) == "b'Patient 25'":
        aux2.append(x)
    else:
        aux3.append(x)

#creo una funcion que me permite organizar los arreglos. esta recibe como input un arrgelo aux donde ya se enecunetran calsificadas las imagenes por paciente, y se organizan de acuerdo
#con el numero del slide definido en el header de cada imagen. para esto convierto en integer el string donde se encuentra el numero del slide.
def organizar(aux):
 saux = []
 for i in range(0,len(aux)):
     saux.append(0)

 for i in range(0,len(saux)):
     num_slide = int(str(aux[i].header['descrip'])[8:-1])
     saux[int(str(aux[i].header['descrip'])[8:-1])-1] = aux[i]

 return saux

#creo 3 variables en las cuales organizo los arreglos con la funcion realizada
pre_vol1 = organizar(aux1)
pre_vol2 = organizar(aux2)
pre_vol3 = organizar(aux3)

#creo una variable vol de 3 dimensiones, que guardara las matrices numericas de cada una de las imágenes de cada paciente. la funcion recibe como input 
# los arreglos ordenados y separados, y devuelve un arreglo tridimensional con estos.

def organizar_volumenes(pre_vol):
    vol = np.zeros((len(pre_vol), 512, 512), dtype = np.single)
    for i in range(0,len(pre_vol)):
        vol[i,:,:] = pre_vol[i].get_fdata()
    return vol

#las variables de 3 dimensiones ya han sido creadas con ayuda de la funcion que creé en el paso anterior
vol1 = organizar_volumenes(pre_vol1)
vol2 = organizar_volumenes(pre_vol2)
vol3 = organizar_volumenes(pre_vol3)



#Grafico todos los cortes por paciente y por eje.

#Paciente 1
input("Press Enter to continue...")
plt.ion()
plt.show()
fig2 , x2 = plt.subplots(1,3)
fig2.suptitle('Paciente 1')
for idx, num in enumerate(np.linspace(0, 1, 200)):
    x2[0].imshow(vol1[idx], cmap='gray')
    x2[0].set_title('Eje transversal')
    x2[0].axis('off')
    x2[1].imshow(vol1[:,idx,:], cmap='gray')
    x2[1].set_title('Eje coronal')
    x2[1].axis('off')
    x2[2].imshow(vol1[:,:,idx], cmap='gray')
    x2[2].set_title('Eje axial')
    x2[2].axis('off')
    plt.pause(0.0001)
    if idx == 164:
        plt.clf()
        break
plt.close(fig=fig2)

#Paciente 25
input("Press Enter to continue...")
fig3 , x3 = plt.subplots(1,3)
fig3.suptitle('Paciente 25')
for idx, num in enumerate(np.linspace(0, 1, 200)):
    x3[0].imshow(vol2[idx], cmap='gray')
    x3[0].set_title('Eje transversal')
    x3[0].axis('off')
    x3[1].imshow(vol2[:,idx,:], cmap='gray')
    x3[1].set_title('Eje coronal')
    x3[1].axis('off')
    x3[2].imshow(vol2[:,:,idx], cmap='gray')
    x3[2].set_title('Eje axial')
    x3[2].axis('off')
    plt.pause(0.0001)
    if idx == 135:
        plt.clf()
        break
plt.close(fig=fig3)

#Paciente 44
input("Press Enter to continue...")
fig4 , x4 = plt.subplots(1,3)
fig4.suptitle('Paciente 44')
for idx, num in enumerate(np.linspace(0, 1, 200)):
    x4[0].imshow(vol3[idx], cmap='gray')
    x4[0].set_title('Eje transversal')
    x4[0].axis('off')
    x4[1].imshow(vol3[:,idx,:], cmap='gray')
    x4[1].set_title('Eje coronal')
    x4[1].axis('off')
    x4[2].imshow(vol3[:,:,idx], cmap='gray')
    x4[2].set_title('Eje axial')
    x4[2].axis('off')
    plt.pause(0.0001)
    if idx == 118:
        plt.clf()
        break
plt.close(fig=fig4)