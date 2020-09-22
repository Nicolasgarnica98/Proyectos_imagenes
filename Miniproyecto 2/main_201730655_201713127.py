import os
import glob
import requests
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

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
    pesos = []
    sk = []
    pixel = []
    table = {}
    for i in range(0,len(comparacion)):
        cant_num = np.count_nonzero(hist == comparacion[i])
        if cant_num != 0:
            cant_pixel.append(cant_num)
            pixel.append(comparacion[i])

    suma = 0
    acum = 0
    for i in range(0,len(cant_pixel)):
        suma = cant_pixel[i] + acum
        pesos.append(cant_pixel[i]/len(hist))
        sk.append(round(((256-1)/len(hist))*suma))
        acum = suma

    sk_unicos = sorted(set(sk))
    repeticiones = []
    for i in range(0,len(sk_unicos)):
        repeticiones.append(sk.count(sk_unicos[i]))
       
    for i in range(0,len(histeq)):
        for j in range(0,len(pixel)):
            if histeq[i] == pixel[j]:
                histeq[i] = sk[j]
                break


    for i in range(0,len(hist)):
        table[str(hist[i])] = str(histeq[i]) 
    
    return histeq, table

# Especificacion de histogramas
tar_hist = np.random.normal(200,6.5,len(imagenes[0].flatten()))

# for i in range(0, len(tar_hist)):
#     tar_hist[i] = np.round(tar_hist[i],0)

# print(tar_hist)

# def MyHistEsp_201730655_201713127(image, target_hist):

#     Gzq = []

#     for i in range(0,len(target_hist)):
        
    
#     return 

    





imagen_eq = np.reshape(MyHistEq_201730655_201713127(imagenes[0].flatten())[0],(136,133))

#Gráfica
input("Press Enter to continue...")
figx, ax1 = plt.subplots(4, 2, figsize=(10, 12), gridspec_kw={'wspace':0.5,'hspace':0.7})
ax1[0][0].set_title('Target hist')
ax1[0][0].hist(tar_hist,bins=255)
ax1[1][0].set_title('Imágen original')
ax1[1][0].imshow(imagenes[0], cmap='gray')
ax1[1][0].axis('off')
ax1[2][0].set_title('Imagen equalizada')
ax1[2][0].imshow(imagen_eq, cmap = 'gray')
ax1[2][0].axis('off')
ax1[3][0].set_title('Imagen especificada')
ax1[3][0].imshow(imagen_eq)
ax1[3][0].axis('off')
# ax1[0][1].set_title('Histograma equalizado')
# ax1[0][1].hist()
ax1[0][1].axis('off')
ax1[1][1].set_title('Histograma')
ax1[1][1].hist(imagenes[0].flatten(), bins=255)
ax1[2][1].set_title('Histograma equalizado')
ax1[2][1].hist(MyHistEq_201730655_201713127(imagenes[0].flatten())[0],bins=255)
ax1[3][1].set_title('Histograma especificado')
# ax1[3][1].hist()
ax1[3][1].axis('off')
plt.show()

