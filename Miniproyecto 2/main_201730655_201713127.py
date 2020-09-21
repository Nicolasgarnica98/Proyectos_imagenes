import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import rgb2gray

#Ruta de la imagen
img = 'https://cdn.shopify.com/s/files/1/1280/3657/products/GM-BA-SET01_1_e5745bf6-253a-4a83-a8d1-5d55ee691e5d_1024x1024.jpg?v=1548113518'

#Obtener imagen de la ruta URL
r = requests.get(img)
with open('monedas.jpg', 'wb') as f:
          f.write(r.content)

#Conversion de la imagen a grises
imagen_monedas = io.imread(os.path.join('monedas.jpg'))
imagen_monedas_grises = rgb2gray(imagen_monedas)