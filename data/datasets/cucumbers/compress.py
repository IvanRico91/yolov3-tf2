from PIL import Image
import os

# Ruta al directorio que contiene las imágenes
directorio = 'images/'

# Calidad de compresión (0 es la peor calidad, 100 es la mejor)
calidad = 50  # Puedes ajustar este valor según tus necesidades

# Lista todos los archivos en el directorio
archivos = os.listdir(directorio)

for archivo in archivos:
    # Verifica si el archivo es una imagen (puedes agregar más extensiones si es necesario)
    if archivo.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        ruta_completa = os.path.join(directorio, archivo)

        # Abre la imagen
        imagen = Image.open(ruta_completa)

        # Guarda la imagen con la calidad deseada
        imagen.save(ruta_completa, optimize=True, quality=calidad)
        print(ruta_completa)

print("Proceso completado.")
