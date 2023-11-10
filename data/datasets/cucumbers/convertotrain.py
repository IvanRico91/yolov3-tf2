import os

# Ruta al directorio que contiene las imágenes
directorio = 'images/'

# Nombre del archivo de texto donde se guardarán los nombres de las imágenes
archivo_txt = 'train.txt'

# Lista todos los archivos en el directorio
archivos = os.listdir(directorio)

# Lista de extensiones de archivos de imagen compatibles
extensiones_imagen = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# Abre el archivo de texto en modo escritura
with open(archivo_txt, 'w') as archivo:
    for archivo_nombre in archivos:
        # Verifica si el archivo es una imagen
        if any(archivo_nombre.endswith(extension) for extension in extensiones_imagen):
            # Elimina la extensión del nombre del archivo
            nombre_sin_extension, _ = os.path.splitext(archivo_nombre)
            archivo.write(nombre_sin_extension + '\n')

print(f"Se han guardado los nombres de las imágenes en '{archivo_txt}'.")
