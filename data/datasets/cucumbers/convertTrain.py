import tensorflow as tf
from PIL import Image
import os

# Rutas a las imágenes y etiquetas
image_dir = 'images/'
label_dir = 'labels/'
output_file = 'dataset.tfrecords'

def create_tf_example(image_path, label_path):
    # Lee la imagen
    image = Image.open(image_path)
    width, height = image.size
    image_bytes = open(image_path, "rb").read()

    # Lee la etiqueta
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Convierte las etiquetas en un ejemplo TFRecord
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'JPEG']),
        'image/object/bbox': tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature)
    
    return example

# Crea el archivo TFRecord
writer = tf.io.TFRecordWriter(output_file)

image_files = os.listdir(image_dir)
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_file = image_file.replace('.jpg', '.txt')
    label_path = os.path.join(label_dir, label_file)
    
    tf_example = create_tf_example(image_path, label_path)
    writer.write(tf_example.SerializeToString())

writer.close()
