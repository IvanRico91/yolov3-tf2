import tensorflow as tf

# Ruta al archivo TFRecord que deseas verificar
tfrecord_file = 'data/custom_train.tfrecord'

# Función para decodificar ejemplos de TFRecord
def decode_tfrecord_fn(serialized_example):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example

# Carga y lee el archivo TFRecord
raw_dataset = tf.data.TFRecordDataset([tfrecord_file])
parsed_dataset = raw_dataset.map(decode_tfrecord_fn)

# Recorre los ejemplos e inspecciónalos
for example in parsed_dataset:
    print("Height:", example['image/height'].numpy())
    print("Width:", example['image/width'].numpy())
    print("Filename:", example['image/filename'].numpy().decode('utf-8'))
    print("Class Labels:", example['image/object/class/label'].values)
    print("Class Texts:", [text for text in example['image/object/class/text'].values])
    # Agrega más campos según tus necesidades

# Cierra el archivo TFRecord
raw_dataset.close()
