import tensorflow as tf

# Ruta al archivo TFRecord que deseas analizar
tfrecord_file = 'data/custom_train.tfrecords'

# Define el rango permitido para los valores de los labels
label_min = 0
label_max = 1  # Por ejemplo, el rango podría ser 0-19

# Función para decodificar ejemplos de TFRecord
def decode_tfrecord_fn(serialized_example):
    feature_description = {
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example

# Carga y lee el archivo TFRecord
raw_dataset = tf.data.TFRecordDataset([tfrecord_file])
parsed_dataset = raw_dataset.map(decode_tfrecord_fn)

# Recorre los ejemplos e inspecciona los valores de los labels
for example in parsed_dataset:
    labels = example['image/object/class/label'].values

    for label_value in labels:
        if label_value < label_min or label_value > label_max:
            print(f"Valor de label fuera del rango permitido: {label_value}")
        # Puedes agregar más lógica según tus necesidades, como contar valores, etc.

# Cierra el archivo TFRecord
raw_dataset.close()
