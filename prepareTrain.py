import tensorflow as tf
import os
from PIL import Image
from absl import app, flags, logging
import hashlib
#from object_detection.utils import dataset_util
import tensorflow.compat.v1 as tf

flags.DEFINE_string('output_path', './data/custom_train.tfrecord', 'outpot dataset')

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
  """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split(' ')[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

flags = tf.app.flags
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = f.read().splitlines()
    return classes



def create_tf_example(image_path, label_path, class_names):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    image = open(image_path, 'rb').read()
    key = hashlib.sha256(image).hexdigest()
    imageAux = Image.open(image_path)
    width, height = imageAux.size
    
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)
    truncated = []
    views = []
    difficult_obj = []

    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        values = line.strip().split(' ')
        if len(values) != 5:
            continue
        class_id, x_center, y_center, box_width, box_height = map(float, values)
        x_min = x_center - (box_width / 2.0)
        x_max = x_center + (box_width / 2.0)
        y_min = y_center - (box_height / 2.0)
        y_max = y_center + (box_height / 2.0)

        xmins.append(x_min)
        xmaxs.append(x_max)
        ymins.append(y_min)
        ymaxs.append(y_max)
        if class_id >= 0 and class_id < len(class_names):
            classes_text.append(class_names[int(class_id)].encode('utf-8'))
            classes.append(int(class_id))
        else:
            classes_text.append('unknown')  # Añadir un valor predeterminado para clases desconocidas
            classes.append(int(2))
        truncated.append(0)
        views.append('Frontal'.encode('utf-8'))
        difficult_obj.append(0)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height':  int64_feature(height),
        'image/width':  int64_feature(width),
        'image/filename':  bytes_feature(image_path.encode('utf-8')),
        'image/key/sha256': bytes_feature(key.encode('utf-8')),
        'image/source_id':  bytes_feature(image_path.encode('utf-8')),
        'image/encoded':  bytes_feature(encoded_image_data),
        'image/format':  bytes_feature(b'JPEG'),
        'image/object/bbox/xmin':  float_list_feature(xmins),
        'image/object/bbox/xmax':  float_list_feature(xmaxs),
        'image/object/bbox/ymin':  float_list_feature(ymins),
        'image/object/bbox/ymax':  float_list_feature(ymaxs),
        'image/object/class/text':  bytes_list_feature(classes_text),
        'image/object/class/label':  int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(views),
    }))
        
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # Carga las clases desde el archivo 'classes.txt'
    class_file = 'data/datasets/cucumbers/names.txt'
    class_names = load_classes(class_file)

    # Define the paths to your image and label directories
    image_dir = 'data/datasets/cucumbers/Train/images'
    label_dir = 'data/datasets/cucumbers/Train/Annotations'

    image_files = os.listdir(image_dir)

    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
          image_path = os.path.join(image_dir, image_file)
          label_file = os.path.splitext(image_file)[0] + '.txt'
          label_path = os.path.join(label_dir, label_file)

          tf_example = create_tf_example(image_path, label_path,class_names)
          writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    app.run(main)
