import tensorflow as tf 
import numpy as np 
import PIL
import timeit
import os
import sys

image_folder = os.path.join("/home/valentin/Desktop/deep_learning/ezrec/ezrec/images")
image_file = os.path.join(image_folder,"0a1fe8530f26.jpg")


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def load_string_image(image_path): 
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3) 
    img = tf.cast(img, tf.float32)

    return img 

def load_image(image_path): 
    img = np.asarray(PIL.Image.open(image_path), dtype=np.float32)
    return img



def create_image_record(image_path): 
    
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img)
    image_shape = img.shape
    # flatten image to one D array
    img = tf.reshape(img, -1).numpy()


    feature = {
        "image" : tf.train.Feature(float_list=tf.train.FloatList(value=img)), 
        "shape" : tf.train.Feature(float_list=tf.train.FloatList(value=image_shape))
    }

    feature = tf.train.Features(feature=feature)

    example = tf.train.Example(features=feature)
    
    return example.SerializeToString()



def write_tf_records(records_list, file_name): 
    print(f"{len(records_list)} files given")
    with tf.io.TFRecordWriter(file_name) as writer:
        for record in records_list: 
            writer.write(record)

    print("complete")
    return 
