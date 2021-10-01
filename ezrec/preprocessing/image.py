import tensorflow as tf 
import numpy as np 
import PIL
import timeit
import os

image_folder = os.path.join("/home/valentin/Desktop/deep_learning/ezrec/ezrec/images")
image_file = os.path.join(image_folder,"0a1fe8530f26.jpg")



def load_string_image(image_path): 
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3) 
    img = tf.cast(img, tf.float32)

    return img 

def load_image(image_path): 
    img = np.asarray(PIL.Image.open(image_path), dtype=np.float32)
    return img

