import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../ezrec")
from  preprocessing.image import load_image, load_string_image, create_image_record, write_tf_records, read_records_file
import unittest
import PIL
import numpy as np
import time
import tensorflow as tf



image_folder = os.path.join("/home/valentin/Desktop/deep_learning/ezrec/ezrec/images")
image_path = os.path.join(image_folder,"0a1fe8530f26.jpg")
image_shape = [1336, 2048,3]

test_image = np.asarray(PIL.Image.open(image_path), dtype=np.float64) / 255.0

class TestRecordMethods(unittest.TestCase): 

    def test_create_save_records(self): 
        """
            Create tf record, encode the image and its shape. 
            Retrieve the data 

            TODO : add more records
        """
        records_path = "test_records"

        tf_record_example = create_image_record(image_path) 
        records = [tf_record_example]
        write_tf_records(records, "test_records")

        # check if the file has been created
        self.assertTrue(os.path.exists("test_records"))
    
        os.remove("test_records")


    def test_write_read_records(self): 
        """
            Create a tf train example, write the tf record file and decode it
        """

        records_path = "test_records"

        tf_record_example = create_image_record(image_path) 
        records = [tf_record_example for i in range(100)]
        
        start = time.time()
        write_tf_records(records, "test_records")
        end = time.time() 
        print(f"writing took {end - start} seconds")

        # read the tf records file 
        parsed_dataset = read_records_file("test_records")
        # checking for the first element as all elements are the same 
        for example in parsed_dataset.take(1): 
            # image is a sparse tensor
            image, shape = example["image"], example["shape"]
            self.assertEqual(shape[0], image_shape[0])
            self.assertEqual(shape[1], image_shape[1])
            self.assertEqual(shape[2], image_shape[2])
            
            image = tf.sparse.to_dense(image)
            image = np.reshape(image.numpy(), shape)
            self.assertTrue(np.max(image) <= 1.0)
            
        return 

if __name__ == '__main__':
    unittest.main()