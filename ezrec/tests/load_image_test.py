import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../ezrec")
from  preprocessing.image import load_image, load_string_image, create_image_record, write_tf_records
import unittest
import PIL
import numpy as np


image_folder = os.path.join("/home/valentin/Desktop/deep_learning/ezrec/ezrec/images")
image_path = os.path.join(image_folder,"0a1fe8530f26.jpg")



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

if __name__ == '__main__':
    unittest.main()