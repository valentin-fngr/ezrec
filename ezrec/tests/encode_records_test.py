import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../ezrec")
from preprocessing.encode_records import DetectionRecordSerializer
import tensorflow as tf
import numpy as np 

image_folder = os.path.join("/home/valentin/Desktop/deep_learning/ezrec/ezrec/images")
image_path = os.path.join(image_folder,"0a1fe8530f26.jpg")
tfrecords_dir = os.path.join(os.getcwd(), "test_records_dir") 

bbox_xyxy = [
    {
        "x1" : 1, 
        "y1" : 1, 
        "x2" : 2, 
        "y2" : 2, 
        "id": 0
    }, 
    {
        "x1" : 1, 
        "y1" : 1, 
        "x2" : 2, 
        "y2" : 2, 
        "id": 10
    }
]


class TestDetectionRecordSerializer(unittest.TestCase): 

    # def test_can_init_serializer(self): 
    #     input_shape = (224,224,3) 
    #     label_shape = (7,7,20) 
    #     bbox_format = "xyxy"
    #     odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
    #     self.assertTrue(isinstance(odr_serializer, DetectionRecordSerializer))

    # def test_raise_error_on_instanciation(self): 
    #     input_shape = (224,224,3) 
    #     label_shape = (7,7,20) 
    #     bbox_format = "yxyx"
    #     with self.assertRaises(ValueError):
    #         DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        
    # def test_create_example_xyxy(self): 
    #     input_shape = (224,224,3) 
    #     label_shape = (7,7,20) 
    #     bbox_format = "xyxy"
    #     odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
    #     example = odr_serializer._create_example(image_path, bbox_xyxy)
    #     print(f"size of parsed example : {sys.getsizeof(example)}")
    #     self.assertTrue(isinstance(example, bytes))

    # def test_create_example_xxyy(self): 
    #     input_shape = (224,224,3) 
    #     bbox_format = "xxyy"
    #     label_shape = (7,7,20) 
    #     odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
    #     example = odr_serializer._create_example(image_path, bbox_xyxy)
    #     print(f"size of parsed example : {sys.getsizeof(example)}")
    #     self.assertTrue(isinstance(example, bytes))

    # def test_create_example_xywh(self): 
    #     input_shape = (224,224,3) 
    #     bbox_format = "xywh"
    #     label_shape = (7,7,20) 
    #     odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
    #     example = odr_serializer._create_example(image_path, bbox_xyxy)
    #     print(f"size of parsed example : {sys.getsizeof(example)}")
    #     self.assertTrue(isinstance(example, bytes))


    # def test_save_records(self): 
    #     input_shape = (224,224,3) 
    #     bbox_format = "xywh"
    #     label_shape = (7,7,20) 
    #     odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
    #     image_paths = [image_path for i in range(1000)] 
    #     bboxs = [bbox_xyxy for i in range(1000)]
    #     odr_serializer.create_records(image_paths, bboxs, tfrecords_dir,file_prefix="tests") 

    #     self.assertTrue(len(os.listdir(tfrecords_dir)) == 4)
        
    #     # delete files then delete folder 
    #     for file_path in os.listdir(tfrecords_dir): 
    #         os.remove(os.path.join(tfrecords_dir,file_path))
    #     os.rmdir(tfrecords_dir)

    def test_to_record_dataset(self): 
        # create .tfrecords 
        print("TESTING DAZTASET !")
        input_shape = (224,224,3) 
        bbox_format = "xywh"
        label_shape = (7,7,20) 
        odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        image_paths = [image_path for i in range(1000)] 
        bboxs = [bbox_xyxy for i in range(1000)]
        odr_serializer.create_records(image_paths, bboxs, tfrecords_dir)

        raw_dataset = odr_serializer.to_record_dataset(tfrecords_dir) 
        
        for sample in raw_dataset.take(2): 
            print(repr(sample))

        print(raw_dataset.reduce(np.int64(0), lambda x,_ : x + 1))
        self.assertEqual(raw_dataset.reduce(np.int64(0), lambda x,_ : x + 1), 1000)
