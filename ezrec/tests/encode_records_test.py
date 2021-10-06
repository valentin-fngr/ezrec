import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../ezrec")
from preprocessing.encode_records import DetectionRecordSerializer
import tensorflow as tf


image_folder = os.path.join("/home/valentin/Desktop/deep_learning/ezrec/ezrec/images")
image_path = os.path.join(image_folder,"0a1fe8530f26.jpg")

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

    def test_can_init_serializer(self): 
        input_shape = (224,224,3) 
        label_shape = (7,7,20) 
        bbox_format = "xyxy"
        odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        self.assertTrue(isinstance(odr_serializer, DetectionRecordSerializer))

    def test_raise_error_on_instanciation(self): 
        input_shape = (224,224,3) 
        label_shape = (7,7,20) 
        bbox_format = "yxyx"
        with self.assertRaises(ValueError):
            DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        
    def test_create_example_xyxy(self): 
        input_shape = (224,224,3) 
        label_shape = (7,7,20) 
        bbox_format = "xyxy"
        odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        example = odr_serializer._create_example(image_path, bbox_xyxy)
        print(f"size of parsed example : {sys.getsizeof(example)}")
        self.assertTrue(isinstance(example, bytes))

    def test_create_example_xxyy(self): 
        input_shape = (224,224,3) 
        bbox_format = "xxyy"
        label_shape = (7,7,20) 
        odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        example = odr_serializer._create_example(image_path, bbox_xyxy)
        print(f"size of parsed example : {sys.getsizeof(example)}")
        self.assertTrue(isinstance(example, bytes))

    def test_create_example_xywh(self): 
        input_shape = (224,224,3) 
        bbox_format = "xywh"
        label_shape = (7,7,20) 
        odr_serializer = DetectionRecordSerializer(input_shape, label_shape, bbox_format) 
        example = odr_serializer._create_example(image_path, bbox_xyxy)
        print(f"size of parsed example : {sys.getsizeof(example)}")
        self.assertTrue(isinstance(example, bytes))
