import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../ezrec")
from preprocessing.encode_records import DetectionRecordSerializer



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
        