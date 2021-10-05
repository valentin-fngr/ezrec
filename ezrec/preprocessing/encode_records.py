import numpy as np 
import tensorflow as tf
import PIL
import os



# todo : 
# Encode Object Detection Records 
# _serialize_examples 
# _write 
# _read 



class DetectionRecordSerializer: 

    def __init__(self, input_shape=None, label_shape=None, bbox_format="xyxy"):
        """
            input_shape : the image input tensor shape 
            label_shape : the label tensor shape 
            bbox_format : the format of the bbox. Can be xyxy or xywh
            norm_coord : boolean telling if the bbox coordinates are normalized 

        """ 
        # test init method
        self.input_shape = input_shape 
        self.label_shape = label_shape 
        if bbox_format in ["xyxy", "xywh"]: 
            self.bbox_format = bbox_format
        else: 
            raise ValueError(f"bbox format should be either xyxy or xywh, received {bbox_format}")

