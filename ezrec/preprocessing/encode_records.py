import numpy as np 
import tensorflow as tf
import PIL
import os
import sys


# todo : 
# Encode Object Detection Records 
# _serialize_examples 
# _write 
# _read 



class DetectionRecordSerializer: 
    """
        DetectionRecordSerializer is responsible for creating tf records and saving them. 
    """
    def __init__(self, input_shape=None, label_shape=None, bbox_format="xyxy", bbox_format_name=["x1", "x2", "y1", "y2", "id"]):
        """
            input_shape : the image input tensor shape 
            label_shape : the label tensor shape 
            bbox_format : the format of the bbox. Can be xxyy, xyxy or xywh. You can find the informations in the label file of your dataset.
            norm_coord : boolean telling if the bbox coordinates are normalized 

        """ 
        # test init method
        self.input_shape = input_shape 
        self.label_shape = label_shape 
        if len(bbox_format_name) != 5:
            # TODO : test raise  
            raise ValueError("Please, enter 5 keys to retrieve coordinates + object label")
        else: 
            self.bbox_format_name = bbox_format_name
        if bbox_format in ["xyxy", "xxyy", "xywh"]: 
            self.bbox_format = bbox_format
        else: 
            raise ValueError(f"bbox format should be either xyxy or xywh, received {bbox_format}")


    def _create_example(self,image_path, bbox): 
        """
            given an image_path and a label, read the image, reshape it if needed, 
            encode it and encode its associated label
        """ 

        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img)
        # normalize the image 
        img = tf.cast(img, tf.float64) / 255.0
        init_height , init_width = img.shape[:2]
        if self.input_shape and self.input_shape[:2] != (init_height, init_width):
            
            height, width = self.input_shape[:2]
            img = tf.image.resize(img, (height, width)) 
   

        # flatten image to one dimensional array
        #img = tf.reshape(img, -1).numpy()
        img = tf.io.encode_jpeg(tf.cast(img, dtype="uint8")).numpy()


        label_ids = []

        if self.bbox_format == "xxyy" or self.bbox_format == "xyxy": 
            # IMPORTANT : WHEN CREATING THE FEATURE, EVEN THO THE BBOX FORMAT WAS XXYY, WE ARE USING the XYXY FORMAT WHEN CREATING EXAMPLES !!!
            # THIS RULE DOESN'T APPLY IF THE GIVEN BBOX FORMAT WAS XYWH 
            xmins, xmaxs = [], []
            ymins, ymaxs = [], [] 
            
            # bbox is a list of dict
            if self.bbox_format == "xxyy": 
                for box in bbox:
                    xmins.append(box[self.bbox_format_name[0]])
                    xmaxs.append(box[self.bbox_format_name[1]]) 
                    ymins.append(box[self.bbox_format_name[2]]) 
                    ymaxs.append(box[self.bbox_format_name[3]])
                    label_ids.append(box[self.bbox_format_name[4]])
                
                
            
            else: 
                # case xyxy 
                for box in bbox:
                    xmins.append(box[self.bbox_format_name[0]])
                    ymins.append(box[self.bbox_format_name[1]]) 
                    xmaxs.append(box[self.bbox_format_name[2]]) 
                    ymaxs.append(box[self.bbox_format_name[3]])
                    label_ids.append(box[self.bbox_format_name[4]])
            
            # creat record
            feature = {
                "image/initial_height" : tf.train.Feature(int64_list=tf.train.Int64List(value=[init_height])), 
                "image/initial_width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[init_height])),
                "image/height" :  tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                "image/width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                "image/encoded" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                "image/obj/xmins": tf.train.Feature(int64_list=tf.train.Int64List(value=xmins)), 
                "image/obj/ymins": tf.train.Feature(int64_list=tf.train.Int64List(value=ymins)),
                "image/obj/xmaxs": tf.train.Feature(int64_list=tf.train.Int64List(value=xmaxs)),
                "image/obj/ymaxs": tf.train.Feature(int64_list=tf.train.Int64List(value=ymaxs)), 
                "image/obj/class_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=label_ids)), 
            }

        else: 
            center_xs, center_ys = [], [] 
            widths, heights = [], []
            for box in bbox: 
                center_xs.append(box[self.bbox_format_name[0]])
                center_ys.append(box[self.bbox_format_name[1]]) 
                widths.append(box[self.bbox_format_name[2]]) 
                heights.append(box[self.bbox_format_name[3]])

            feature = {
                "image/initial_height" : tf.train.Feature(int64_list=tf.train.Int64List(value=[init_height])), 
                "image/initial_width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[init_height])),
                "image/height" :  tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                "image/width" : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                "image/encoded" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                "image/obj/center_xs": tf.train.Feature(int64_list=tf.train.Int64List(value=center_xs)), 
                "image/obj/center_ys": tf.train.Feature(int64_list=tf.train.Int64List(value=center_ys)),
                "image/obj/widths": tf.train.Feature(int64_list=tf.train.Int64List(value=widths)),
                "image/obj/heights": tf.train.Feature(int64_list=tf.train.Int64List(value=heights)), 
                "image/obj/class_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=label_ids)), 
            }
        
        features = tf.train.Features(feature=feature)

        example = tf.train.Example(features=features)

        return example.SerializeToString()


    def create_records(self, image_paths, bboxs, tfrecords_dir, num_sample=300, file_prefix=None): 
        """
            given a list of image_paths and a list of corresponding bboxs, 
            parse all examples into seralized string and save them as tfrecords
            
            Arguments: 
                image_paths : a list of image_paths 
                bboxs : a list of bboxs
                tfrecords_path : the path where the tfrecords file will be saved
                num_sample : the number for sample per files 
                file_prefix : if None doesn't set a file prefix. Otherwise, save the .tfrecords file as <prefix>_<number>.tfrecords

        """

        # TODO : divide into multiple tf records if too big

        if len(image_paths) != len(bboxs): 
            raise ValueError("Image_paths list and bboxs list should have the same length !")

        num_tfrecords = len(image_paths) // num_sample
        if len(image_paths) % num_sample != 0: 
            num_tfrecords += 1
        
        if not os.path.exists(tfrecords_dir): 
            os.makedirs(tfrecords_dir)

        for i in range(num_tfrecords): 
            sample_path = image_paths[i*num_sample:(i+1)*num_sample]
            sample_bboxs = bboxs[i*num_sample:(i+1)*num_sample]

            with tf.io.TFRecordWriter(f'{tfrecords_dir}/file_{i}_{len(sample_path)}.tfrecords') as writer:
                
                for i in range(len(sample_path)): 
                    serialized_example = self._create_example(sample_path[i], sample_bboxs[i]) 
                    writer.write(serialized_example)
                print(f"Successfuly created tf record file file_{i}_{len(sample_path)}.tfrecords")
                print()

        return 

    def to_record_dataset(self, directory): 
        """ 
            Read .tfrecords file in a given directory and return tf.data.TFRecordDataset
        """ 
        if not os.path.exists(directory): 
            raise ValueError("Unable to find directory", directory) 
        
        if len(os.listdir(directory)):
            tfrecord_filenames = [os.path.join(directory, filename) for filename in os.listdir(directory)]
            raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames) 

            return raw_dataset


    def parse_dataset(self, raw_dataset): 
        """
            return the parsed dataset, previously loaded using the to_record_dataset 
            This is equivalent to : dataset.map(parse_function)
            Arguments: 
                raw_dataset : the unparsed dataset
            
        """

        if self.bbox_format == ["xyxy"] or self.bbox_format == ["xxyy"]: 
            
            feature_description = {
                "image/initial_height" : tf.io.FixedLenFeature([], tf.int64), 
                "image/initial_width" : tf.io.FixedLenFeature([], tf.int64),
                "image/height" :  tf.io.FixedLenFeature([], tf.int64),
                "image/width" : tf.io.FixedLenFeature([], tf.int64),
                "image/encoded" : tf.io.FixedLenFeature([], tf.string),
                "image/obj/xmins": tf.io.VarLenFeature([], tf.int64), 
                "image/obj/ymins": tf.io.VarLenFeature([], tf.int64),
                "image/obj/xmaxs": tf.io.VarLenFeature([], tf.int64),
                "image/obj/ymaxs": tf.io.VarLenFeature([], tf.int64), 
                "image/obj/class_ids": tf.io.VarLenFeature([], tf.int64), 
            }

                        
        elif self.bbox_format == ["xywh"]: 
            
            feature_description = {
                "image/initial_height" : tf.io.FixedLenFeature([], tf.int64), 
                "image/initial_width" : tf.io.FixedLenFeature([], tf.int64),
                "image/height" :  tf.io.FixedLenFeature([], tf.int64),
                "image/width" : tf.io.FixedLenFeature([], tf.int64),
                "image/encoded" : tf.io.FixedLenFeature([], tf.string),
                "image/obj/center_xs": tf.io.VarLenFeature([], tf.int64), 
                "image/obj/center_ys": tf.io.VarLenFeature([], tf.int64),
                "image/obj/widths": tf.io.VarLenFeature([], tf.int64),
                "image/obj/heights": tf.io.VarLenFeature([], tf.int64), 
                "image/obj/class_ids": tf.io.VarLenFeature([], tf.int64), 
            }

        
        def _parse_image_function(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        # map the raw dataset 

        return raw_dataset.map(_parse_image_function)