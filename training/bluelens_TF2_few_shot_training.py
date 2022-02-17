import os
import random
import numpy as np
import tensorflow as tf
from numpy import asarray
from PIL import Image
from logzero import logger

from object_detection.utils import config_util
from object_detection.builders import model_builder


# Parameters
DATA_FOLDER = '../data/'

num_classes = 3

cloud_class_id = 0
water_class_id = 1
ground_class_id = 2

category_index = {
  cloud_class_id: {'id': cloud_class_id, 'name': 'cloud'},
  water_class_id: {'id': water_class_id, 'name': 'water'},
  ground_class_id: {'id': ground_class_id, 'name': 'ground'},
}
label_2_category = {'cloud':0,'water':1,'ground':2}
# Dimension of former astro pi mission images in data/ folder
width, heigth = (2592, 1944)


import xml.etree.ElementTree as ET

# Get Image Annotation from XML Pascal VOC file
def get_img_annotations(img_xml_annotations_path):
  img_tree = ET.parse(img_xml_annotations_path)
  img_root = img_tree.getroot()
  labels_list = []
  bbox_list = []
  for child in img_root:
    if child.tag == "object":
      for element in child:
        if element.tag == "name":
          labels_list.append(element.text)
        if element.tag == "bndbox":
          bbox = []
          for coordinates in element:
            if coordinates.tag == "xmin":
              bbox.append(coordinates.text)
            if coordinates.tag == "ymin":
              bbox.append(coordinates.text)
            if coordinates.tag == "xmax":
              bbox.append(coordinates.text)
            if coordinates.tag == "ymax":
              bbox.append(coordinates.text)
          bbox_list.append(bbox)
  return labels_list, bbox_list


  # Prepare Bounding Box so that it is normalized & invariant from image scale dimension
def get_prepared_bbox(bbox, width, heigth):
  for coord in bbox:
    xmin_norm=int(bbox[0])/width
    ymin_norm=int(bbox[1])/heigth
    xmax_norm=int(bbox[2])/width
    ymax_norm=int(bbox[3])/heigth
  prepared_bbox = [ymin_norm,xmin_norm,ymax_norm,xmax_norm] # Object Detection Viz utils expected format
  return prepared_bbox


  # Get name of astropi Jpeg images
def get_images_name(data_folder):
  data = os.listdir(data_folder)
  images_file_list = [file for file in data if os.path.splitext(file)[1] != '.xml'] # remove xml file
  images_file_list.sort()
  return images_file_list

# Get name of astropi XML annotations
def get_bboxes_file_name(data_folder):
  data = os.listdir(data_folder)
  bbox_file_list = [file for file in data if os.path.splitext(file)[1] != '.jpg']
  bbox_file_list.sort()
  return bbox_file_list

# Get a List of astropi numpy images
def get_numpy_images(images_file,data_folder):
  np_images_list = []
  for image_file in images_file:
    img_pil = Image.open(data_folder+image_file)
    img_np = asarray(img_pil)
    np_images_list.append(img_np)
  return np_images_list

# Get a List of bboxes & classes for images
def get_gt_box_class(bbox_file_list,data_folder):
  gt_boxes_list = []
  classes_list = []
  for bboxes in bbox_file_list: 
    classes,gt_boxes=get_img_annotations(data_folder+bboxes)
    gt_boxes_np = np.array(gt_boxes,dtype=np.float32)
    gt_boxes_list.append(gt_boxes_np)
    classes_list.append(classes)
  return gt_boxes_list, classes_list
  
# Get a List of class_id for images
def get_class_id(classes_list): 
  classes_id_list = []
  for classes in classes_list:
    id_list  = []
    for label in classes:
      id = label_2_category[label]
      id_list.append(id)
    classes_id_list.append(id_list)
  return classes_id_list


# Prepare Data for Training: convert all numpy arrays & lists to TF tensors

def get_tensors_data(np_images_list,gt_boxes_list,classes_id_list):
    image_tensors = [] # images tensor list from numpy array
    gt_box_tensors = [] # bbox tensor list from gt_box list
    class_one_hot_tensors = [] # 1-hot class tensor list from class list

    for (image_np, gt_box_np,class_id) in zip(np_images_list, gt_boxes_list,classes_id_list):
        image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        indexed_classes = tf.convert_to_tensor(class_id,dtype=tf.int32)
        class_one_hot_tensors.append(tf.one_hot(
            indexed_classes, num_classes))
    return image_tensors,gt_box_tensors,class_one_hot_tensors


# Main function definition
def main():

  logger.info(f'Getting Images XML Annotations')

  # Grab Data from raw JPEG & XML files to lists
  images_list = get_images_name(DATA_FOLDER)
  np_images_list = get_numpy_images(images_list,DATA_FOLDER)
  bboxes_file_list = get_bboxes_file_name(DATA_FOLDER)
  gt_boxes_list, classes_list = get_gt_box_class(bboxes_file_list,DATA_FOLDER)
  classes_id_list = get_class_id(classes_list)


  tf.keras.backend.clear_session()

  # Pipeline & Checkpoint config
  logger.info('Building model and restoring weights for fine-tuning...')
  num_classes = 3
  pipeline_config = '../models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config'
  checkpoint_path = '../models/research/object_detection/test_data/checkpoint/ckpt-0'

  # This will be where we save checkpoint & config for TFLite conversion later.
  output_directory = '../output/'
  output_checkpoint_dir = os.path.join(output_directory, 'checkpoint')

  # Load pipeline config and build a detection model.
  configs = config_util.get_configs_from_pipeline_file(pipeline_config)
  model_config = configs['model']
  model_config.ssd.num_classes = num_classes # to set number of classes to 3
  model_config.ssd.freeze_batchnorm = True
  detection_model = model_builder.build(
        model_config=model_config, is_training=True) # the actual SSD Mobilenet Detection model
        
  # Save new pipeline config
  pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
  config_util.save_pipeline_config(pipeline_proto, output_directory)

  # To save checkpoint for TFLite conversion.
  exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
  ckpt_manager = tf.train.CheckpointManager(
      exported_ckpt, output_checkpoint_dir, max_to_keep=1)

  # Run model through a dummy image so that variables are created
  image, shapes = detection_model.preprocess(tf.zeros([1, 320, 320, 3]))
  prediction_dict = detection_model.predict(image, shapes)
  _ = detection_model.postprocess(prediction_dict, shapes)
  logger.info('Weights restored!')


  # Training Data
  logger.info('Preparing Data')
  image_tensors,gt_box_tensors,class_one_hot_tensors = get_tensors_data(np_images_list,gt_boxes_list,classes_id_list)

  # Training parameters

  tf.keras.backend.set_learning_phase(True)

  batch_size = 8
  learning_rate = 0.15
  num_batches = 30

  # Select variables in top layers to fine-tune.
  trainable_variables = detection_model.trainable_variables
  to_fine_tune = []
  prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
  for var in trainable_variables:
    if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
      to_fine_tune.append(var)


  # Set up forward + backward pass for a single train step.
  def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                      groundtruth_boxes_list,
                      groundtruth_classes_list):
      shapes = tf.constant(batch_size * [[320, 320, 3]], dtype=tf.int32)
      model.provide_groundtruth(
          groundtruth_boxes_list=groundtruth_boxes_list,
          groundtruth_classes_list=groundtruth_classes_list)
      with tf.GradientTape() as tape:
        preprocessed_images = tf.concat(
            [detection_model.preprocess(image_tensor)[0]
            for image_tensor in image_tensors], axis=0)
        prediction_dict = model.predict(preprocessed_images, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        gradients = tape.gradient(total_loss, vars_to_fine_tune)
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
      return total_loss

    return train_step_fn


  # SGD optimizer
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  train_step_fn = get_model_train_step_function(
      detection_model, optimizer, to_fine_tune)

  # Training using train_step function
  logger.info('Start fine-tuning Training!')
  for idx in range(num_batches):
    # Grab keys for a random subset of examples
    all_keys = list(range(len(np_images_list)))
    random.shuffle(all_keys)
    example_keys = all_keys[:batch_size]

    # Instantiate gt_boxes, gt_classes, images 
    gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
    gt_classes_list = [class_one_hot_tensors[key] for key in example_keys]
    image_tensors = [image_tensors[key] for key in example_keys]

    # Training step (forward pass + backwards pass)
    total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

    if idx % 2 == 0:
      logger.info('batch ' + str(idx) + ' of ' + str(num_batches)
      + ', loss=' +  str(total_loss.numpy()))

  logger.info('Done fine-tuning!')

  ckpt_manager.save()
  logger.info('Checkpoint saved!')


# Main execution
if __name__ == '__main__':
    main()
