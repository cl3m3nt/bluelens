import argparse
import os
import numpy as np
from numpy import asarray
import pandas as pd
from PIL import Image
import tensorflow as tf
from logzero import logger

# Parameters
DATA_FOLDER = '../data/'


# Helper Functions

def get_pad_resized_img(size,img_tensor):
    resized_img = tf.image.resize_with_pad(
        img_tensor, size,size,
        antialias=False
    )
    resized_img = tf.cast(resized_img,dtype=tf.uint8)
    return resized_img


def get_predictions(image_name,model_fn):
  image_path = DATA_FOLDER+image_name+'.jpg'
  image = Image.open(image_path)
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  input_tensor = get_pad_resized_img(320,input_tensor)
  output_dict = model_fn(input_tensor)
  return output_dict
  

def get_prediction_bboxes(image_name,model_fn):
  image_path = DATA_FOLDER+image_name+'.jpg'
  image = Image.open(image_path)
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  input_tensor = get_pad_resized_img(320,input_tensor)
  output_dict = model_fn(input_tensor)
  bboxes = output_dict['detection_boxes'].numpy()
  return bboxes[0]


def get_prediction_classes(image_name,model_fn):
  image_path = DATA_FOLDER+image_name+'.jpg'
  image = Image.open(image_path)
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  input_tensor = get_pad_resized_img(320,input_tensor)
  output_dict = model_fn(input_tensor)
  classes = output_dict['detection_classes'].numpy()
  return classes[0]


def get_prediction_scores(image_name,model_fn):
  image_path = DATA_FOLDER+image_name+'.jpg'
  image = Image.open(image_path)
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  input_tensor = get_pad_resized_img(320,input_tensor)
  output_dict = model_fn(input_tensor)
  classes = output_dict['detection_scores'].numpy()
  return classes[0]


def get_predictions_all(image_name,model_fn):
  image_path = DATA_FOLDER+image_name+'.jpg'
  image = Image.open(image_path)
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  input_tensor = get_pad_resized_img(320,input_tensor)
  output_dict = model_fn(input_tensor)
  bboxes = output_dict['detection_boxes'].numpy()
  classes = output_dict['detection_classes'].numpy()
  scores = output_dict['detection_scores'].numpy()
  return bboxes,classes,scores



def main(argv):

  logger.info('Starting inference process')
  # image name from parser input
  img = argv.image_name

  # Load savedModel
  logger.info('Loading object detection model')

  tf.keras.backend.clear_session()
  model = tf.saved_model.load('../export/saved_model')
  model_fn = model.signatures['serving_default']

  # Get prediction 
  logger.info('Making inference')
  predictions = get_predictions_all(img,model_fn)
  print(predictions)

  logger.info('Ending successfully inference process')

# Defining parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_name', required=True,
                    help='image name in data folder to make inference on')

# Create args parsing standard input
args = parser.parse_args()

# Main execution
if __name__ == '__main__':
    main(args)

