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

def get_pad_resized_img(size,img_tensor):
    resized_img = tf.image.resize_with_pad(
        img_tensor, size,size,
        antialias=False
    )
    resized_img = tf.cast(resized_img,dtype=tf.uint8)
    return resized_img


def get_tflite_prepared_image(image_name):
  image_file = image_name + '.jpg'
  img_pil = Image.open(DATA_FOLDER+image_file)
  img_np = asarray(img_pil)
  img_tensor = tf.expand_dims(tf.convert_to_tensor(
    img_np, dtype=tf.float32), axis=0)
  preprocessed_img =  get_pad_resized_img(320,img_tensor)
  return preprocessed_img


def get_tflite_predictions(image_name,tflite_interpreter):
  # Prepare image
  preprocessed_img = get_tflite_prepared_image(image_name)
  input_tensor = preprocessed_img
  input_tensor = tf.cast(input_tensor,dtype=tf.float32)
  image_tflite_np = input_tensor.numpy()


  input_details = tflite_interpreter.get_input_details()
  output_details = tflite_interpreter.get_output_details() 

  # Prepare data for TFLite inference
  input_shape = input_details[0]['shape'] # tflite model input shape
  input_data = image_tflite_np.reshape(input_shape) # tflite model input data as numpy

  # This load input_data into tlfite interpreter
  tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
  tflite_interpreter.invoke()

  # All useful Data from tflite inference: boxes, classes, scores
  boxes = tflite_interpreter.get_tensor(output_details[1]['index'])
  classes = tflite_interpreter.get_tensor(output_details[3]['index'])
  scores = tflite_interpreter.get_tensor(output_details[0]['index'])

  return boxes, classes, scores



def main(argv):

    logger.info('Starting inference process')
    # image name from parser input
    img = argv.image_name

    # Load the TFLite model and allocate tensors.
    logger.info('Loading object detection model')
    tflite_model_path  = '../tflite/model.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get Prediction
    logger.info('Making inference')
    predictions =  get_tflite_predictions(img,interpreter)
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