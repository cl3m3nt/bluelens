import csv
import os
from logzero import logger
import numpy as np
import pandas as pd
import tensorflow as tf
from time import sleep
from datetime import datetime, timedelta
from numpy import asarray
from orbit import ISS
from picamera import PiCamera
from PIL import Image
from pathlib import Path
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

# Parameters
DIR_PATH = Path(__file__).parent.resolve()
DATA_FILE = DIR_PATH/'data.csv'
MODEL_FILE = 'efficientdet_bluelens_edgetpu.tflite'

# RPI Camera settings
cam = PiCamera()
cam.resolution = (1296,972)


def convert(angle):
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (rationals)
    Return a tuple containing a boolean and the converted angle,
    with the boolean indicating if the angle is negative.
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle


def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    point = ISS.coordinates()

    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(point.latitude)
    west, exif_longitude = convert(point.longitude)

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # Capture the image
    camera.capture(image+'.jpg')


def get_iss_gps():
    """Return ISS GPS coordinates as Degre Decimal
    Returns:
        latitude/longitude (tuple): latitude and longitude as tuple
    """
    location = ISS.coordinates()
    latitude = location.latitude.degrees
    longitude = location.longitude.degrees
    return latitude,longitude


# Load  TFLite model and allocate tensors.
def load_tflite_model(model_path):
    """Helper function to load the tflite object detection model
    Args:
        model_path (str): path of tflite for tpu model
    Returns:
        interpreter: the actual tflite interpreter/model
    """
    tflite_model_path  = model_path
    interpreter = edgetpu.make_interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter


def prepare_image(img_path,interpreter):
    """Prepare an image so that it's ready for interpreter inference
    Args:
        img_path (str): the path of the image to open
        interpreter(object): the tflite interpreter
    Returns:
        img (PIL): the opened image
        scale (tuple): ratio applied to image
    """
    image = Image.open(img_path)
    img, scale = common.set_resized_input(
    interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    return img,scale 


def do_inference(img_path,interpreter):
    """Do inference on image using object detection interpreter/model
    Args:
        img_path (str): the path of the image to open
        interpreter(object): the tflite interpreter
    Returns:
        obj_detection (object): the object detections done by interpreter/model
    """
    img,scale = prepare_image(img_path,interpreter)
    interpreter.invoke()
    obj_detection = detect.get_objects(interpreter, score_threshold=0.31, image_scale=scale)
    return obj_detection

def create_csv(data_file):
    """Create csv file to store bluelens experiment data
    Args:
        data_file (str): the file path where to store data
    """
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Datetime", "Location", "Picture Name", "Prediction")
        writer.writerow(header)

def add_csv_data(data_file,timestamp,picture_name,prediction):
    """Add data to bluelens experiment data file
    Args:
        data_file (str): the file path where to store data
        timestamp (datetime): datetime when data is added
        picture_name (str): name of picture data is recorded about
        predictions (object): object detection list from do_infernce
    """
    row = (timestamp, picture_name+'.jpg', get_iss_gps(), prediction)
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


# Main function definition
def main():
    # Starting Experiment
    start_time = datetime.now()
    logger.info(f'Starting Blulens Astro Pi team experiment at {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Creating Log File
    logger.info(f'Creating Log file at {str(DATA_FILE)}')
    create_csv(DATA_FILE)

    # Loading TFLite model
    interpreter = load_tflite_model(str(DIR_PATH)+'/'+MODEL_FILE)

    # Starting loop over 3 hours
    i = 0
    now_time = datetime.now()
    while (now_time < start_time + timedelta(minutes=165)):
        # Taking Earth Picture
        picture_name = f'bluelens_image_{i}'
        capture(cam,str(DIR_PATH/picture_name))

        # Doing Prediction
        tpu_prediction = do_inference(picture_name+'.jpg',interpreter)
        logger.info(f'Doing inference, {tpu_prediction}')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Writing data to CSV file
        add_csv_data(DATA_FILE,timestamp,picture_name,tpu_prediction)

        i = i+1
        now_time = datetime.now()
        
        # 5 seconds sleep to avoid taking too many pictures
        sleep(5)

    end_time = datetime.now()
    logger.info(f'Finishing Bluelens Astro Pi team experiment at {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    experiment_time = end_time - start_time
    logger.info(f'Bluelens Astro Pi team experiment run time {experiment_time}')

# Main execution
if __name__ == '__main__':
    main()
