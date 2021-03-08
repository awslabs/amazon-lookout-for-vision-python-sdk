# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
  
#       http://www.apache.org/licenses/LICENSE-2.0
  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import numpy as np
import boto3
from skimage import io, img_as_ubyte
from skimage.transform import rescale, resize
import math
import random
from multiprocessing.pool import ThreadPool


class Image():
    """Helper class to assist with image handling.

    This class can help you check your local image sizes, resize images
    check if they are compliant and help you upload local images
    into your S3 bucket.

    Attributes:
    s3              The Amazon S3 boto3 client.

    """

    def __init__(self):
        """Creates a Image helper object for Amazon Lookout for Vision.
        It will assist you in checking image sizes, complying with the
        Amazon Lookout for Vision image boundaries, rescaling and uploading
        local images to S3.

        Technical documentation on how Amazon Lookout for Vision works can be
        found at: https://aws.amazon.com/lookout-for-vision/

        Args:
            None

        """
        super(Image, self).__init__()
        self.s3 = boto3.client("s3")

    @classmethod
    def __image_size_checker(self, image_type="good", verbose=False):
        """Internal image size checker method.
        It will count the number of images that are compliant with
        Amazon Lookout for Vision.

        Args:
            image_type (str): can either be "good" or "bad".
                Make sure you have a local folders named "good"
                and "bad" with the respective images.
            verbose (bool): Decide to include image metadata in output.

        Returns:
            json: a JSON object with the meta information on the images.

        """
        # List all files and count the number of images.
        # Also set:
        # - internal counter 'cnt'
        # - minimal image size 'img_min'
        # - maximal image size 'img_max'
        # - image_info as the JSON object to collect image metadata
        files = os.listdir(image_type)
        no_of_images = len(files)
        cnt, img_min, img_max, image_info = 0, 64, 4096, {}
        # For each file...
        for file in files:
            # ...set the path and read the image in
            path = "{}/{}".format(image_type, file)
            if not (".jpeg" in path or ".jpg" in path or ".png" in path):
                print("The following image is not compliant: {}".format(path))
                continue
            image = io.imread(path)
            # Check the image min and max size and
            # log the shape
            min_size = min(image.shape[:2])
            max_size = max(image.shape[:2])
            image_info[path] = image.shape
            # If image complies cnt += 1
            if min_size >= img_min and max_size <= img_max:
                cnt += 1
            else:
                # if not log the smalles/largest image found
                img_min = min(img_min, min_size)
                img_max = max(img_max, max_size)
        # Check if all images are compliant
        compliant = True if cnt == no_of_images else False
        # Set output
        response = {
            "no_of_images": no_of_images,
            "compliant_images": cnt,
            "compliant": compliant
        }
        # Add some metadata if image is not compliant
        if not compliant:
            response["min_size"] = img_min
            response["max_size"] = img_max
        # Add more metadata
        if verbose:
            response["image_metadata"] = image_info
        return response

    @classmethod
    def __image_shape_checker(self, image_type="good", verbose=False):
        """Internal image shape checker method.
        This method can validate if all images have the same shape
        or if they need adjustment before Amazon Lookout for Vision
        can train a model.

        Args:
            image_type (str): can either be "good" or "bad".
                Make sure you have a local folders named "good"
                and "bad" with the respective images.
            verbose (bool): Decide to include image metadata in output.

        Returns:
            json: a JSON object with the meta information on the images.

        """
        # List all files and count the number of images.
        # Also set:
        # - internal counter 'cnt'
        # - placeholder for image shape 'shape'
        # - image_info as the JSON object to collect image metadata
        files = os.listdir(image_type)
        no_of_images = len(files)
        cnt, shape, image_info = 1, (0, 0, 0), {}
        value = 1e6
        # For each image...
        for file in files:
            # ...set path and read image
            path = "{}/{}".format(image_type, file)
            if not (".jpeg" in path or ".jpg" in path or ".png" in path):
                print("The following image is not compliant: {}".format(path))
                continue
            image = io.imread(path)
            # For the first image set 'shape' to the images's shape
            if shape == (0, 0, 0):
                shape = image.shape
            else:
                # Check how often images have the same shape
                if shape == image.shape:
                    cnt += 1
                else:
                    # If images found that are smaller than the other
                    # set the new shape value here (that way you can
                    # make sure to know what is the scaling factor to
                    # downscale images to):
                    minvalue = min(shape[:-1])
                    maxvalue = max(shape[:-1])
                    if shape[0] > shape[1] and maxvalue/minvalue < value:
                        value = maxvalue/minvalue
                        shape = image.shape
            # Log shape in output
            image_info[path] = image.shape
        # Set status if all images have the same size
        status = "Image sizes are equal!" if cnt == no_of_images else "Downsize images!"
        # Populate response
        response = {
            "no_of_images": no_of_images,
            "compliant": cnt,
            "status": status,
            "min_image_shape": shape
        }
        # Log more if asked for:
        if verbose:
            response["image_metadata"] = image_info
        return response

    @classmethod
    def __check_image_number(self, no_of_files):
        """This method will check the test images, if the total number overceed 5000.
        This is roughly based on the S3 API throttle limit:
        https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html

        Args:
            no_of_files (int): count of number of files

        Returns:
            json: a JSON object with the meta information on the images.

        """
        if no_of_files > 5000:
            print("Warning: You have too many images, this will slow down the upload process!")
            print("Images will be uploaded. If you want to speed up the process please use <5000 images.")
        return True

    def __upload_s3(self, filename, bucket, fname):
        """This method is used to do the upload to s3

        Args:
            filename (str): filname to upload
            bucket (str): S3 bucket
            fname (str): S3 object key

        Returns:
            None
        """
        self.s3.upload_file(
            Filename=filename,
            Bucket=bucket,
            Key=fname)

    @classmethod
    def __is_compliant(self, response):
        """Helper to return if all images are compliant.

        Args:
            response (json): Returned object from __image_size_checker

        Returns:
            bool: True or False (compliant or not compliant)

        """
        good = response["good"]["compliant"]
        bad = response["bad"]["compliant"]
        return good and bad

    @classmethod
    def __is_too_small(self, response):
        """Helper to return if any image used is too small.

        Args:
            response (json): Returned object from __image_size_checker

        Returns:
            bool: True or False

        """
        compliant = self.__is_compliant(response=response)
        if not compliant:
            return min(response["good"]["min_size"], response["bad"]["min_size"]) < 64
        return False

    @classmethod
    def __is_too_large(self, response):
        """Helper to return if any image used is too large.

        Args:
            response (json): Returned object from __image_size_checker

        Returns:
            bool: True or False

        """
        compliant = self.__is_compliant(response=response)
        if not compliant:
            return max(response["good"]["max_size"], response["bad"]["max_size"]) > 64
        return False

    def check_image_sizes(self, prefix="", verbose=False):
        """Check your image sizes with this function.

        Args:
            verbose (bool): Decide to include image metadata in output.

        Returns:
            json: Metadata on the image and their compliance.

        """
        # If you don't have folders "good" and "bad" this libary
        # will not work:
        folders = os.listdir()
        if "{}good".format(prefix) not in folders and "{}bad".format(prefix) not in folders:
            print(
                "Error: Methods requires folders good/ and bad/ with images in this location!")
            return {}
        # For each folder set a response
        response = {}
        for path in ["{}good".format(prefix), "{}bad".format(prefix)]:
            response[path] = self.__image_size_checker(
                image_type=path, verbose=verbose)
        return response

    def check_image_shapes(self, prefix="", verbose=False):
        """Check your image sahpes with this function.

        Args:
            verbose (bool): Decide to include image metadata in output.

        Returns:
            json: Metadata on the image and their compliance.

        """
        # If you don't have folders "good" and "bad" this libary
        # will not work:
        folders = os.listdir()
        if "{}good".format(prefix) not in folders and "{}bad".format(prefix) not in folders:
            print(
                "Error: Methods requires folders {}good/ and {}bad/ with images in this location!".format(prefix, prefix))
            return {}
        # For each folder set a response
        response = {}
        for path in ["{}good".format(prefix), "{}bad".format(prefix)]:
            response[path] = self.__image_shape_checker(
                image_type=path, verbose=verbose)
        good = response["{}good".format(prefix)]["min_image_shape"]
        bad = response["{}bad".format(prefix)]["min_image_shape"]
        if max(good[:-1])/min(good[:-1]) < max(bad[:-1])/min(bad[:-1]):
            response["shape_recommendation"] = good
        else:
            response["shape_recommendation"] = bad
        return response

    def rescale(self, prefix="rescaled_", force_rescale=False):
        """This method will rescale all your images to a common size.
        This function will automatically determine if your images comply
        and if they need rescaling. It will only rescale if your images
        are not of same size or violate the boundaries of Amazon Lookout for Vision.

        Args:
            prefix (str): The prefix for the new folder with your resized images.
                Note: if set to "" this function will overwrite your original images.
            force_rescale (bool): force to rescale even for compliant images.

        Returns:
            json: Object with the new local image paths.

        """
        # Factor = 1 means images need no rescaling
        # Also, if the function returns the default output all images are
        # good as they are:
        factor = 1
        output = {
            "{}good".format(prefix): "Ok",
            "{}bad".format(prefix): "Ok"
        }
        # If you don't have folders "good" and "bad" this libary
        # will not work:
        folders = os.listdir()
        if "good" not in folders and "bad" not in folders:
            print(
                "Error: Methods requires folders good/ and bad/ with images in this location!")
            return {}
        shapes = self.check_image_shapes()
        shape_rec = shapes["shape_recommendation"]
        if max(shape_rec) > 4096 or min(shape_rec[:-1]) < 64:
            print("Warning: Your images shapes imply that rescaling will lead to a lot of information loss! Please check if you can collect better quality images!")
        compliant = self.__is_compliant(response=shapes)
        if compliant and not force_rescale:
            print("No rescaling needed!")
        else:
            # ...take each image and...
            for path in ["good", "bad"]:
                # ...create a new directory using the prefix
                os.mkdir("{}{}".format(prefix, path))
                files = os.listdir(path)
                # For each file...
                for file in files:
                    # ...load the image, resize it and save it back
                    # to the new folder
                    img_path = "{}/{}".format(path, file)
                    if not (".jpeg" in img_path or ".jpg" in img_path or ".png" in img_path):
                        print(
                            "The following image is not compliant: {}".format(img_path))
                        continue
                    image = io.imread(img_path)
                    rescaled = resize(image, shape_rec)
                    rescaled = img_as_ubyte(image=rescaled)
                    fname = "{}{}/{}".format(prefix, path, file)
                    io.imsave(fname=fname, arr=rescaled)
                # Log new image paths in output object
                output[path] = "{}{}".format(prefix, path)
        return output

    def upload_from_local(self, bucket, train_and_test=True, test_split=0.2, prefix="",
                          content_type="image/jpeg", processes_num=10):
        """This method will help you upload your local images to S3.
        Based on your folders "good" and "bad" - that are mandatory - it will
        upload the images to S3 accordingly.

        Args:
            bucket (str): The S3 bucket name
            train_and_test (bool): Have both present train and validation data
                Note: You can decide here wheter your Amazon Lookout for Vision
                project will use a train and validation set.
            test_split (float): The percentage to split between train and validation
            prefix (str): The prefix used for your "good" and "bad" folders.
                If you rescaled your new folders might be named "rescaled_good" and
                "rescaled_bad"
            content_type (str): The image type. Valid are "image/jpeg" and "image/png"

        Returns:
            json: Object with the S3 locations of the data.
        """ 
        # If you don't have folders "good" and "bad" this libary
        # will not work:
        folders = os.listdir()      
        if "{}good".format(prefix) not in folders and "{}bad".format(prefix) not in folders:
            print(
                "Error: Methods requires folders good/ and bad/ with images in this location!")
            return {}
        # For both folders "good" and "bad"...
        for p in ["{}good".format(prefix), "{}bad".format(prefix)]:
            files = os.listdir(path=p)
            # check if there are more than 5000 images in each "good" and "bad" folder, if so, exit
            check = self.__check_image_number(no_of_files=len(files))
            # shuffle and split the images into training and test set
            random.shuffle(files)
            training_set = files[:int(len(files)*(1-test_split))] 
            validation_set = files[-int(len(files)*test_split):]
            batches = [files]
            if train_and_test:
                batches = [training_set, validation_set]
                if "good" in p and len(training_set) < 10:
                    print("Warning: You don't have enough good images in your training dataset available for training a model!")
                    print("""
                        From the documentation:
                        A single dataset project needs at least 20 images labeled as normal and at least 10 images labeled as anomalous.
                        A project with separate training and test datasets, needs a training dataset with at least 10 images labeled as normal. The test dataset needs at least 10 images labeled as normal and at least 10 images labeled as anomalous.
                        We recommend that you add more than the minimum number of labeled images. Unlabeled images aren't used to train your model.
                    """)
                if "good" in p and len(validation_set) < 10:
                    print("Warning: You don't have enough good images in your validation dataset available for training a model!")
                    print("""
                        From the documentation:
                        A single dataset project needs at least 20 images labeled as normal and at least 10 images labeled as anomalous.
                        A project with separate training and test datasets, needs a training dataset with at least 10 images labeled as normal. The test dataset needs at least 10 images labeled as normal and at least 10 images labeled as anomalous.
                        We recommend that you add more than the minimum number of labeled images. Unlabeled images aren't used to train your model.
                    """)
                if "bad" in p and len(validation_set) < 10:
                    print("Warning: You don't have enough bad images in your validation dataset available for training a model!")
                    print("""
                        From the documentation:
                        A single dataset project needs at least 20 images labeled as normal and at least 10 images labeled as anomalous.
                        A project with separate training and test datasets, needs a training dataset with at least 10 images labeled as normal. The test dataset needs at least 10 images labeled as normal and at least 10 images labeled as anomalous.
                        We recommend that you add more than the minimum number of labeled images. Unlabeled images aren't used to train your model.
                    """)
            else:
                if "good" in p and len(files) < 20:
                    print("Warning: You don't have enough good images in your training dataset available for training a model!")
                    print("""
                        From the documentation:
                        A single dataset project needs at least 20 images labeled as normal and at least 10 images labeled as anomalous.
                        A project with separate training and test datasets, needs a training dataset with at least 10 images labeled as normal. The test dataset needs at least 10 images labeled as normal and at least 10 images labeled as anomalous.
                        We recommend that you add more than the minimum number of labeled images. Unlabeled images aren't used to train your model.
                    """)
                if "bad" in p and len(files) < 10:
                    print("Warning: You don't have enough bad images in your training dataset available for training a model!")
                    print("""
                        From the documentation:
                        A single dataset project needs at least 20 images labeled as normal and at least 10 images labeled as anomalous.
                        A project with separate training and test datasets, needs a training dataset with at least 10 images labeled as normal. The test dataset needs at least 10 images labeled as normal and at least 10 images labeled as anomalous.
                        We recommend that you add more than the minimum number of labeled images. Unlabeled images aren't used to train your model.
                    """)
            for batch_files in batches:
                subfolder = "normal"
                if "bad" in p:
                    subfolder = "anomaly"
                folder = "training"
                if batch_files == validation_set:
                    folder = "validation"
                # generate the input for batch uploading to s3 bucket
                batch_upload_input = []
                for file in batch_files:
                    filename = '{}/{}'.format(p, file)
                    if not (".jpeg" in file or ".jpg" in file or ".png" in file):
                        print("The following file was skipped: {}".format(file))
                        continue
                    input_tuple = (filename, bucket, "{}/{}/{}".format(folder, subfolder, file))
                    batch_upload_input.append(input_tuple)
                
                pool = ThreadPool(processes=processes_num)
                pool.starmap(self.__upload_s3, batch_upload_input)
        # Set metadata in response:
        response = {
            "train": {
                "normal": "s3://{}/training/normal/".format(bucket),
                "anomaly": "s3://{}/training/anomaly/".format(bucket)
            }
        }
        if train_and_test:
            response["test"] = {
                "normal": "s3://{}/validation/normal/".format(bucket),
                "anomaly": "s3://{}/validation/anomaly/".format(bucket)
            }
        return response

    def copy_from_s3(self, input_bucket, output_bucket, prefix_good="good", prefix_bad="bad",
                     train_and_test=True, test_split=0.2):
        """This method will help you copy images within S3.
        Based on your keys prefix_good and prefix_bad it will
        upload the images to S3 accordingly.

        Args:
            input_bucket (str): The S3 bucket name for input
            output_bucket (str): The S3 bucket name for output
            prefix_good (str): Path in S3 to good images
            prefix_bad (str): Path in S3 to bad images
            train_and_test (bool): Have both present train and validation data
                Note: You can decide here wheter your Amazon Lookout for Vision
                project will use a train and validation set.
            test_split (float): The percentage to split between train and validation

        Returns:
            json: Object with the S3 locations of the data.

        """
        s3 = boto3.client('s3')
        # Create a reusable Paginator
        paginator = s3.get_paginator('list_objects_v2')
        cnt = 0

        for p in [prefix_good, prefix_bad]:
            # Create a PageIterator from the Paginator
            page_iterator = paginator.paginate(Bucket=input_bucket, Prefix=p)
            # Paginate through S3
            for page in page_iterator:
                # For each key
                for key in page["Contents"]:
                    if key["Key"][-1] == "/":
                        continue
                    # only consider allowed images
                    if not (".jpeg" in key["Key"] or ".jpg" in key["Key"] or ".png" in key["Key"]):
                        print(
                            "The following file was skipped: {}".format(key["Key"]))
                        continue
                    # Then set the filename and depending on
                    # which folder you are in distinguish between
                    # normal ("good") and anomaly ("bad"):
                    subfolder = "normal"
                    if p == prefix_bad:
                        subfolder = "anomaly"
                    # Make a train-test-split:
                    choice = np.random.choice(a=[0, 1], size=1, replace=True, p=[
                                              1-test_split, test_split])[0]
                    folder = "training"
                    if choice == 1 and train_and_test:
                        folder = "validation"
                    # Set S3 object name and copy file:
                    fname = "{}/{}/{}".format(folder,
                                              subfolder, key["Key"].split("/")[-1])
                    
                    try:
                        
                        response = s3.copy({
                            "Bucket": input_bucket,
                            "Key": key["Key"]
                        }, output_bucket, fname)
                        cnt += 1
                    except Exception as e:
                        print("Warning: Copy did not work!")
                        print("With error message: {}".format(e))
        # Set metadata in response:
        response = {
            "status": "Success!",
            "objects_copied": cnt,
            "train": {
                "normal": "s3://{}/training/normal/".format(output_bucket),
                "anomaly": "s3://{}/training/anomaly/".format(output_bucket)
            }
        }
        if train_and_test:
            response["test"] = {
                "normal": "s3://{}/validation/normal/".format(output_bucket),
                "anomaly": "s3://{}/validation/anomaly/".format(output_bucket)
            }
        return response