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

from datetime import datetime
import logging
import json
import boto3


class Manifest():
    """Helper class to assist with handling and creating manifest files.

    This class will assist you in generating your manifests. It also
    has a helper to push the manifest file to the correct location.

    Attributes:
    bucket          The name of the S3 bucket.
    datasets        The datasets used for the manifest (train and/or test)
    s3              The Amazon S3 boto3 client.

    """

    def __init__(self, bucket, datasets=["training"]):
        """Creates a Manifest helper object for Amazon Lookout for Vision.
        It can create a manifest file based on the data you stored on S3
        for your Amazon Lookout for Vision project. It can also push that
        artifact file to the correct place to get started with model training.

        Technical documentation on how Amazon Lookout for Vision works can be
        found at: https://aws.amazon.com/lookout-for-vision/

        Args:
            bucket (str): Name of the Amazon S3 bucket with your images.
            dataset (list): A list that needs at least "training" be present.
                Other option is "test".

        """
        super(Manifest, self).__init__()
        self.bucket = bucket
        self.datasets = datasets
        self.s3 = boto3.client("s3")

    def get_bucket(self):
        """Getter for bucket name.

        Args:
            None

        Returns:
            str: a S3 bucket name

        """
        return self.bucket

    def get_datasets(self):
        """Getter for datasets list.

        Args:
            None

        Returns:
            list: a list with the datasets to be used.

        """
        return self.datasets

    def generate_manifests(self):
        """Generates manifest file(s).
        It generates a manifest file for each option: training and test. If
        only training is specified for *datasets* then this function will only
        generate a manifest file for the training data stored in S3.

        Args:
            None

        Returns:
            json: a JSON object with one entry per image.

        """
        # Set variables:
        manifests = {}
        now = datetime.now()
        dttm = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
        # For each dataset (namely training and/or test)...
        for ds in self.datasets:
            # ...add an empty JSON object to the manifests and...
            manifests[ds] = {}
            # ...for each folder of a dataset (needs to be 'normal' and 'anomaly')...
            for folder in ["normal", "anomaly"]:
                # ...list the objects with the prefix
                # e.g training/anomaly
                objects = self.s3.list_objects_v2(
                    Bucket=self.bucket, Prefix="{}/{}".format(ds, folder))["Contents"]
                # Manifest files assume labels to be:
                # 1 = good image
                # 0 = bad image
                label = 1
                if folder == "anomaly":
                    label = 0
                # Each manifest will be represented as a string:
                manifest = ""
                # For each file from the S3 objects...
                for file in objects:
                    # ...extract the filename and...
                    f = file["Key"].split("/")[-1]
                    # ...set the manifest object as per:
                    manifest_obj = {
                        "source-ref": "s3://{}/{}/{}/{}".format(self.bucket, ds, folder, f),
                        "auto-label": label,
                        "auto-label-metadata": {
                            "confidence": 1,
                            "job-name": "labeling-job/auto-label",
                            "class-name": folder,
                            "human-annotated": "yes",
                            "creation-date": dttm,
                            "type": "groundtruth/image-classification"
                        }
                    }
                    # Add a dump of each object to the manifest
                    manifest += (json.dumps(manifest_obj)+"\n")
                # Once a manifest is created add it to all manifests:
                manifests[ds][folder] = manifest
        return manifests

    def push_manifests(self):
        """Generates and push manifest file(s).
        It will use the generate_manifests(...) method to generate the
        manifest files and push them to S3.

        Args:
            None

        Returns:
            json: a JSON object either empty (failure) or the S3 locations
                of the manifest files

        """
        # Generate manifest files
        manifests = self.generate_manifests()
        success = {}
        # Try...
        try:
            # for each dataset (namely training and/or test)...
            for ds in self.datasets:
                # ...each manifest will be represented as a string
                manifest = ""
                for key in manifests[ds]:
                    # Add a dump of each object to the manifest
                    manifest += manifests[ds][key]
                # Upload the manifest to S3
                upload = self.s3.put_object(
                    Bucket=self.bucket, Key="{}.manifest".format(ds), Body=manifest)
                # Store location
                success[ds] = {
                    "bucket": self.bucket,
                    "key": "{}.manifest".format(ds),
                    "location": "s3://{}/{}.manifest".format(self.bucket, ds)}
        except Exception as e:
            # Log error!
            logging.error(e)
        return success