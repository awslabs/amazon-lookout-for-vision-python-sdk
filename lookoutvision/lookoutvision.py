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
import boto3
from collections import defaultdict
import inspect
import json
import logging
import time


class LookoutForVision():
    """LookoutForVision class to build, train and deploy.

    This class helps to build, train and deploy a Amazon Lookout for Vision
    project. It implements the three most common methods for model deployment:
    # - .fit()
    # - .deploy()
    # - .predict()

    Attributes:
    project_name    The name of the Amazon Lookout for Vision project.
    lv              The Amazon Lookout for Vision boto3 client.
    model_version   The (initial) model version.

    """

    def __init__(self, project_name, model_version="1"):
        """Build, train and deploy Amazon Lookout for Vision models.

        Technical documentation on how Amazon Lookout for Vision works can be
        found at: https://aws.amazon.com/lookout-for-vision/

        Args:
            project_name (str): Name of the Amazon Lookout for Vision to interact with.
            model_version (str): The (initial) model version.

        """
        #super(LookoutForVision, self).__init__()
        self.project_name = project_name
        self.lv = boto3.client("lookoutvision")
        self.s3 = boto3.client("s3")
        self.model_version = model_version
        self.describe_project()

    @classmethod
    def _get_param_names(self):
        """Internal get parameter names helper.
        It will retrieve all the parameters used within your class.

        Args:
            None

        Returns:
            list: all class parameters

        """
        init = getattr(self.__init__, 'deprecated_original', self.__init__)
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters])

    def describe_project(self):
        """Describe a project.

        Args:
            None

        Returns:
            json: The project details

        """

        project = {}
        # First try to describe the project given by the name:
        try:
            project = self.lv.describe_project(
                ProjectName=self.project_name)["ProjectDescription"]["ProjectArn"]
            print('Project already exists with arn: ' + project)
        except Exception as e:
            if 'ResourceNotFoundException' in str(e):
                print(f"Project {self.project_name} does not exist yet...use the create_project() method to set up your first project")
            else:
                raise Exception
        return project

    def create_project(self):
        """Create a project.

        Args:
            None

        Returns:
            json: The project details

        """
        project = {}
        # First try to create a new project:
        try:
            project = self.lv.create_project(
                ProjectName=self.project_name
            )["ProjectMetadata"]["ProjectArn"]
            print(f"Creating the project: {self.project_name}")
        except Exception as e:
            if 'ConflictException' in str(e):
                project = self.lv.describe_project(
                    ProjectName=self.project_name
                )["ProjectDescription"]["ProjectArn"]
            else:
                raise Exception
        return project

    def get_params(self, deep=True):
        """Get class parameters.

        Args:
            deep (bool): Make a deep copy of parameters for output.

        Returns:
            json: an object with the internal parameters and their values.

        """
        output = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                output.update((key + '__' + i, val) for i, val in deep_items)
            output[key] = value
        return output

    def set_params(self, **params):
        """Set class parameters.

        Args:
            **params (dict): New parameters in key=value format.

        Returns:
            self: the class itself

        """
        if not params:
            return self
        valid = self.get_params(deep=True)

        nested = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub = key.partition('__')
            if key not in valid:
                raise ValueError('Invalid parameter %s for class %s. '
                                 'Check the list of available parameters '
                                 'with `cls.get_params().keys()`.' %
                                 (key, self))
            if delim:
                nested[key][sub] = value
            else:
                setattr(self, key, value)
                valid[key] = value

        for key, sub_params in nested.items():
            valid[key].set_params(**sub_params)

        return self

    def update_datasets(self, dataset_type, wait=True):
        """Create a dataset.

        Args:
            dataset_type (dict): A setting from where to get input data from
                Format of this input is of type:
                "train": {
                    "bucket": "my_s3_bucket",
                    "key": "training.manifest",
                    "version": "1"
                },
                "test": {
                    "bucket": "my_s3_bucket",
                    "key": "validation.manifest",
                    "version": "1"
                }
            wait (bool): Either to wait in the console uppon succes or escape function

        Returns:
            json: an object with metadata on success

        """

        datasets = {}
        # For each dataset used...
        for key in dataset_type:
            # ...create a dataset
            d_type = "train" if (
                key == "training" or key == "train") else "test"
            try:
                dataset = self.lv.delete_dataset(
                    ProjectName=self.project_name,
                    DatasetType=d_type
                )
            except Exception as e:
                print("Error in dataset deletion with exception: {}".format(e))
                print("Please check CloudWatch logs for troubleshooting!")
        return self.create_datasets(dataset_type=dataset_type, wait=wait)

    def create_datasets(self, dataset_type, wait=True):
        """Create a dataset.

        Args:
            dataset_type (dict): A setting from where to get input data from
                Format of this input is of type:
                "train": {
                    "bucket": "my_s3_bucket",
                    "key": "training.manifest",
                    "version": "1"
                },
                "test": {
                    "bucket": "my_s3_bucket",
                    "key": "validation.manifest",
                    "version": "1"
                }
            wait (bool): Either to wait in the console uppon succes or escape function

        Returns:
            json: an object with metadata on success

        """
        datasets = {}
        # For each dataset used...
        for key in dataset_type:
            # ...create a dataset
            d_type = "train" if (
                key == "training" or key == "train") else "test"
            try:
                dataset = self.lv.create_dataset(
                    ProjectName=self.project_name,
                    DatasetType=d_type,
                    DatasetSource={
                        'GroundTruthManifest': {
                            'S3Object': {
                                'Bucket': dataset_type[key]["bucket"],
                                'Key': dataset_type[key]["key"],
                                'VersionId': self.model_version
                            }
                        }
                    }
                )["DatasetMetadata"]
                # Log output
                datasets[key] = dataset
            except Exception as e:
                if 'ConflictException' in str(e):
                    print("Dataset already existed in the project")
                    print(
                        "If the dataset already exists try updating it with: update_datasets")
                else:    
                    print("Error in create_datasets with exception: {}".format(e))
                    raise Exception
                return datasets
        # Notify user when creation is done:
        print("Creating dataset(s):", end=" ")
        if wait:
            status = "CREATING"
            while status != "CREATE_COMPLETE":
                cnt = 0
                for key in dataset_type:
                    d_type = "train" if (
                        key == "training" or key == "train") else "test"
                    stat = self.lv.describe_dataset(
                        ProjectName=self.project_name,
                        DatasetType=d_type
                    )["DatasetDescription"]["Status"]
                    if stat == "CREATE_COMPLETE":
                        cnt += 1
                        datasets[key]["Status"] = "CREATE_COMPLETE"
                if cnt == len(dataset_type):
                    status = "CREATE_COMPLETE"
                print("-", end="")
                time.sleep(5)
            print("!")
        for key in datasets:
            datasets[key]["StatusMessage"] = "Dataset created."
        return datasets

    def fit(self, output_bucket, model_prefix=None, train_and_test=True, wait=True):
        """Train the model.
        Create a model from the datasets
        At first check whether the minimum no of images are available to train the model.
        There should be min 20 normal and 10 anomalous images in training/train dataset.

        Args:
            output_bucket (str): The output S3 bucket to be used for model logging.
            model_prefix (str): Optional to add a model prefix name for logging.
            train_and_test (bool): Whether to us train or train and test set
            wait (bool): Either to wait in the console uppon succes or escape function

        Returns:
            json: an object with metadata on success

        """

        test_dataset = {"Status": "No test dataset used!"}
        if train_and_test:
            test_dataset = self.lv.describe_dataset(
                ProjectName=self.project_name,
                DatasetType="test"
            )["DatasetDescription"]
        train_dataset = self.lv.describe_dataset(
            ProjectName=self.project_name,
            DatasetType="train"
        )["DatasetDescription"]
        normal_no_images_train = train_dataset["ImageStats"]["Normal"]
        anomaly_no_images_train = train_dataset["ImageStats"]["Anomaly"]

        if ((normal_no_images_train >= 20 and anomaly_no_images_train >= 10)):
            model = self.lv.create_model(
                ProjectName=self.project_name,
                OutputConfig={
                    'S3Location': {
                        'Bucket': output_bucket,
                        'Prefix': model_prefix if model_prefix is not None else ""
                    }
                })["ModelMetadata"]
            if wait:
                print("Model training started:", end=" ")
                version = model["ModelVersion"]
                status = model["Status"]
                while status == "TRAINING":
                    update = self.lv.describe_model(
                        ProjectName=self.project_name,
                        ModelVersion=version
                    )["ModelDescription"]
                    status = update["Status"]
                    print("-", end="")
                    time.sleep(60)
                print("!")
            else:
                print("""Model is being created. Training will take a while.\n
                         Please check your Management Console on progress.\n
                         You can continue with deployment once the model is trained.\n
                      """)
            # Return success
            return {
                "status": "Success!",
                "project": self.project_name,
                "train_datasets": train_dataset,
                "test_datasets": test_dataset,
                "model": model
            }
        else:
            print("""Number of images is not sufficient, at least 20 normal and 10 anomaly\n
                     imgages are required for training images
                  """)
            return {
                "status": "Failure!",
                "project": self.project_name,
                "train_datasets": train_dataset,
                "test_datasets": test_dataset,
                "model": None
            }

    def deploy(self, min_inf_units=1, model_version=None, wait=True):
        """Deploy your model.

        Args:
            min_inf_units (int): Minimal number of inference units.
            model_version (str): The model version to deploy.
            wait (bool): Either to wait in the console uppon succes or escape function

        Returns:
            json: an object with metadata on success

        """
        # Check the current status of the model
        current_status = self.lv.describe_model(
            ProjectName= self.project_name,
            ModelVersion=self.model_version if model_version is None else model_version
        )['ModelDescription']['Status']
        # If model status is TRAINED , then only start the model. 
        # otherwise print the message with actual status that it can not be started
        if (current_status != 'TRAINED'):
            print('current model with version {} is in the status {}, hence it can not be started/hosted. The model needs to be in TRAINED status to be started/hosted'.format(self.model_version if model_version is None else model_version,current_status))
        else:
            # Start the model either using the internal model version
            # or use the one supplied to this method:    
            status = self.lv.start_model(
                ProjectName=self.project_name,
                ModelVersion=self.model_version if model_version is None else model_version,
                MinInferenceUnits=min_inf_units
            )["Status"]
            # Wait until model is trained:
            print("Model will be hosted now")
            if wait:
                while status != "HOSTED":
                    status = self.lv.describe_model(
                        ProjectName=self.project_name,
                        ModelVersion=self.model_version if model_version is None else model_version
                    )["ModelDescription"]["Status"]
                    print("-", end="")
                    time.sleep(60)
                print("!")
                print("Your model is now hosted!")
            # Return success:
            return {
                "status": status,
                "model": self.lv.describe_model(
                    ProjectName=self.project_name,
                    ModelVersion=self.model_version if model_version is None else model_version
                )["ModelDescription"]
            }
        
    def predict(self, model_version=None, local_file="",
                bucket="", key="", content_type="image/jpeg"):
        """Predict using your Amazon Lookout for Vision model.
        You can either predict from S3 object or local images.

        Args:
            model_version (str): The model version to deploy.
            local_file (str): Path to local image.
            bucket (str): S3 bucket name.
            key (str): Object in S3 bucket.
            content_type (str): Either "image/jpeg" or "image/png".

        Returns:
            json: an object with results of prediction

        """
        # If no paths are set return warning:
        if local_file == "" and bucket == "" and key == "":
            print("Warning: either local_file OR bucket & key need to be present!")
            return {'Source': {'Type': 'warning'}, 'IsAnomalous': None, 'Confidence': -1.0}
        # If paths for local file AND S3 bucket are set return another warning:
        if local_file != "" and bucket != "" and key != "":
            print("Warning: either local_file OR bucket & key need to be present!")
            return {'Source': {'Type': 'warning'}, 'IsAnomalous': None, 'Confidence': -1.0}
        # If method is used properly then...
        obj = None
        if local_file != "":
            # ...set obj to bytearray using local image...
            try:
                with open(local_file, "rb") as image:
                    f = image.read()
                    obj = bytearray(f)
            except IsADirectoryError as e:
                print("Warning: you specified a directory, instead of a single file path")
                print("Maybe you would like to try \'_batch_predict_local\' or \'batch_predict_s3\' method!")
                return
        elif bucket != "" and key != "":
            # ...or a byte object by pulling from S3
            obj = boto3.client("s3").get_object(
                Bucket=bucket,
                Key=key)["Body"].read()
        else:
            # If file not found:
            print("Warning: No file found!")
            return {'Source': {'Type': 'warning'}, 'IsAnomalous': None, 'Confidence': -1.0}
        # Predict using your model:
        result = self.lv.detect_anomalies(
            ProjectName=self.project_name,
            ModelVersion=self.model_version if model_version is None else model_version,
            Body=obj,
            ContentType=content_type
        )["DetectAnomalyResult"]
        return result

    def _batch_predict_local(self, local_path, model_version=None, content_type="image/jpeg"):
        """Predict for all the images using your Amazon Lookout for Vision model 
        from S3 objects.

        Args:
            local_path (str): Path to local images.
            model_version (str): The model version to deploy.
            content_type (str): Either "image/jpeg" or "image/png".

        Returns:
            json: s3 objects with location which stores the results of prediction

        """
        predictions = []
        files = os.listdir(local_path)
        for file in files:
            filename = '{}/{}'.format(local_path, file)
            # ...set obj to bytearray using local image...
            with open(filename, "rb") as image:
                f = image.read()
                obj = bytearray(f)
                try:
                    # Predict using your model:
                    result = self.lv.detect_anomalies(
                        ProjectName=self.project_name,
                        ModelVersion=self.model_version if model_version is None else model_version,
                        Body=obj,
                        ContentType=content_type
                    )["DetectAnomalyResult"]
                    predictions.append(result)
                except Exception as e:
                    print("Warning: prediction failed for file: {}".format(filename))
                    print("with error message: {}".format(e))
        return {'status': 'Success!', 'predicted_result': predictions}

    def batch_predict(self, model_version=None, local_path=None, input_bucket="", input_prefix=None,
                      output_bucket="", output_prefix=None, content_type="image/jpeg"):
        """Predict for all the images using your Amazon Lookout for Vision model 
        from S3 objects.

        Args:
            model_version (str): The model version to deploy.
            local_path (str): Path to local images.
            input_bucket (str): S3 bucket name for input images.
            input_prefix(str): S3 folder names ( if any) for input bucket location
            output_bucket (str): S3 bucket name to store predicted results.
            output_prefix(str): S3 folder names ( if any) for output bucket location to store predicted results
            content_type (str): Either "image/jpeg" or "image/png".

        Returns:
            json: s3 objects with location which stores the results of prediction

        """
        if local_path is not None:
            return self._batch_predict_local(local_path=local_path,
                                             model_version=model_version, content_type=content_type)
            # If no input bucket is set return warning:
        if input_bucket == "":
            print("Warning: S3 bucket need to be present for input images!")
            return {'status': 'Error!', 'predicted_result': None}
        # If no output bucket is set return warning:
        if output_bucket == "":
            print("Warning: S3 bucket need to be present to load prediction result!")
            return {'status': 'Error!', 'predicted_result': None}
        obj = None
        if input_bucket != "":
            
            success = {}
            kwargs = {'Bucket': input_bucket}
            if (input_prefix != None):
                if isinstance(input_prefix, str):
                    kwargs['Prefix'] = input_prefix
                else:
                    kwargs['Prefix'] = str(input_prefix)
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(**kwargs)

            # Try...
            try:
            
                for page in pages:
                    for obj in page['Contents']:           
                
                        key = obj['Key']
                        if key[-1] == "/":
                            continue
                    
                        body = self.s3.get_object(Bucket=input_bucket, Key=key)[
                            "Body"].read()
                        file_name = key.split('/')[-1]
                    # Predict using your model:
                        result = self.lv.detect_anomalies(
                            ProjectName=self.project_name,
                            ModelVersion=self.model_version if model_version is None else model_version,
                            Body=body,
                            ContentType=content_type)["DetectAnomalyResult"]
                    # Upload the manifest to S3
                        upload = self.s3.put_object(Bucket=output_bucket, Key=output_prefix+"{}.json".format(
                            file_name), Body=json.dumps(result), ServerSideEncryption='AES256')
                        success = {"output_bucket": output_bucket,
                                   "prdected_file_key": output_prefix+"{}.json".format(file_name)
                               
                                   }
                        print('Predicted output is uploaded to s3 :' +
                              json.dumps(success))
                    
            except Exception as e:
                logging.error(e)
                print('Key object corresponding to error :' + key)
                       
                        
            return {
                'status': 'Success!',
                'predicted_result': "s3://{}/{}".format(output_bucket, output_prefix)
            }

        else:
            # If file not found:
            print("Warning: No file found!")
            return {'status': 'Error!', 'predicted_result': None}

    def stop_model(self, model_version=None, wait=True):
        """Stop deployed model.

        Args:
            model_version (str): The model version to deploy.

        Returns:
            json: an object with results of prediction

        """
        response = {}
        try:
            # Stop the model
            ModelVersion = self.model_version if model_version is None else model_version
            print('Stopping model version ' + ModelVersion +
                  ' for project ' + self.project_name)
            response = self.lv.stop_model(ProjectName=self.project_name,
                                          ModelVersion=ModelVersion)
            status = response["Status"]
            print("Model will be stopped now")
            if wait:
                while status != "TRAINED":
                    status = self.lv.describe_model(
                        ProjectName=self.project_name,
                        ModelVersion=ModelVersion
                    )["ModelDescription"]["Status"]
                    print("-", end="")
                    time.sleep(5)
                print("!")
                print("Your model is now stopped!")
            print('Status: ' + response['Status'])
        except Exception as e:
            response["Error"] = e
            print("Something went wrong: ", e)
        return response
