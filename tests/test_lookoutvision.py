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

import pandas as pd

from lookoutvision import lookoutvision

PROJECT_NAME = "myproject"
MODEL_VERSION = "1"
BUCKET_NAME = "mybucket"
DATASET_TYPE = ["training", "validation"]
TARGET_BUCKET_BATCH_PREDICTION = "mytargetdbucket"
TARGET_BUCKET_PREFIX_BATCH_PREDICTION = "prdictedresult/"


class DummyFramework(lookoutvision.LookoutForVision):
    """docstring for DummyFramework"""

    def __init__(self, project_name, model_version):
        self.project_name = project_name
        self.model_version = model_version

    def get_params(self):
        return {
            "project_name": self.project_name,
            "model_version": self.model_version
        }

    def set_params(self):
        return self

    def create_project(self):
        return {
            'status': 'Success!',
            'project': 'arn:aws:lookoutvision:REGION:123456789:project/myproject'
        }

    def describe_project(self):
        return {
            'status': 'Success!',
            'project': 'arn:aws:lookoutvision:REGION:123456789:project/myproject'
        }

    def create_datasets(self, dataset_type):
        return {
            'datasets': {
                'train': {
                    'DatasetType': 'train',
                    'CreationTimestamp': '1970-01-01 00:00:00.0',
                    'Status': 'CREATE_IN_PROGRESS',
                    'StatusMessage': 'Dataset created.'
                },
                'test': {'DatasetType': 'test',
                         'CreationTimestamp': '1970-01-01 00:00:00.0',
                         'Status': 'CREATE_IN_PROGRESS',
                         'StatusMessage': 'Dataset created.'
                         }
            }
        }

    def fit(self, output_bucket, model_prefix=None):
        return {

            'model': {
                'CreationTimestamp': '1970-01-01 00:00:00.0',
                'ModelVersion': '1',
                'ModelArn': 'arn:aws:lookoutvision:REGION:123456789:model/{}/{}'.format(PROJECT_NAME, MODEL_VERSION),
                'Status': 'TRAINING',
                'StatusMessage': 'The model is being trained.'
            }
        }

    def deploy(self, min_inf_units=1, model_version=None):
        return {
            'status': 'HOSTED',
            'model': {
                'ModelVersion': '1',
                'ModelArn': 'arn:aws:lookoutvision:REGION:123456789:model/{}/{}'.format(PROJECT_NAME, MODEL_VERSION),
                'CreationTimestamp': '1970-01-01 00:00:00.0',
                'Status': 'HOSTED',
                'StatusMessage': 'The model is running',
                'Performance': {
                    'F1Score': 1.0,
                    'Recall': 1.0,
                    'Precision': 1.0
                },
                'OutputConfig': {
                    'S3Location': {
                        'Bucket': '{}'.format(BUCKET_NAME),
                        'Prefix': ''
                    }
                },
                'EvaluationManifest': {
                    'Bucket': '{}'.format(BUCKET_NAME),
                    'Key': 'EvaluationManifest-{}-1.json'.format(PROJECT_NAME)
                },
                'EvaluationResult': {
                    'Bucket': '{}'.format(BUCKET_NAME),
                    'Key': 'EvaluationResult-{}-1.json'.format(PROJECT_NAME)
                },
                'EvaluationEndTimestamp': '1970-01-01 00:00:00.0'
            }
        }

    def predict(self, model_version=None, local_file="", bucket="", key="", content_type="image/jpeg"):
        return {
            'Source': {
                'Type': 'direct'
            },
            'IsAnomalous': True,
            'Confidence': 1.0
        }

    def batch_predict(self, model_version=None, input_bucket="", input_prefix=None, output_bucket="",
                      output_prefix=None, content_type="image/jpeg"):
        return {
            'status': 'Success!',
            'predicted_result': 's3://{}/{}'.format(TARGET_BUCKET_BATCH_PREDICTION,
                                                    TARGET_BUCKET_PREFIX_BATCH_PREDICTION)
        }

    def stop_model(self, model_version=None):
        return {
            "status": "STOPPING_HOSTING"
        }

    def delete_lookoutvision_project(self, project_name: str):
        return {'Success': True}

    def train_one_fold(self, input_bucket: str, output_bucket: str, s3_path: str, model_prefix: str, i_split: int,
                       delete_kfold_projects: bool = True):
        return {

            'model': {
                'CreationTimestamp': '1970-01-01 00:00:00.0',
                'ModelVersion': '1',
                'ModelArn': 'arn:aws:lookoutvision:REGION:123456789:model/{}/{}'.format(PROJECT_NAME, MODEL_VERSION),
                'Status': 'TRAINING',
                'StatusMessage': 'The model is being trained.'
            }
        }

    def train_k_fold(self):
        return pd.DataFrame()


def test_get_params():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.get_params() == {'model_version': '1',
                                'project_name': 'myproject'}


def test_set_params():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.set_params() == l4v


def test_create_project():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.create_project() == {
        'status': 'Success!',
        'project': 'arn:aws:lookoutvision:REGION:123456789:project/myproject'
    }


def test_describe_project():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.describe_project() == {
        'status': 'Success!',
        'project': 'arn:aws:lookoutvision:REGION:123456789:project/myproject'
    }


def test_create_datasets():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.create_datasets(dataset_type=DATASET_TYPE) == {
        'datasets': {
            'train': {
                'DatasetType': 'train',
                'CreationTimestamp': '1970-01-01 00:00:00.0',
                'Status': 'CREATE_IN_PROGRESS',
                'StatusMessage': 'Dataset created.'
            },
            'test': {'DatasetType': 'test',
                     'CreationTimestamp': '1970-01-01 00:00:00.0',
                     'Status': 'CREATE_IN_PROGRESS',
                     'StatusMessage': 'Dataset created.'
                     }
        }
    }


def test_fit():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.fit(output_bucket=BUCKET_NAME) == {
        'model': {
            'CreationTimestamp': '1970-01-01 00:00:00.0',
            'ModelVersion': '1',
            'ModelArn': 'arn:aws:lookoutvision:REGION:123456789:model/myproject/1',
            'Status': 'TRAINING',
            'StatusMessage': 'The model is being trained.'
        }
    }


def test_deploy():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.deploy() == {
        'status': 'HOSTED',
        'model': {
            'ModelVersion': '1',
            'ModelArn': 'arn:aws:lookoutvision:REGION:123456789:model/myproject/1',
            'CreationTimestamp': '1970-01-01 00:00:00.0',
            'Status': 'HOSTED',
            'StatusMessage': 'The model is running',
            'Performance': {
                'F1Score': 1.0,
                'Recall': 1.0,
                'Precision': 1.0
            },
            'OutputConfig': {
                'S3Location': {
                    'Bucket': 'mybucket',
                    'Prefix': ''
                }
            },
            'EvaluationManifest': {
                'Bucket': 'mybucket',
                'Key': 'EvaluationManifest-myproject-1.json'
            },
            'EvaluationResult': {
                'Bucket': 'mybucket',
                'Key': 'EvaluationResult-myproject-1.json'
            },
            'EvaluationEndTimestamp': '1970-01-01 00:00:00.0'
        }
    }


def test_predict():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.predict() == {
        'Source': {
            'Type': 'direct'
        },
        'IsAnomalous': True,
        'Confidence': 1.0
    }


def test_batch_predict():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.batch_predict() == {
        'status': 'Success!',
        'predicted_result': 's3://mytargetdbucket/prdictedresult/'
    }


def test_stop_model(model_version=None):
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.stop_model() == {
        "status": "STOPPING_HOSTING"
    }


def test_delete_lookoutvision_project():
    l4v = DummyFramework(project_name=PROJECT_NAME,
                         model_version=MODEL_VERSION)
    assert l4v.delete_lookoutvision_project(PROJECT_NAME) == {
        "Success": True
    }


# def test_train_one_fold():
#     l4v = DummyFramework(project_name=PROJECT_NAME,
#                          model_version=MODEL_VERSION)
#     assert l4v.train_one_fold(input_bucket=BUCKET_NAME, output_bucket=BUCKET_NAME, s3_path='', model_prefix='',
#                               i_split=0) == {
#                'model': {
#                    'CreationTimestamp': '1970-01-01 00:00:00.0',
#                    'ModelVersion': '1',
#                    'ModelArn': 'arn:aws:lookoutvision:REGION:123456789:model/myproject/1',
#                    'Status': 'TRAINING',
#                    'StatusMessage': 'The model is being trained.'
#                }
#            }


# def test_train_k_fold():
#     l4v = DummyFramework(project_name=PROJECT_NAME,
#                          model_version=MODEL_VERSION)
#     assert type(l4v.train_k_fold()) == type(pd.DataFrame())
