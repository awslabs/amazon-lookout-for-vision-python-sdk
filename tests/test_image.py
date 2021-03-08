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

from lookoutvision import image

INPUT_BUCKET_NAME = "mybucket"
OUTPUT_BUCKET_NAME = "newbucket"


class DummyFramework(image.Image):
    """docstring for DummyFramework"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def __image_size_checker(self, image_type="good", verbose=False):
        return {'no_of_images': 5, 'compliant_images': 5, 'compliant': True}

    @classmethod
    def __image_shape_checker(self, image_type="good", verbose=False):
        return {
            'no_of_images': 5,
            'compliant': 5,
            'status': 'Image sizes are equal!',
            'min_image_shape': (1750, 1732, 3)
        }

    @classmethod
    def __is_compliant(self, response):
        good = response["good"]["compliant"]
        bad = response["bad"]["compliant"]
        return good and bad

    @classmethod
    def __is_too_small(self, response):
        compliant = self.__is_compliant(response=response)
        if not compliant:
            return min(response["good"]["min_size"], response["bad"]["min_size"]) < 64
        return False

    @classmethod
    def __is_too_large(self, response):
        compliant = self.__is_compliant(response=response)
        if not compliant:
            return max(response["good"]["max_size"], response["bad"]["max_size"]) > 64
        return False

    def check_image_sizes(self, verbose=False):
        return {
            'good': {
                'no_of_images': 5,
                'compliant_images': 5,
                'compliant': True
            },
            'bad': {
                'no_of_images': 4,
                'compliant_images': 4,
                'compliant': True
            }
        }

    def check_image_shape(self, verbose=False):
        return {
            'good': {
                'no_of_images': 5,
                'compliant': 5,
                'status': 'Image sizes are equal!',
                'min_image_shape': (1750, 1732, 3)
            },
            'bad': {
                'no_of_images': 4,
                'compliant': 4,
                'status': 'Image sizes are equal!',
                'min_image_shape': (1750, 1732, 3)
            },
            "shape_recommendation": (42, 42, 42)
        }

    def rescale(self, prefix="rescaled_"):
        response = self.check_image_sizes()
        too_large = self.__is_too_large(response=response)
        too_small = self.__is_too_small(response=response)
        return {'good': 'Ok', 'bad': 'Ok'}

    def upload_from_local(self, bucket, train_and_test=True, test_split=0.2, prefix="", content_type="image/jpeg"):
        return {
            'train_data': {
                'normal': 's3://{}/training/normal/'.format(bucket),
                'anomaly': 's3://{}/training/anomaly/'.format(bucket)
            },
            'test_data': {
                'normal': 's3://{}/validation/normal/'.format(bucket),
                'anomaly': 's3://{}/validation/anomaly/'.format(bucket)
            }
        }

    def copy_from_s3(self, input_bucket, output_bucket, prefix_good="good", prefix_bad="bad",
                     train_and_test=True, test_split=0.2):
        return {
            "status": "Success!",
            "objects_copied": 42,
            'train_data': {
                'normal': 's3://{}/training/normal/'.format(output_bucket),
                'anomaly': 's3://{}/training/anomaly/'.format(output_bucket)
            },
            'test_data': {
                'normal': 's3://{}/validation/normal/'.format(output_bucket),
                'anomaly': 's3://{}/validation/anomaly/'.format(output_bucket)
            }
        }


def test_rescale():
    l4v = DummyFramework()
    assert l4v.rescale() == {'good': 'Ok', 'bad': 'Ok'}


def test_check_image_shape():
    l4v = DummyFramework()
    assert l4v.check_image_shape() == {
        'good': {
            'no_of_images': 5,
            'compliant': 5,
            'status': 'Image sizes are equal!',
            'min_image_shape': (1750, 1732, 3)
        },
        'bad': {
            'no_of_images': 4,
            'compliant': 4,
            'status': 'Image sizes are equal!',
            'min_image_shape': (1750, 1732, 3)
        },
        "shape_recommendation": (42, 42, 42)
    }


def test_check_image_sizes():
    l4v = DummyFramework()
    assert l4v.check_image_sizes() == {
        'good': {
            'no_of_images': 5,
            'compliant_images': 5,
            'compliant': True
        },
        'bad': {
            'no_of_images': 4,
            'compliant_images': 4,
            'compliant': True
        }
    }


def test_upload_from_local():
    l4v = DummyFramework()
    assert l4v.upload_from_local(bucket=INPUT_BUCKET_NAME) == {
        'train_data': {
            'normal': 's3://mybucket/training/normal/',
            'anomaly': 's3://mybucket/training/anomaly/'
        },
        'test_data': {
            'normal': 's3://mybucket/validation/normal/',
            'anomaly': 's3://mybucket/validation/anomaly/'
        }
    }


def test_copy_from_s3():
    l4v = DummyFramework()
    assert l4v.copy_from_s3(input_bucket=INPUT_BUCKET_NAME, output_bucket=OUTPUT_BUCKET_NAME) == {
        "status": "Success!",
        "objects_copied": 42,
        'train_data': {
            'normal': 's3://newbucket/training/normal/',
            'anomaly': 's3://newbucket/training/anomaly/'
        },
        'test_data': {
            'normal': 's3://newbucket/validation/normal/',
            'anomaly': 's3://newbucket/validation/anomaly/'
        }
    }
