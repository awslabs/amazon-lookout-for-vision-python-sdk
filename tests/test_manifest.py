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

from lookoutvision import manifest

BUCKET_NAME = "mybucket"
S3_PATH = "mypath"
DATASETS = ["training", "validation"]

class DummyFramework(manifest.Manifest):
	"""docstring for DummyFramework"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def generate_manifests(self):
		return {
			"training": {
				"normal": {
					"source-ref": "s3://{}/training/normal/good.jpg".format(BUCKET_NAME),
					"auto-label": 1,
					"auto-label-metadata": {
					"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "normal",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				},
				"anomaly": {
						"source-ref": "s3://{}/training/anomaly/bad.jpg".format(BUCKET_NAME),
						"auto-label": 0,
						"auto-label-metadata": {"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "anomaly",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				}
			},
			"validation": {
				"normal": {
					"source-ref": "s3://{}/validation/normal/good.jpg".format(BUCKET_NAME),
					"auto-label": 1,
					"auto-label-metadata": {
					"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "normal",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				},
				"anomaly": {
						"source-ref": "s3://{}/validation/anomaly/bad.jpg".format(BUCKET_NAME),
						"auto-label": 0,
						"auto-label-metadata": {"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "anomaly",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				}
			}
		}

	def push_manifests(self):
		success = {}
		for ds in self.datasets:
			success[ds] = {
				"bucket": "{}".format(self.bucket),
				"key": "{}.manifest".format(ds),
				"location": "s3://{}/{}.manifest".format(self.bucket, ds)
			}
		return success

def test_get_bucket():
	mft = manifest.Manifest(bucket=BUCKET_NAME, s3_path=S3_PATH, datasets=DATASETS)
	assert mft.get_bucket() == BUCKET_NAME

def test_get_datasets():
	mft = manifest.Manifest(bucket=BUCKET_NAME, s3_path=S3_PATH, datasets=DATASETS)
	assert mft.get_datasets() == DATASETS

def test_push_manifests():
	mft = DummyFramework(bucket=BUCKET_NAME, s3_path=S3_PATH, datasets=DATASETS)
	assert mft.push_manifests() == {
		"training": {
			"bucket": "mybucket",
			"key": "training.manifest",
			"location": "s3://mybucket/training.manifest"
		},
		"validation": {
			"bucket": "mybucket",
			"key": "validation.manifest",
			"location": "s3://mybucket/validation.manifest"
		}
	}

def test_generate_manifests():
	mft = DummyFramework(bucket=BUCKET_NAME, s3_path=S3_PATH, datasets=DATASETS)
	assert mft.generate_manifests() == {
			"training": {
				"normal": {
					"source-ref": "s3://mybucket/training/normal/good.jpg",
					"auto-label": 1,
					"auto-label-metadata": {
					"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "normal",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				},
				"anomaly": {
						"source-ref": "s3://mybucket/training/anomaly/bad.jpg",
						"auto-label": 0,
						"auto-label-metadata": {"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "anomaly",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				}
			},
			"validation": {
				"normal": {
					"source-ref": "s3://mybucket/validation/normal/good.jpg",
					"auto-label": 1,
					"auto-label-metadata": {
					"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "normal",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				},
				"anomaly": {
						"source-ref": "s3://mybucket/validation/anomaly/bad.jpg",
						"auto-label": 0,
						"auto-label-metadata": {"confidence": 1,
						"job-name": "labeling-job/auto-label",
						"class-name": "anomaly",
						"human-annotated": "yes",
						"creation-date": "1970-01-01T00:00:00.0",
						"type": "groundtruth/image-classification"
					}
				}
			}
		}