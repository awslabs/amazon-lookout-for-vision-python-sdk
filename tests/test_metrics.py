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

from lookoutvision import metrics
import pandas as pd

PROJECT_NAME = "myproject"

class DummyFramework(metrics.Metrics):
	"""docstring for DummyFramework"""
	def __init__(self, project_name, *args, **kwargs):
		super().__init__(project_name=project_name, *args, **kwargs)

	def describe_model(self, model_version="1"):
		return pd.DataFrame()

	def describe_models(self):
		return pd.DataFrame()

	def compare_models(self, figsize=(8, 6)):
		return None

def test_describe_model():
	mrc = DummyFramework(project_name=PROJECT_NAME)
	assert type(mrc.describe_model(model_version="3")) == type(pd.DataFrame())

def test_describe_models():
	mrc = DummyFramework(project_name=PROJECT_NAME)
	assert type(mrc.describe_models()) == type(pd.DataFrame())

def test_compare_models():
	mrc = DummyFramework(project_name=PROJECT_NAME)
	assert mrc.compare_models() == None