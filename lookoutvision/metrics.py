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

import boto3
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import json


class Metrics():
    """Return metrics for a given Amazon Lookout for Vision project.

        Attributes:
        project_name    The name of the Amazon Lookout for Vision project.
        lv              The Amazon Lookout for Vision boto3 client.

    """

    def __init__(self, project_name):
        """Creates a Metrics object f°or Amazon Lookout for Vision.
        It can retrieve key metrics, namely precision, recall and f1 score
        from a given Amazon Lookout for Vision project.

        Technical documentation on how Amazon Lookout for Vision works can be
        found at: https://aws.amazon.com/lookout-for-vision/

        Args:
            project_name (str): Name of the Amazon Lookout for Vision to interact with.

        """
        super(Metrics, self).__init__()
        self.project_name = project_name
        self.lv = boto3.client("lookoutvision")
        self.s3 = boto3.client("s3")

    def describe_model(self, model_version="1"):
        """Gets and outputs the metrics for a given model version of your project.

        Args:
            model_version (str): The version of the model that should be describedmodel_version

        Returns:
            pandas.core.frame.DataFrame: a DataFrame with the metadata of your model

        """
        status = self.lv.list_models(
            ProjectName=self.project_name
        )["Models"]
        model = [x for x in status if x["ModelVersion"] == model_version]
        if model[0]["Status"] not in ["TRAINED", "HOSTED"]:
            print("Warning: Your model needs to be in TRAINED or HOSTED state!")
            return pd.DataFrame([None])
        else:
            return pd.DataFrame(model[0])

    def describe_models(self):
        """Gets and outputs all metrics of your project.

        Args:
            None

        Returns:
            pandas.core.frame.DataFrame: a DataFrame with the metadata of all models

        """
        models = self.lv.list_models(
            ProjectName=self.project_name
        )["Models"]
        output = pd.DataFrame()
        for model in models:
            if model["Status"] not in ["TRAINED", "HOSTED"]:
                print("Warning: Your model needs to be in TRAINED or HOSTED state! Skipping model version {}".format(
                    model["ModelVersion"]))
            else:
                output = pd.concat(objs=[output, pd.DataFrame(model)], axis=0)
        return output

    def compare_models(self, figsize=(8, 6)):
        """Visualizes the metrics of all models against each other

        Args:
            figisze (tupel): a tupel for (width, length)

        Returns:
            None

        """
        models = self.describe_models()
        plt.figure(figsize=figsize)
        ax = sns.barplot(x="ModelVersion", y="Performance",
                         hue=models.index, data=models)
        plt.show()

    def k_fold_model_summary(self, output_bucket: str, model_prefix: str, n_splits: int):
        """
        Summary of evaluation of k fold cross validation. Each model returns an evaluation result. This function
        summarizes these evaluation results into one DataFrame.

        Arguments:
            output_bucket (str): The output S3 bucket which was used for model evaluation results.
            model_prefix (str): The optional model prefix which was used for the evaluation results.
            n_splits (int): number of splits within the k-fold cross validation. n_splits = k in that regard.
        """
        client = self.s3
        project_name = self.project_name
        df_metrics_list = []
        for i_split in range(n_splits):
            result = client.get_object(Bucket=output_bucket,
                                       Key=f'{model_prefix}EvaluationResult-{project_name}_{i_split}-1.json')
            text = result["Body"].read().decode()
            content = json.loads(text)
            df_m = pd.DataFrame(content['AggregatedEvaluationResults']['Metrics'], index=[i_split])
            df_m['model_name'] = f'{project_name}_{i_split}'
            df_m['model_version'] = content['Version']
            df_m['NumberOfTrainImages'] = content['EvaluationDetails']['NumberOfTrainingImages']
            df_m['NumberOfTestImages'] = content['EvaluationDetails']['NumberOfTestingImages']
            df_metrics_list.append(df_m)

        return pd.concat(df_metrics_list)
