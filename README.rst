===================================
Amazon Lookout for Vision Python SDK
===================================

.. image:: https://img.shields.io/pypi/v/lookoutvision.svg
   :target: https://pypi.python.org/pypi/lookoutvision
   :alt: Latest Version

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
   :target: https://github.com/python/black
   :alt: Code style: black

.. image:: https://img.shields.io/badge/Made%20With-Love-orange.svg
   :target: https://pypi.python.org/pypi/lookoutvision
   :alt: Made With Love

The Amazon Lookout for Vision Python SDK is an open-source library that allows data
scientists and software developers to easily build, train and deploy computer vision (CV)
models using Amazon Lookout for Vision.

* Computer Vision - `Comput vision is an interdisciplinary field that deals with how computers can be made to gain a high level understanding from digital images or videos <https://en.wikipedia.org/wiki/Computer_vision#Definition>`_
* Amazon Lookout for Vision - https://aws.amazon.com/lookout-for-vision/

The Amazon Lookout for Vision Python SDK enables you to do the following.

- Easily check your images for compliants (e.g. right size, shape, etc.)
- Easily rescale images to make them compliant
- Data upload to the necessary S3 structure
- Simple creation of manifest files (incl. upload to the correct location)
- Train a computer vision model using Amazon Lookout for Vision
- Deploy a computer vision model using Amazon Lookout for Vision
- (Batch) Predict using Amazon Lookout for Vision
- Stop the model hosting after you are done/whenever necessary


Table of Contents
-----------------
- `Getting Started With Sample Jupyter Notebooks <#getting-started-with-sample-jupyter-notebooks>`__
- `Installing the Amazon Lookout for Vision Python SDK <#installing-the-amazon-lookout-for-vision-python-sdk>`__
- `Further Readings <#further-readings>`__
- `Licensing <#licensing>`__


Getting Started With Sample Jupyter Notebooks
---------------------------------------------

The best way to quickly review how the Amazon Lookout for Vision Python SDK works
is to review the related example notebooks. These notebooks provide code and
descriptions for creating and running a full project in Amazon Lookout for Vision Using
the Amazon Lookout for Vision Python SDK.


Example Notebooks in SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Amazon SageMaker, upload the Jupyter notebook from the **example/** folder of this repository.

1. To run this example `Create a Notebook Instance <https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html>`__ in SageMaker.
2. Add an inline policy to your Amazon SageMaker role in IAM with the following JSON structure

::

	{
	    "Version": "2012-10-17",
	    "Statement": [
	        {
	            "Effect": "Allow",
	            "Action": [
	                "lookoutvision:*"
	            ],
	            "Resource": "*"
	        }
	    ]
	}
3. Upload the Jupyter notebook from **example/** folder.
4. Bring your good and bad images and upload them to your notebook instance.

*Note:* Store the good images in a folder named **good/** and the bad images in a folder named **bad/**.


Run Example Notebooks Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run theAmazon Lookout for Vision Python SDK example notebooks locally, download
the sample notebook and open them in a working Jupyter instance.

1. Install Jupyter: https://jupyter.readthedocs.io/en/latest/install.html

2. Download the following files from:
   https://github.com/aws/amazon-lookout-for-vision-python-sdk/tree/master/example.

  * :code:`lookout_for_vision_example.ipynb`

3. Open the files in Jupyter.


Installing the Amazon Lookout for Vision Python SDK
--------------------------------------------------

The Amazon Lookout for Vision Python SDK is built to PyPI and can be installed with
pip as follows.


::

        pip install lookoutvision

You can install from source by cloning this repository and running a pip install
command in the root directory of the repository:

::

    git clone https://github.com/aws/amazon-lookout-for-vision-python-sdk.git
    cd amazon-lookout-for-vision-python-sdk
    pip install .


Supported Operating Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Amazon Lookout for Vision Python SDK supports Unix/Linux and Mac.


Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

The Amazon Lookout for Vision Python SDK is tested on:

* Python 3.6


Overview of SDK
---------------

The Amazon Lookout for Vision Python SDK provides a Python API that enables you to
create computer vision models using Amazon Lookout for Vision and directly in your
Python code and Jupyter notebooks.

Using this SDK you can:

1. Easily check your images for compliants (e.g. right size, shape, etc.)
2. Easily rescale images to make them compliant
3. Data upload to the necessay S3 structure
4. Simple creation of manifest files (incl. upload to the correct location)
5. Train a computer vision model using Amazon Lookout for Vision
6. Deploy a computer vision model using Amazon Lookout for Vision
7. (Batch) Predict using Amazon Lookout for Vision
8. Stop the hosting of the model when you are done.

For a detailed API reference of the Amazon Lookout for Vision Python SDK,
be sure to view this documentation on


Amazon Lookout for Vision
~~~~~~~~~~~~~~~~~~

Amazon Lookout for Vision is a machine learning (ML) service that spots defects and anomalies in visual representations using computer vision (CV). With Amazon Lookout for Vision, manufacturing companies can increase quality and reduce operational costs by quickly identifying differences in images of objects at scale. For example, Amazon Lookout for Vision can be used to identify missing components in products, damage to vehicles or structures, irregularities in production lines, miniscule defects in silicon wafers, and other similar problems. Amazon Lookout for Vision uses ML to see and understand images from any camera as a person would, but with an even higher degree of accuracy and at a much larger scale. Amazon Lookout for Vision allows customers to eliminate the need for costly and inconsistent manual inspection, while improving quality control, defect and damage assessment, and compliance. In minutes, you can begin using Amazon Lookout for Vision to automate inspection of images and objects–with no machine learning expertise required.


AWS Permissions
---------------
As a managed service, Amazon Lookout for Vision performs operations on your behalf on
AWS hardware that is managed by Amazon Lookout for Vision.  Amazon Lookout for Vision can
perform only operations that the user permits.  You can read more about which
permissions are necessary in the `AWS Documentation
<https://docs.aws.amazon.com/lookout-for-vision/latest/developer-guide/what-is.html>`__.

The Amazon Lookout for Vision Python SDK should not require any additional permissions
aside from what is required for using .boto3.  However, if you are
using an IAM role with a path in it, you should grant permission for
``iam:GetRole``.


Security
---------------

See https://github.com/aws-samples/amazon-lookout-for-vision-python-sdk/blob/main/CONTRIBUTING.md#security-issue-notifications for more information.


Licensing
---------
Amazon Lookout for Vision Python SDK is licensed under the Apache 2.0 License. It is
copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved. The
license is available at: http://aws.amazon.com/apache2.0/