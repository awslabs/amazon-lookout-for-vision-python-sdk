from setuptools import find_packages, setup

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lookoutvision',
    packages=find_packages(include=['lookoutvision']),
    version='0.1.0',
    description='Python SDK for Amazon Lookout for Vision',
    author='Michael Wallner',
    author_email="wallnm@amazon.com",
    license='Apache-2.0',
    install_requires=[
        'numpy',
        'pandas',
        'boto3',
        'scikit-image',
        'seaborn',
        'matplotlib'
    ],
    url='',
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    zip_safe=True)
