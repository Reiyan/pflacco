import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
     name='pflacco',  
     version='1.0.1',
     author="Raphael Patrick Prager",
     author_email="raphael.prager@gmx.de",
     description="An python interface to the R package flacco for computing ELA features.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     license='MIT',
     install_requires=required,
     url="https://pypi.org/project/pflacco/",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )