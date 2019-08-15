import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='pflacco',  
     version='0.3',
     author="Raphael Patrick Prager",
     author_email="raphael.prager@uni-muenster.de",
     description="An python interface to the R package flacco for computing ELA features.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     license = 'MIT',
     url="https://github.com/Reiyan/pflacco",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )