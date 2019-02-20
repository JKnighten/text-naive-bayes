# Naive Bayes Text Classification


## Installation

To install this package into your current python environment, first
ensure your environment has an updated pip installation. Then execute 
the following command while in this package's main directory(the
directory containing setup.py):
```
 pip install -e .
```

This package requires numpy to function. If you want to run the scripts
located in the sample_data directory, you will also need to install
NLTK to your environment. The scripts in sample_data will create sample
data sets to be used with this package.

To install NLTK while installing the package simply execute the 
following:
```
pip install -e .[sample_data]
```

## Provided Sample Data Scripts

Contained within this package is two scripts that help create sample
data to be used with this package. One is from the NLTK package and
contains wine review data and the other is from 
[Kaggle](https://www.kaggle.com/) and contains SMS spam and ham data.
The NLTK data is self contained and will download the data when the 
script is executed. The Kaggle data will require you to make an account
and download it from 
[here](https://www.kaggle.com/uciml/sms-spam-collection-dataset). Once 
you acquire the data place spam.csv into the following directory:
sample_data/kaggle/spam. Both of these scripts will create a pickle file
that contains different versions of the data that have already been
preprocessed. Both scripts have an optional argument that will remove
the top N most frequent words from the data.

## Generate Documentation

This package can automatically generate documentation thanks to the
[Sphinx](http://www.sphinx-doc.org/en/master/) package. Sphinx provides
numerous documentation outputs from HTML to LaTeX and even plain
text(for a full list look 
[here](https://www.sphinx-doc.org/en/master/man/sphinx-build.html)). To
build the documentation simply navigate to the docs directory and
execute the following:
```
make <doc type>
```

where <doc type> is the flag for the type of documentation output you
want. The documentation will be found in the docs/_build directory.