# Naive Bayes Text Classification

Naive Bayes is a machine learning algorithm used for classification. It
is a popular model in the domain of text classification. This package
is designed specifically for the application of Naive Bayes for text
classification.

In text classification the goal is to classify a text document in some 
way. Here are a few examples of text classification tasks:
* Given an Email/SMS determine if it is Spam or Ham(Not Spam)
* Given a review of an item determine if it is positive or negative 
review
* Given a news article determine what category it belongs to(sports,
technology, politics,... )
* Given a book determine who the book's authors is(from a finite list of
 authors)
 
This package provides two different implementations of Multinomial
Naive Bayes. The main difference between the two models is what type
of data is expected by the model. One model expects documents to be 
represented as lists of words, while the other wants documents to be
numpy vectors representing word frequencies.

It is up to the user to pre-process their text data. The most important
pre-processing step its to ensure your data is in the right format
for this package's models. To ensure higher levels of model accuracy, it
is important for text data to be pre-processed in other various ways. 
Some popular pre-processing steps are:
* Remove All Non-Text Characters
* Make All Text Lowercase
* Make Your Text Into a [N-Gram](https://en.wikipedia.org/wiki/N-gram) 
Model
* [Stemming](https://en.wikipedia.org/wiki/Stemming) Your Text
* Removing The Top N Most Frequent Words
* Remove [Stop Words](https://en.wikipedia.org/wiki/Stop_words)
(Similar To Above)
* Remove Numeric Data(If You Think It Is Not Important)
 
It is possible to use this package for tasks other than text 
classification. If your data can be represented by counts(ie how many
times some event/item occur for a given data item) then models in this
package can still be used. This package does not support the use of 
continuous data as input.

For more information about Naive Bayes check these links:
* [Wikipedia - Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
* [A Practical Explanation of a Naive Bayes Classifier](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/)
* [Cornell - Bayes Classifier and Naive Bayes](http://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote05.html)


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

## Models Provided
This package provides two different Multinomial Naive Bayes models. One
operates by using pure python data structures and the other relies on 
the use of numpy.

The models I provided also can be used for online learning. They have
methods to update the model as new data is acquired.

### models.dictionary
This model uses python data structures only. To train this model, the 
user most provide a list of list of words and a list of labels. It is 
up to the user to parse documents into a list of words and provide a
label for each document. To predict a documents label, the user must
provide the document as a list of words.

### models.vector
This model uses numpy arrays as its main data structures. To train this
model the user must provide a matrix of document data and an array of
labels. The matrix's rows represent each document and its columns
represent a specific word. Essentially each document is represented by
a row vector whose elements represent the frequency which a specific 
word appears in that document. To predict a document label, the document
must be transformed into a vector of word frequencies as described as 
above. **Ensure that your training vectors and testing vectors 
correspond to one another. This means they have the same length and 
columns correspond to the same word. For example all vectors are 
length 1 and column 1 represents the word "test"**

## Example Use

Here is an example of using the dictionary model:
```
from naivebayes.models.dictionary import Multinomial

train_labels = ["good", "good", "bad", "bad"]
train_data = [["i", "liked", "it"],
              ["it", "was", "very", "good"],
              ["i", "hated", "it"],
              ["it", "was", "bad]]
              
test_data = [["i", "thought", "it", "was", "bad"]]
              
model = Multinomial()
model.train(train_labels, train_data)
predictions, scores = model_dict.predict(raw_data)
```

Here is an example of using the vector model:
```
import numpy
from naivebayes.models.vector import Multinomial

# 0 = bad, 1 = good
train_labels = np.array([1, 1, 0, 0])

word_map = {"i" : 0, "liked": 1, "it": 2, "was": 3, "very": 4, "good": 5,
            "hated": 6, "bad": 7}
            
train_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 0, 0],
                       [1, 0, 1, 0, 0, 0, 1, 0],
                       [0, 0, 1, 1, 0, 0, 0, 1]])
                       
# "i thought it was bad"
# Notice "thought" was droped because there is no column for "thought"
test_data = np.array([[1, 0, 1, 1, 0, 0, 0, 1]])

model = Multinomial()
model.train(train_labels, train_data)
predictions, scores = model_dict.predict(raw_data)
```

For more examples check *example_usage.py* in the package's main 
directory.

## Possible Future Updates

Possible updates to this package include:
* Implement Bernoulli Naive Bayes
* Implement Complement Naive Bayes
* Implement Gaussian Naive Bayes
