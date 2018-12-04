from distutils.core import setup

setup(
    name='naivebayes',
    version='0.1',
    description='Naive Bayes classification for text data implemented in Python',
    url='https://github.com/JKnighten/text-naive-bayes',
    author='Jonathan Knighten',
    author_email='jknigh28@gmail.com',
    install_requires=['nltk'],
    packages=['naivebayes'],
    zip_safe=False
)
