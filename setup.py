from distutils.core import setup

setup(
    name='naivebayes',
    version='1.0',
    description='Naive Bayes classification for text data implemented in Python',
    url='https://github.com/JKnighten/text-naive-bayes',
    author='Jonathan Knighten',
    author_email='jknigh28@gmail.com',
    install_requires=['numpy'],
    extras_require={'sample_data': ['nltk']},
    packages=['naivebayes'],
    zip_safe=False
)
