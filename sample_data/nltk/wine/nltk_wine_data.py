from functools import reduce

import nltk
import string
import pickle
import numpy as np
import copy

# Download WebText Corpus If Not Already Downloaded
nltk.download('webtext')
from nltk.corpus import webtext


# Break Raw Data Into Individual Reviews
wine_reviews_raw = webtext.raw("wine.txt").split("\n")

# Used To Remove Non-Text Data
translator = str.maketrans('', '', string.punctuation)


################
# Bag Of Words #
################

cleaned_review_data = []
review_labels = []
label_bag_of_words = ["", ""]
for review in wine_reviews_raw:

    # Extract Score For Each Review
    score = 0

    if "*" in review:
        asterisk_count = review.count("*")

        # Don't Count Stars In ()
        if "(*)" in review:
            asterisk_count -= 1

        score = asterisk_count

    # Create Two Level Labels
    # 0 If Score Is 2 or Lower (Bad Wine)
    # 1 If Score Is 3 Or Higher (Good Wine)
    if score <= 2:
        review_labels.append(0)
    else:
        review_labels.append(1)

    # Convert All Text To Lowercase
    # Remove All Non-Text Data
    # Remove "No Stars" On
    if score <= 2:

        # Remove "No Stars" From Reviews With 0 Stars
        if "No Stars" in review:
            review = review.replace('No Stars', '')

        label_bag_of_words[0] += review.lower().translate(translator) + " "
        cleaned_review_data.append(nltk.tokenize.word_tokenize(review.lower().translate(translator)))
    else:
        label_bag_of_words[1] += review.lower().translate(translator) + " "
        cleaned_review_data.append(nltk.tokenize.word_tokenize(review.lower().translate(translator)))


# Tokenize And Create Frequency Distribution For Each Label
label_0_tokens = nltk.tokenize.word_tokenize(label_bag_of_words[0])
freq_dist_0 = nltk.FreqDist(label_0_tokens)

label_1_tokens = nltk.tokenize.word_tokenize(label_bag_of_words[1])
freq_dist_1 = nltk.FreqDist(label_1_tokens)

# Remove Top N Frequent Words
n = 25
label_0_top_50 = {word for word, freq in freq_dist_0.most_common(n)}
label_1_top_50 = {word for word, freq in freq_dist_1.most_common(n)}

words_to_remove = label_0_top_50.union(label_1_top_50)

for word_to_remove in words_to_remove:
    del freq_dist_0[word_to_remove]
    del freq_dist_1[word_to_remove]

# Make Dictionary Containing Frequency Distribution For Each Label
wine_freq_data = {}
wine_freq_data[0] = freq_dist_0
wine_freq_data[1] = freq_dist_1

# Get Label Counts
label_counts = {}
label_counts[0] = len(list(filter(lambda label: label == 0, review_labels)))
label_counts[1] = len(list(filter(lambda label: label == 1, review_labels)))


cleaned_review_data_top_removed = copy.deepcopy(cleaned_review_data)
# Remove Frequent Words From Raw Data
for i, word_list in enumerate(cleaned_review_data_top_removed):

    for word in list(word_list):

        if word in words_to_remove:
            word_list.remove(word)


###################
# Vectorized Data #
###################

dictionary = reduce(lambda key_a, key_b: set(wine_freq_data[key_a].keys()).union(set(wine_freq_data[key_b].keys())),
                    wine_freq_data)

data_as_vectors = np.zeros((len(wine_reviews_raw), len(dictionary)))
label_vectors = np.zeros((len(wine_reviews_raw))).astype("int64")

# Create map
word_map = {}
for i, key in enumerate(dictionary):
    word_map[key] = i

# Create Vectors
for i, review in enumerate(wine_reviews_raw):

    # Put Review Label Into A Vector
    label_vectors[i] = review_labels[i]

    if review_labels[i] == 0:

        # Remove "No Stars" From Reviews With 0 Stars
        if "No Stars" in review:
            review = review.replace('No Stars','')

    words = nltk.tokenize.word_tokenize(review.lower().translate(translator))

    for word in words:
        if word in word_map:
            data_as_vectors[i, word_map[word]] += 1

final_data = {"bagofwords": {"freq_data": wine_freq_data, "label_counts": label_counts, "raw_data": cleaned_review_data,
                             "labels": review_labels, "raw_data_top_removed": cleaned_review_data_top_removed},
              "vectors": {"vectors": data_as_vectors, "word_map": word_map, "labels": label_vectors}}

# Write Pickle File
with open('wine_data.pkl', 'wb') as f:
        pickle.dump(final_data, f, pickle.HIGHEST_PROTOCOL)
