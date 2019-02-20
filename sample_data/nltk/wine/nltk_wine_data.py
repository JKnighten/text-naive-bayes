import argparse
import copy
import nltk
import numpy as np
import pickle
import string
import os


###################
# Parse Arguments #
###################
parser = argparse.ArgumentParser(description="Optional Arguments For NLTK Wine Data")

parser.add_argument("--top_n_remove", type=int, help="Top N Frequent Words To Remove", default=0)

args = parser.parse_args()

if args.top_n_remove < 0:
    parser.error("--top_n_remove Must Be 0 Or Greater")
else:
    n = args.top_n_remove

print("Number Of Most Frequent Words To Remove: " + str(args.top_n_remove))


# Download WebText Corpus If Not Already Downloaded
nltk.download("webtext")
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
good_review_freq = nltk.FreqDist()
bad_review_freq = nltk.FreqDist()

for review in wine_reviews_raw:

    # Extract Score Label For Each Review
    score = 0
    if "*" in review:
        asterisk_count = review.count("*")

        # Don't Count Stars In ()
        if "(*)" in review:
            asterisk_count -= 1

        score = asterisk_count

    # Create Two Level Labels
    # "bad If Score Is 2 or Lower (Bad Wine)
    # "good If Score Is 3 Or Higher (Good Wine)
    if score <= 2:
        review_labels.append("bad")
    else:
        review_labels.append("good")

    # Convert All Text To Lowercase
    # Remove All Non-Text Data
    # Remove "No Stars" On
    tokens = nltk.tokenize.word_tokenize(review.lower().translate(translator))

    # Put Tokens of Review In Correct Word Freq Distribution
    if score <= 2:
        # Remove "No Stars" From Reviews With 0 Stars
        if "No Stars" in review:
            review = review.replace("No Stars", '')

        bad_review_freq.update(tokens)
    else:
        good_review_freq.update(tokens)

    cleaned_review_data.append(tokens)

# Remove Top N Frequent Words
bad_review_top = {word for word, freq in bad_review_freq.most_common(n)}
good_review_top = {word for word, freq in good_review_freq.most_common(n)}
words_to_remove = bad_review_top.union(good_review_top)

# Remove Frequent Words From Cleaned Review  Data
cleaned_review_data_top_removed = copy.deepcopy(cleaned_review_data)
for i, word_list in enumerate(cleaned_review_data_top_removed):
    for word in list(word_list):
        if word in words_to_remove:
            word_list.remove(word)


###################
# Vectorized Data #
###################

# Remove Top N Words From Word Frequency Distributions
for word_to_remove in words_to_remove:
    del bad_review_freq[word_to_remove]
    del good_review_freq[word_to_remove]

dictionary = set(bad_review_freq.keys()).union(set(good_review_freq.keys()))

data_as_vectors = np.zeros((len(wine_reviews_raw), len(dictionary)))
label_vectors = np.zeros((len(wine_reviews_raw))).astype("int64")

# Create Word -> Index Map
word_map = {key: i for i, key in enumerate(dictionary)}

# Create Vector For Each Review
for i, review in enumerate(cleaned_review_data_top_removed):

    # Put Review Label Into A Vector
    if review_labels[i] == "bad":
        label_vectors[i] = 0
    else:
        label_vectors[i] = 1

    # Create Vector For Current Review
    for word in review:
        if word in word_map:
            data_as_vectors[i, word_map[word]] += 1

# Final Output of Script
final_data = {"bagofwords": {"raw_data": cleaned_review_data, "raw_data_top_removed": cleaned_review_data_top_removed,
                             "labels": review_labels},
              "vectors": {"vectors": data_as_vectors, "word_map": word_map, "labels": label_vectors}}

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Write Pickle File
with open(os.path.join(__location__, "wine_data.pkl"), "wb") as f:
        pickle.dump(final_data, f, pickle.HIGHEST_PROTOCOL)
