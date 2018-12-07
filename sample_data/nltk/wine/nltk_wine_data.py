import nltk
import string
import pickle

# Download WebText Corpus If Not Already Downloaded
nltk.download('webtext')
from nltk.corpus import webtext


# Break Raw Data Into Individual Reviews
wine_reviews_raw = webtext.raw("wine.txt").split("\n")

# Used To Remove Non-Text Data
translator = str.maketrans('', '', string.punctuation)

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

# Make Dictionary Containing Frequency Distribution For Each Label
wine_freq_data = {}
wine_freq_data[0] = freq_dist_0
wine_freq_data[1] = freq_dist_1

# Get Label Counts
label_counts = {}
label_counts[0] = len(list(filter(lambda label: label == 0, review_labels)))
label_counts[1] = len(list(filter(lambda label: label == 1, review_labels)))

# Write Pickle File
with open('wine_data.pkl', 'wb') as f:
        pickle.dump([wine_freq_data, label_counts, cleaned_review_data, review_labels], f, pickle.HIGHEST_PROTOCOL)
