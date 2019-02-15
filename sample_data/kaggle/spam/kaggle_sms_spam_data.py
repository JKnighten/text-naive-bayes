# TODO - Take Argument For Top N Words To Remove

import csv
import string
import nltk
import pickle
import numpy as np

# Kaggle SMS Spam Dataset
# Found Here: https://www.kaggle.com/uciml/sms-spam-collection-dataset
with open("./spam.csv", encoding="ISO-8859-1") as csv_file:
    reader = csv.reader(csv_file)

    # Column Names
    next(reader)

    ################
    # Bag Of Words #
    ################

    labels = []
    raw_msgs = []
    cleaned_msgs_list = []
    spam_word_freq = nltk.FreqDist()
    ham_word_freq = nltk.FreqDist()

    # Used To Remove Non-Text Data
    translator = str.maketrans('', '', string.punctuation)

    # Read Each Spam/Ham Message
    for line in reader:
        label = line[0]
        msg = line[1]

        labels.append(line[0])
        raw_msgs.append(line[1])

        # Convert All Text To Lowercase
        # Remove All Non-Text Data
        cleaned_msg = msg.lower().translate(translator)
        tokens = nltk.tokenize.word_tokenize(cleaned_msg)
        cleaned_msgs_list.append(tokens)

        if label == "spam":
            spam_word_freq.update(tokens)
        else:
            ham_word_freq.update(tokens)

    # Find Top N Frequent Words
    n = 100
    spam_top = {word for word, freq in spam_word_freq.most_common(n)}
    ham_top = {word for word, freq in ham_word_freq.most_common(n)}
    words_to_remove = spam_top.union(ham_top)

    # Remove Top N Words From Cleaned Messages
    for i, word_list in enumerate(cleaned_msgs_list):
        for word in list(word_list):
            if word in words_to_remove:
                word_list.remove(word)

    ###################
    # Vectorized Data #
    ###################

    # Remove Top N Words From Word Frequency Distributions
    for word_to_remove in words_to_remove:
        del spam_word_freq[word_to_remove]
        del ham_word_freq[word_to_remove]

    dictionary = set(spam_word_freq.keys()).union(set(ham_word_freq.keys()))

    data_as_vectors = np.zeros((len(labels), len(dictionary)))
    label_vectors = np.zeros((len(labels))).astype("int64")

    # Create Word -> Index Map
    word_map = {key: i for i, key in enumerate(dictionary)}

    # Create Vector For Each Message
    for i, word_list in enumerate(cleaned_msgs_list):

        # Put Msg Label Into A Vector
        if labels[i] == 'spam':
            label_vectors[i] = 1
        else:
            label_vectors[i] = 0

        # Create Vector For Current Message
        for word in word_list:
            if word in word_map:
                data_as_vectors[i, word_map[word]] += 1

    # Final Output of Script
    final_data = {"bagofwords": {"raw_data": cleaned_msgs_list, "labels": labels,
                                 "raw_data_top_removed": cleaned_msgs_list},
                  "vectors": {"vectors": data_as_vectors, "word_map": word_map, "labels": label_vectors}}

    # Write Pickle File
    with open('./sms_spam.pkl', 'wb') as f:
        pickle.dump(final_data, f, pickle.HIGHEST_PROTOCOL)
