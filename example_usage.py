import pickle

from naivebayes.models.dictionary import Multinomial as DictMultinomial
from naivebayes.models.vector import Multinomial as VectMultinomial
import numpy as np

#####################
# Wine Data Example #
#####################

print("Wine Data Example")

with open('./sample_data/nltk/wine/wine_data.pkl', 'rb') as f:
    wine_data = pickle.load(f)

    ####################
    # Dictionary Model #
    ####################

    wine_bow_data = wine_data["bagofwords"]

    raw_data_top_removed = wine_bow_data["raw_data_top_removed"]
    raw_data = wine_bow_data["raw_data"]
    labels = wine_bow_data["labels"]

    model_dict = DictMultinomial()
    model_dict.train(labels, raw_data_top_removed)
    dict_predictions, dict_scores = model_dict.predict(raw_data)

    matches = 0
    for i in range(0, len(labels)):
        if dict_predictions[i] == labels[i]:
            matches += 1

    print("- Dictionary")
    print("Accuracy: " + str(matches / len(labels)))

    ################
    # Vector Model #
    ################

    wine_vector_data = wine_data["vectors"]

    index_map = wine_vector_data["word_map"]
    reviews_as_vector = wine_vector_data["vectors"]
    labels_as_vectors = wine_vector_data["labels"]

    model_vect = VectMultinomial()
    model_vect.train(labels_as_vectors, reviews_as_vector)
    vect_predictions, vect_scores = model_vect.predict(reviews_as_vector)

    accuracy = np.sum(vect_predictions == labels_as_vectors) / labels_as_vectors.shape[0]
    print("- Vector")
    print("Accuracy: " + str(accuracy))

#####################
# Spam Data Example #
#####################

print("\nSpam Data Example")

with open('./sample_data/kaggle/spam/sms_spam.pkl', 'rb') as f:
    spam_data = pickle.load(f)

    ####################
    # Dictionary Model #
    ####################

    spam_bow_data = spam_data["bagofwords"]

    raw_data_top_removed = spam_bow_data["raw_data_top_removed"]
    raw_data = spam_bow_data["raw_data"]
    labels = spam_bow_data["labels"]

    model_dict = DictMultinomial()
    model_dict.train(labels, raw_data_top_removed)
    dict_predictions, dict_scores = model_dict.predict(raw_data)

    matches = 0
    for i in range(0, len(labels)):
        if dict_predictions[i] == labels[i]:
            matches += 1

    print("- Dictionary")
    print("Accuracy: " + str(matches / len(labels)))

    ################
    # Vector Model #
    ################

    spam_vector_data = spam_data["vectors"]

    index_map = spam_vector_data["word_map"]
    msgs_as_vector = spam_vector_data["vectors"]
    labels_as_vectors = spam_vector_data["labels"]

    model_vect = VectMultinomial()
    model_vect.train(labels_as_vectors, msgs_as_vector)
    vect_predictions, vect_scores = model_vect.predict(msgs_as_vector)

    accuracy = np.sum(vect_predictions == labels_as_vectors) / labels_as_vectors.shape[0]
    print("- Vector")
    print("Accuracy:" + str(accuracy))

    for i, prediction in enumerate(dict_predictions):
        if dict_predictions[i] == "spam":
            dict_predictions[i] = 1
        else:
            dict_predictions[i] = 0

    dict_predictions = np.array(dict_predictions)
    test = np.equal(vect_predictions, dict_predictions)

##############################
# Simple Sports Text Example #
##############################

# Data Found: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

print("\nSports Text Example")

####################
# Dictionary Model #
####################

sports_data = [["a", "great", "game"],
               ["the", "election", "was", "over"],
               ["very", "clean", "match"],
               ["a", "clean", "but", "forgettable", "game"],
               ["it", "was", "a", "close", "election"]]
sports_labels = ["sport", "not sport", "sport", "sport", "not sport"]

model_dict = DictMultinomial()
model_dict.train(sports_labels, sports_data)
dict_predictions, dict_scores = model_dict.predict(sports_data)

matches = 0
for i in range(0, len(dict_predictions)):
    if dict_predictions[i] == sports_labels[i]:
        matches += 1

print("- Dictionary")
print("Accuracy: " + str(matches / len(sports_labels)))

################
# Vector Model #
################

train_data_sports = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
labels_sports = np.array([0, 1, 0, 0, 1])

model_vect = VectMultinomial()
model_vect.train(labels_sports, train_data_sports)
vect_predictions, vect_scores = model_vect.predict(train_data_sports)

accuracy = np.sum(vect_predictions == labels_sports) / labels_sports.shape[0]
print("- Vector")
print("Accuracy: " + str(accuracy))
