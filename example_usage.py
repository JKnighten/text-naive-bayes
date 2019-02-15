import pickle

from naivebayes.models.dictionary import Multinomial as DictMultinomial
from naivebayes.models.vector import Multinomial as VectMultinomial
import numpy as np

#####################
# Wine Data Example #
#####################

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
    dict_predictions, dict_scores = model_dict.predict(raw_data, return_scores=True)

    matches = 0
    for i in range(0, len(labels)):
        if dict_predictions[i] == labels[i]:
            matches += 1

    print("Wine Data Example - Dictionary")
    print("Accuracy - " + str(matches/len(labels)))


    ################
    # Vector Model #
    ################

    wine_vector_data = wine_data["vectors"]

    index_map = wine_vector_data["word_map"]
    reviews_as_vector = wine_vector_data["vectors"]
    labels_as_vectors = wine_vector_data["labels"]

    model_vect = VectMultinomial()
    model_vect.train(labels_as_vectors, reviews_as_vector)
    vect_predictions, vect_scores = model_vect.predict(reviews_as_vector, return_scores=True)

    accuracy = np.sum(vect_predictions == labels_as_vectors)/labels_as_vectors.shape[0]
    print("Wine Data Example - Vector")
    print("Accuracy - " + str(accuracy))


#####################
# Spam Data Example #
#####################

with open('./sample_data/kaggle/spam/sms_spam.pkl', 'rb') as f:
    spam_data = pickle.load(f)

    spam_bow_data = spam_data["bagofwords"]

    raw_data_top_removed = spam_bow_data["raw_data_top_removed"]
    raw_data = spam_bow_data["raw_data"]
    labels = spam_bow_data["labels"]

    model_dict = DictMultinomial()
    model_dict.train(labels, raw_data_top_removed)
    dict_predictions, dict_scores = model_dict.predict(raw_data, return_scores=True)

    matches = 0
    for i in range(0, len(labels)):
        if dict_predictions[i] == labels[i]:
            matches += 1

    print("Spam Data Example - Dictionary")
    print("Accuracy - " + str(matches/len(labels)))

    ################
    # Vector Model #
    ################

    spam_vector_data = spam_data["vectors"]

    index_map = spam_vector_data["word_map"]
    msgs_as_vector = spam_vector_data["vectors"]
    labels_as_vectors = spam_vector_data["labels"]

    model_vect = VectMultinomial()
    model_vect.train(labels_as_vectors, msgs_as_vector)
    vect_predictions, vect_scores = model_vect.predict(msgs_as_vector, return_scores=True)

    accuracy = np.sum(vect_predictions == labels_as_vectors)/labels_as_vectors.shape[0]
    print("Wine Data Example - Vector")
    print("Accuracy - " + str(accuracy))

    for i, prediction in enumerate(dict_predictions):
        if dict_predictions[i] == "spam":
            dict_predictions[i] = 1
        else:
            dict_predictions[i] = 0

    dict_predictions = np.array(dict_predictions)
    test = np.equal(vect_predictions, dict_predictions)
