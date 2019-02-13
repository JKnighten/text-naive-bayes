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

    freq_data = wine_bow_data["freq_data"]
    label_counts = wine_bow_data["label_counts"]
    raw_data = wine_bow_data["raw_data"]
    labels = wine_bow_data["labels"]

    model = DictMultinomial()
    model.train(label_counts, freq_data)

    dict_predictions, dict_scores = model.predict(raw_data, return_scores=True)

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

    model = VectMultinomial()
    model.train(labels_as_vectors, reviews_as_vector)
    vect_predictions, vect_scores = model.predict(reviews_as_vector, return_scores=True)

    accuracy = np.sum(vect_predictions == labels_as_vectors)/labels_as_vectors.shape[0]
    print("Wine Data Example - Vector")
    print("Accuracy - " + str(accuracy))
