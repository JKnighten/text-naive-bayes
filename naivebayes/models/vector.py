import numpy as np


class Multinomial:

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.priors = np.empty(0)
        self.likelihoods = np.empty(0)

    def train(self, labels, train_data):

        # Create Prior Vectors
        sum_of_labels = np.bincount(labels)
        self.priors = sum_of_labels / np.sum(sum_of_labels)

        # Create Empty Likelihood Vectors
        self.likelihoods = np.zeros((self.priors.shape[0], train_data.shape[1]))

        # For Each Label Sum All Vectors Belonging To Label
        for i in range(self.likelihoods.shape[0]):
            self.likelihoods[i, :] = np.sum(train_data[labels == i], axis=0)

        # Likelihood Smoothing
        numb_of_words_in_label = np.sum(self.likelihoods, axis=1)
        numerator = self.likelihoods + 1
        denominator = numb_of_words_in_label + self.smoothing + train_data.shape[1]
        self.likelihoods = numerator / denominator[:, np.newaxis]

    def predict(self, test_data, return_scores=False):

        # Convert To Log Space
        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)

        # Find Product Between Every Test Vector and Likelihood Vector THen Add Correct Prior To Each
        log_space_sums = log_priors + test_data.dot(log_likelihoods.T)

        # Find Label With Max Log Space Sum For Each Test Vector
        predicted_labels = np.argmax(log_space_sums, axis=1)

        # If Desired Exponentiate Log Space Sums And Return Them Along With Predicted Labels
        if return_scores:
            return predicted_labels, np.exp(log_space_sums)

        return predicted_labels
