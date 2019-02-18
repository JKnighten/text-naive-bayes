import numpy as np


class Multinomial:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.priors = np.empty(0)
        self.likelihoods = np.empty(0)
        self.empty_likelihoods = np.empty(0)
        self.label_counts = np.empty(0)

    def train(self, labels, train_data):

        # Create Prior Vectors
        self.label_counts = np.bincount(labels)
        self.priors = self.label_counts / np.sum(self.label_counts)

        # Create Empty Likelihood Vectors
        self.likelihoods = np.zeros((self.priors.shape[0], train_data.shape[1]))

        # For Each Label Sum All Vectors Belonging To Label
        for i in range(self.likelihoods.shape[0]):
            self.likelihoods[i, :] = np.sum(train_data[labels == i], axis=0)

        # Likelihood Smoothing
        numb_of_words_in_label = np.sum(self.likelihoods, axis=1)

        numerator = self.likelihoods + self.smoothing
        denominator = numb_of_words_in_label + self.smoothing * train_data.shape[1]
        self.likelihoods = numerator / denominator[:, np.newaxis]

        # Store Empty Likelihood Values To Be Used In The Update Method
        self.empty_likelihoods = np.zeros((self.priors.shape[0], 1))
        self.empty_likelihoods = self.smoothing / (numb_of_words_in_label + self.smoothing * train_data.shape[1])

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

    def update(self, new_labels, new_train_data):

        dictionary_size = self.likelihoods.shape[1]

        # Find The Number Of Words Appearing In Each Label
        label_word_count = (self.smoothing - self.empty_likelihoods * self.smoothing * dictionary_size) \
                           / self.empty_likelihoods

        # Find The Frequency Of Each Word For Each Label
        self.likelihoods = self.likelihoods * (label_word_count[:, np.newaxis] + self.smoothing * dictionary_size) \
                           - self.smoothing

        # Add The New Training Data To The Model
        # Updates:
        #   Word Frequency For Each Label - Currently Stored in self.likelihoods
        #   Label Counts
        #   Number Of Words In Each Label
        for i in range(self.likelihoods.shape[0]):
            self.likelihoods[i, :] = self.likelihoods[i, :] + np.sum(new_train_data[new_labels == i], axis=0)
            label_word_count[i] = label_word_count[i] + np.sum(new_train_data[new_labels == i])
            self.label_counts[i] = self.label_counts[i] + np.sum(new_labels == i)

        # Perform Likelihood Smoothing
        numerator = self.likelihoods + self.smoothing
        denominator = label_word_count + self.smoothing * dictionary_size
        self.likelihoods = numerator / denominator[:, np.newaxis]

        # Update Priors
        self.priors = self.label_counts / np.sum(self.label_counts)
