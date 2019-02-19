import numpy as np


class Multinomial:

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.priors = np.empty(0)
        self.likelihoods = np.empty(0)
        self.empty_likelihoods = np.empty(0)
        self.label_counts = np.empty(0)  # Could Store Just The Number Of Training Instances

    def train(self, labels, train_data):

        # Get Size of Dictionary Being Used
        dictionary_size = train_data.shape[1]

        # Count The Number of Occurrences For Each Label
        self.label_counts = np.bincount(labels)

        # Create Priors Vector
        self.priors = self.label_counts / np.sum(self.label_counts)

        # Create Empty Vectors To Store Likelihoods
        self.likelihoods = np.zeros((self.priors.shape[0], dictionary_size))

        # Start By Finding The Frequency Which Each Words Occurs In Each Label
        for i in range(self.likelihoods.shape[0]):
            self.likelihoods[i, :] = np.sum(train_data[labels == i], axis=0)

        # Count The Total Number Of Words That Appear In Each Label
        label_word_count = np.sum(self.likelihoods, axis=1)

        # Likelihood Smoothing
        numerator = self.likelihoods + self.smoothing
        denominator = (label_word_count + self.smoothing * dictionary_size)[:, np.newaxis]
        self.likelihoods = numerator / denominator

        # Store Empty Likelihood Values - Values Used When A Word Does Not Appear In A Label
        self.empty_likelihoods = np.zeros((self.priors.shape[0], 1))
        self.empty_likelihoods = self.smoothing / (label_word_count + self.smoothing * dictionary_size)

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

        # Get Size of Dictionary Being Used
        dictionary_size = self.likelihoods.shape[1]

        # Find The Total Number Of Words That Appear In Each Label
        # Use Basic Algebra To Rework The Smoothed Likelihood For Empty Word Counts
        label_word_count = (self.smoothing - self.empty_likelihoods * self.smoothing * dictionary_size) \
                           / self.empty_likelihoods

        # Find The Frequency Which Each Words Occurs In Each Label
        self.likelihoods = self.likelihoods * (label_word_count[:, np.newaxis] + self.smoothing * dictionary_size) \
                           - self.smoothing

        # Add The New Training Data To The Model
        # Updates That Must Be Made:
        #   Word Frequency For Each Label - Currently Stored in self.likelihoods
        #   Label Counts
        #   Number Of Words In Each Label
        for i in range(self.likelihoods.shape[0]):
            self.likelihoods[i, :] = self.likelihoods[i, :] + np.sum(new_train_data[new_labels == i], axis=0)
            label_word_count[i] = label_word_count[i] + np.sum(new_train_data[new_labels == i])
            self.label_counts[i] = self.label_counts[i] + np.sum(new_labels == i)

        # Perform Likelihood Smoothing
        numerator = self.likelihoods + self.smoothing
        denominator = (label_word_count + self.smoothing * dictionary_size)[:, np.newaxis]
        self.likelihoods = numerator / denominator

        # Update Priors
        self.priors = self.label_counts / np.sum(self.label_counts)

        # Update Empty Likelihood Values
        self.empty_likelihoods = self.smoothing / (label_word_count + self.smoothing * dictionary_size)

    def update_dictionary(self, old_map, new_map):

        # Get Size of Dictionary Being Used
        dictionary_size = self.likelihoods.shape[1]

        # Find The Total Number Of Words That Appear In Each Label
        # Use Basic Algebra To Rework The Smoothed Likelihood For Empty Word Counts
        label_word_count = (self.smoothing - self.empty_likelihoods * self.smoothing * dictionary_size) \
                           / self.empty_likelihoods

        # Find The Frequency Which Each Words Occurs In Each Label
        self.likelihoods = self.likelihoods * (label_word_count[:, np.newaxis] + self.smoothing * dictionary_size) \
                           - self.smoothing

        # Create Storage For New Likelihood Vectors
        new_likelihoods = np.zeros((self.likelihoods.shape[0], len(new_map.keys())))

        # Copy Likelihood Data For Any Word That Is In The New And Old Map
        for word in new_map:
            if word in old_map:
                new_likelihoods[:, new_map[word]] = self.likelihoods[:, old_map[word]]

        self.likelihoods = new_likelihoods

        # Get Size of The New Dictionary
        new_dictionary_size = self.likelihoods.shape[1]

        # Perform Likelihood Smoothing
        numerator = self.likelihoods + self.smoothing
        denominator = (label_word_count + self.smoothing * new_dictionary_size)[:, np.newaxis]
        self.likelihoods = numerator / denominator

        # Update Empty Likelihood Values
        self.empty_likelihoods = self.smoothing / (label_word_count + self.smoothing * new_dictionary_size)
