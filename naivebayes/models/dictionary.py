from collections import Counter


class Multinomial:

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.label_counts = {}
        self.labels_used = {}
        self.dictionary = set()
        self.likelihoods = {}
        self.empty_likelihoods = {}
        self.priors = {}

    # train_data - List of List of Words
    # labels - List of Labels
    def train(self, labels, train_data):

        self.labels_used = set(labels)

        # Setup Likelihood and Label Counts Dictionaries
        for label in self.labels_used:
            self.likelihoods[label] = Counter()
            self.label_counts[label] = 0

        # Calculate:
        #    Label Counts
        #    Dictionary
        #    Word Frequency By Label(Stored In Likelihoods)
        for i, label in enumerate(labels):
            self.label_counts[label] += 1

            for word in train_data[i]:
                self.likelihoods[label][word] += 1
                self.dictionary.add(word)

        # Get Size of Dictionary Being Used
        dictionary_size = len(self.dictionary)

        # Calculate Smoothed Likelihood and Priors
        for label in self.labels_used:
            # Number Of Words Belonging To Current Label
            label_word_count = sum(self.likelihoods[label].values())

            # Calculate Smoothed Likelihood For Each Word
            for word in self.likelihoods[label].keys():
                self.likelihoods[label][word] = (self.likelihoods[label][word] + self.smoothing) / \
                                                (label_word_count + self.smoothing * dictionary_size)

            # Value Used For Words In Dictionary That Do Not Belong To The Label
            self.empty_likelihoods[label] = self.smoothing / (label_word_count + self.smoothing * dictionary_size)

            # Prior Calculation
            self.priors[label] = self.label_counts[label] / len(labels)

    # test_data - List of List of Words
    # return_scores - Return Scores Used To Classify List of Words
    def predict(self, test_data, return_scores=False):

        predicted_labels = []
        all_scores = []

        # Make A Prediction For Each List of Words in test_data
        for word_list in test_data:
            scores = {}

            # Initialize By Putting Priors In First
            for label in self.labels_used:
                scores[label] = self.priors[label]

            # For Each Word Update Score With Smoothed Likelihood
            for word in word_list:
                # Skip Words Not In Dictionary
                if word in self.dictionary:
                    for label in self.labels_used:
                        scores[label] *= self.likelihoods[label].get(word, self.empty_likelihoods[label])

            # Append Label With Highest Score To Output
            predicted_labels.append(max(scores, key=scores.get))
            all_scores.append(scores)

        # Return Scores If Desired
        if return_scores:
            return predicted_labels, all_scores

        return predicted_labels

    # new_train_data - List of List of Words To Add To Model
    # new_labels - List of Labels To Add To Model
    def update(self, new_labels, new_train_data):

        # Get Size of Dictionary Being Used
        dictionary_size = len(self.dictionary)

        # Convert Original Smoothed Likelihoods Back Into Word Frequency Distributions
        for label in self.likelihoods:

            label_word_count = (self.smoothing - self.empty_likelihoods[label] * self.smoothing * dictionary_size) \
                               / self.empty_likelihoods[label]

            for word in self.likelihoods[label]:
                smoothed_likelihood = self.likelihoods[label][word]
                word_freq = smoothed_likelihood * (label_word_count + self.smoothing * dictionary_size) - self.smoothing

                self.likelihoods[label][word] = word_freq

        # Update The Word Frequency Distributions With New Training Data
        for i, word_list in enumerate(new_train_data):
            # Update Label Counts
            self.label_counts[new_labels[i]] += 1

            for word in word_list:
                # Assume User Wants To Add New Words To Dictionary
                if word not in self.dictionary:
                    self.dictionary.add(word)

                # Add 1 To Word Frequency
                self.likelihoods[new_labels[i]][word] += 1

        # Recalculate Smoothed Likelihoods and Priors
        for label in self.labels_used:
            # Number Of Words Belonging To Current Label
            label_word_count = sum(self.likelihoods[label].values())

            # Calculate Smoothed Likelihood For Each Word
            for word in self.likelihoods[label].keys():
                self.likelihoods[label][word] = (self.likelihoods[label][word] + self.smoothing) / \
                                                (label_word_count + self.smoothing * dictionary_size)

            # Value Used For Words In Dictionary That Do Not Belong To The Label
            self.empty_likelihoods[label] = self.smoothing / (label_word_count + self.smoothing * dictionary_size)

            # Prior Calculation
            self.priors[label] = self.label_counts[label] / sum(self.label_counts.values())

    def update_dictionary(self, new_dict):

        # Grab New Words
        new_words = new_dict.difference(self.dictionary)

        old_dictionary_size = len(self.dictionary)

        # Add New Words To Dictionary
        self.dictionary.update(new_words)

        # Convert Original Smoothed Likelihoods Back Into Word Frequency Distributions
        for label in self.likelihoods:

            label_word_count = (self.smoothing - self.empty_likelihoods[label] * self.smoothing * old_dictionary_size) \
                               / self.empty_likelihoods[label]

            for word in self.likelihoods[label]:
                smoothed_likelihood = self.likelihoods[label][word]
                word_freq = smoothed_likelihood * (label_word_count + self.smoothing * old_dictionary_size) \
                                   - self.smoothing

                self.likelihoods[label][word] = (word_freq + self.smoothing) / \
                                                (label_word_count + self.smoothing * len(self.dictionary))

            self.empty_likelihoods[label] = self.smoothing / (label_word_count + self.smoothing * len(self.dictionary))
