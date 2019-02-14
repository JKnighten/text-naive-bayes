from collections import Counter


class Multinomial:

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.label_counts = {}
        self.labels_used = {}
        self.dictionary = set()
        self.likelihoods = {}
        self.empty_likelihood = {}
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

        # Calculate Smoothed Likelihood and Priors
        for label in self.labels_used:

            # Number Of Words Belonging To Current Label
            label_word_count = sum(self.likelihoods[label].values())

            for word in self.likelihoods[label].keys():
                self.likelihoods[label][word] = (self.likelihoods[label][word] + self.smoothing) / \
                                                (label_word_count + self.smoothing * len(self.dictionary))

            self.empty_likelihood[label] = self.smoothing / (label_word_count + self.smoothing * len(self.dictionary))

            self.priors[label] = self.label_counts[label] / len(labels)

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
                        scores[label] *= self.likelihoods[label].get(word, self.empty_likelihood[label])

            # Append Label With Highest Score To Output
            predicted_labels.append(max(scores, key=scores.get))
            all_scores.append(scores)

        # Return Scores If Desired
        if return_scores:
            return predicted_labels, all_scores

        return predicted_labels
