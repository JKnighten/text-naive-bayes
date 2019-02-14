from functools import reduce


class Multinomial:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.label_counts = None
        self.labels_used = None
        self.dictionary = None
        self.likelihoods = {}
        self.empty_likelihood = {}
        self.priors = {}

    # TODO - Accept A List Of Labels Instead of Counts
    def train(self, label_counts, train_data):
        self.label_counts = label_counts
        self.labels_used = set(train_data.keys())

        # Create Dictionary Using All Labels
        self.dictionary = \
            reduce(lambda key_a, key_b: set(train_data[key_a].keys()).union(set(train_data[key_b].keys())), train_data)

        # Get Total Number Of Training Instances
        numb_training_instances = sum(self.label_counts.values())

        # Calculate Likelihoods And Apply Smoothing
        for label in self.labels_used:

            # Number Of Words Belonging To Current Label
            label_word_count = sum(train_data[label].values())

            # Calculate Smoothed Likelihood For Each Word In Label
            label_likelihoods = {}
            for word in train_data[label]:
                label_likelihoods[word] = (train_data[label][word] + self.smoothing) / \
                                          (label_word_count + self.smoothing * len(self.dictionary))

            self.likelihoods[label] = label_likelihoods
            self.empty_likelihood[label] = self.smoothing / \
                                           (label_word_count + self.smoothing * len(self.dictionary))

            # Label Prior Calculation
            self.priors[label] = self.label_counts[label] / numb_training_instances

    def predict(self, test_data, return_scores = False):

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
