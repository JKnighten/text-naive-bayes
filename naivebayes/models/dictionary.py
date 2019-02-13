from functools import reduce


class Multinomial:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.label_counts = None
        self.train_data = None
        self.dictionary = None
        self.smoothed_likelihood = {}
        self.smoothed_likelihood_empty = {}
        self.label_prior = {}

    def train(self, label_counts, train_data):
        self.label_counts = label_counts
        self.train_data = train_data

        # Create Dictionary Using All Labels
        self.dictionary = \
            reduce(lambda key_a, key_b: set(self.train_data[key_a].keys()).union(set(self.train_data[key_b].keys())),
                   self.train_data)

        # Get Total Number Of Training Instances
        numb_training_instances = sum(self.label_counts.values())

        # Calculate Likelihoods And Apply Smoothing
        for label in self.train_data:

            # Number Of Words Belonging To Current Label
            label_word_count = sum(self.train_data[label].values())

            # Calculate Smoothed Likelihood For Each Word In Label
            label_likelihoods = {}
            for word in train_data[label]:
                label_likelihoods[word] = (self.train_data[label][word] + self.smoothing) / \
                                          (label_word_count + self.smoothing * len(self.dictionary))

            self.smoothed_likelihood[label] = label_likelihoods
            self.smoothed_likelihood_empty[label] = self.smoothing / \
                                                    (label_word_count + self.smoothing * len(self.dictionary))

            # Label Prior Calculation
            self.label_prior[label] = self.label_counts[label] / numb_training_instances

    def predict(self, test_data, return_scores = False):

        predicted_labels = []
        all_scores = []

        # Make A Prediction For Each List of Words in test_data
        for word_list in test_data:

            scores = {}

            # Initialize By Putting Priors In First
            for label in self.train_data.keys():
                scores[label] = self.label_prior[label]

            # For Each Word Update Score With Smoothed Likelihood
            for word in word_list:
                # Skip Words Not In Dictionary
                if word in self.dictionary:
                    for label in self.train_data.keys():
                        scores[label] *= self.smoothed_likelihood[label].get(word,
                                                                             self.smoothed_likelihood_empty[label])
            # Append Label With Highest Score To Output
            predicted_labels.append(max(scores, key=scores.get))
            all_scores.append(scores)

        # Return Scores If Desired
        if return_scores:
            return predicted_labels, all_scores

        return predicted_labels
