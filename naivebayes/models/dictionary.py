from functools import reduce


class Multinomial:

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.labels_counts = None
        self.train_data = None
        self.dictionary = None
        self.smoothed_params = {}
        self.smoothed_params_empty = {}
        self.label_probs = {}

    def train(self, label_counts, train_data):
        self.labels_counts = label_counts
        self.train_data = train_data

        # Create Global Dictionary
        self.dictionary = \
            reduce(lambda key_a, key_b: set(self.train_data[key_a].keys()).union(set(self.train_data[key_b].keys())),
                   self.train_data)

        # Get Total Number Of Training Instances
        numb_training_instances = reduce(lambda key_a, key_b: self.labels_counts[key_a] + self.labels_counts[key_b],
                           self.labels_counts)

        # Smooth Params
        for label in train_data:

            # Number Of Times Label Occurred
            label_word_count = self.labels_counts[label]

            # Calculate Smoothed Param For Each Word In Label
            label_smoothed_params = {}
            for word in train_data[label]:
                label_smoothed_params[word] = (self.train_data[label][word] + self.smoothing) /\
                                                    (label_word_count + self.smoothing*len(self.dictionary))

            self.smoothed_params[label] = label_smoothed_params
            self.smoothed_params_empty[label] = self.smoothing / (label_word_count + self.smoothing*len(self.dictionary))

            self.label_probs[label] = self.labels_counts[label] / numb_training_instances

    def predict(self, test_data):

        predicted_labels = []

        for word_list in test_data:

            scores = {}

            for label in self.train_data.keys():
                scores[label] = self.label_probs[label]

            for word in word_list:

                if word in self.dictionary:
                    for label in self.train_data.keys():
                        scores[label] *= self.smoothed_params[label].get(word, self.smoothed_params_empty[label])

            predicted_labels.append(max(scores, key=scores.get))

        return predicted_labels
