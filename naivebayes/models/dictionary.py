from functools import reduce


class Multinomial:

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.labels = None
        self.train_data = None
        self.dictionary = None
        self.smoothed_params = {}

    def train(self, labels, train_data):
        self.labels = labels
        self.train_data = train_data

        # Create Global Dictionary
        self.dictionary = \
            reduce(lambda key_a, key_b: set(self.train_data[key_a].keys()).union(set(self.train_data[key_b].keys())),
                   self.train_data)

        # Smooth Params
        for label in train_data:
            label_word_count = reduce(lambda key_a, key_b: self.train_data[label][key_a] + self.train_data[label][key_b]
                                      , self.train_data[label])

            for word in train_data[label]:
                self.smoothed_params[label][word] = (self.train_data[label][word] + self.smoothing) /\
                                                    (label_word_count + self.smoothing*len(self.dictionary))

    def predict(self, test_data):
        pass
