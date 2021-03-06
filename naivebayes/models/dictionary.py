from collections import Counter


class Multinomial:
    """ A dictionary based multinomial naive bayes model.

    Uses additive smoothing for the likelihood calculation. This model is able to have its dictionary updated and it can
    be trained online by feeding it more training data after being initially trained.

    This model is implemented in pure python. The data supplied to the model requires little pre-processing. Simply
    turn each document into a list of words and then supply the lists to the training method.

    Attributes:
            smoothing (float): Additive smoothing factor used in likelihood calculations.
            label_counts (:obj:`dict` of :obj:`int`) The number of times each label appeared in the training data.
            labels_used (:obj:`set`): The set of labels used in the model
            dictionary (:obj:`set`): The set of words used in the model
            likelihoods (:obj:`dict` of :obj:`dict` of :obj:`float`): Smoothed likelihoods calculated for each label.
            empty_likelihoods (:obj:`dict` of :obj:`float`): The empty likelihood calculated for each label. Used when:
                a word has frequency of zero, providing more training data, and the dictionary is updated.
            priors (:obj:`dict` of :obj:`float`): Priors calculated for each label.

    """

    def __init__(self, smoothing=1.0):
        """ Create a dictionary based multinomial naive bayes model.

        Args:
            smoothing (float, optional): Additive smoothing factor used in likelihood calculations. Defaults to
                1.0(Laplace Smoothing).

        """
        # Catch Invalid Smoothing Factor
        if smoothing < 0:
            raise ValueError("Smoothing factor cannot be less than 0: it was set to {}".format(smoothing))

        self.smoothing = smoothing
        self.label_counts = {}
        self.labels_used = {}
        self.dictionary = set()
        self.likelihoods = {}
        self.empty_likelihoods = {}
        self.priors = {}

    def train(self, labels, train_data):
        """ Trains the naive bayes model using the supplied training data.

        Args:
            labels (:obj:`list` of :obj:`str`): An list of document labels.
            train_data (:obj:`list` of :obj:`list` of :obj:`str`): A collection of list of words. Each list of words
                represents a document.

        """

        # Check Labels Are In A List
        if not isinstance(labels, list):
            raise TypeError("Labels must be in a list: it was a {}".format(type(labels)))

        # Check That Labels Are Strings Or Int
        if sum([isinstance(label, (str, int)) for label in labels]) != len(labels):
            raise TypeError("The labels must be strs or ints: the shape was {}".format(labels))

        # Check Train Data Is Type List
        if not isinstance(train_data, list):
            raise TypeError("Train data must be in a list: it was a {}".format(type(train_data)))

        # Check Inner Train Data Is Type List
        if sum([isinstance(doc, list) for doc in train_data]) != len(train_data):
            raise ValueError("The data in train data must be lists: the internal type were {}".format(train_data))

        # Check Most Inner Train Data Is Type Strings
        if sum([sum([isinstance(word, str) for word in doc]) == len(doc) for doc in train_data]) != len(train_data):
            raise ValueError("The lists inside of train data must contain strs: the internal type were {}"
                             .format(train_data))

        # Check That Every Document In train_data Has A Label Associated With It
        if len(labels) != len(train_data):
            raise ValueError("The number of labels and documents are different: {} labels and {} documents"
                             .format(len(labels), len(train_data)))

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

    def predict(self, test_data):
        """ Predicts the labels for a collection of documents.

        Args:
            test_data (:obj:`list` of :obj:`list` of :obj:`str`): A collection of documents whose label will be
                predicted by using the model.

        Returns:
            (tuple): tuple containing:
                predictions (list): A list of predicted labels.
                scores (:obj:`list` of :obj:`dict`): The scores used to make the predictions.

        """
        if len(self.likelihoods) == 0:
            raise ValueError("The model needs to be trained before making predictions")

        # Check Test Data Is In A List
        if not isinstance(test_data, list):
            raise TypeError("Train data must be in a list: it was a {}".format(type(test_data)))

        # Check Inner Test Data Is Lists
        if sum([isinstance(doc, list) for doc in test_data]) != len(test_data):
            raise ValueError("The data in train data must be lists: the internal type were {}".format(test_data))

        # Check Most Inner Test Data Is strs
        if sum([sum([isinstance(word, str) for word in doc]) == len(doc) for doc in test_data]) != len(test_data):
            raise ValueError("The lists inside of train data must contain strs: the internal type were {}"
                             .format(test_data))

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

        return predicted_labels, all_scores

    def update(self, new_labels, new_train_data):
        """ Updates the model using more training data.

        Any new words encountered in the additional training data will be added to the models dictionary.

        Args:
            new_labels (:obj:`list` of :obj:`str`): An list of document labels.
            new_train_data (:obj:`list` of :obj:`list` of :obj:`str`): A collection of list of words. Each list of words
                represents a document.

        """

        # Check Model Is Trained Before Updating The Model
        if len(self.likelihoods) == 0:
            raise ValueError("The model needs to be trained before updating the model")

        # Check New Labels Are In A List
        if not isinstance(new_labels, list):
            raise TypeError("Labels must be in a list: it was a {}".format(type(new_labels)))

        # Check That New Labels Are Strings Or Int
        if sum([isinstance(label, (str, int)) for label in new_labels]) != len(new_labels):
            raise TypeError("The labels must be strings or ints: the shape was {}".format(new_labels))

        # Check New Train Data Is Type List
        if not isinstance(new_train_data, list):
            raise TypeError("Train data must be in a list: it was a {}".format(type(new_train_data)))

        # Check Inner New Train Data Is Type List
        if sum([isinstance(doc, list) for doc in new_train_data]) != len(new_train_data):
            raise ValueError("The data in train data must be lists: the internal type were {}".format(new_train_data))

        # Check Most Inner New Train Data Is Type Strings
        if sum([sum([isinstance(word, str) for word in doc]) == len(doc) for doc in new_train_data]) != \
                len(new_train_data):
            raise ValueError("The lists inside of train data must contain strings: the internal type were {}"
                             .format(new_train_data))

        # Check That Every Document In New Train Data Has A Label Associated With It
        if len(new_labels) != len(new_train_data):
            raise ValueError("The number of labels and documents are different: {} labels and {} documents"
                             .format(len(new_labels), len(new_train_data)))

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

        # Get Size of Dictionary Being Used
        new_dictionary_size = len(self.dictionary)

        # Recalculate Smoothed Likelihoods and Priors
        for label in self.labels_used:
            # Number Of Words Belonging To Current Label
            label_word_count = sum(self.likelihoods[label].values())

            # Calculate Smoothed Likelihood For Each Word
            for word in self.likelihoods[label].keys():
                self.likelihoods[label][word] = (self.likelihoods[label][word] + self.smoothing) / \
                                                (label_word_count + self.smoothing * new_dictionary_size)

            # Value Used For Words In Dictionary That Do Not Belong To The Label
            self.empty_likelihoods[label] = self.smoothing / (label_word_count + self.smoothing * new_dictionary_size)

            # Prior Calculation
            self.priors[label] = self.label_counts[label] / sum(self.label_counts.values())

    def update_dictionary(self, new_dict):
        """ Add/Remove words to/from the dictionary.

        Args:
            new_dict (:obj:'set' of :obj:`str`): a set of new words to be added or removed from the dictionary.

        """

        # Check Model Is Trained Before Updating The Dictionary
        if len(self.likelihoods) == 0:
            raise ValueError("The model needs to be trained before updating the dictionary")

        # Check If New Dictionary Is Of Type Set
        if not isinstance(new_dict, set):
            raise TypeError("The new dictionary needs to be of type set: type {} was found".format(type(new_dict)))

        # Check If New Dictionary Is Empty
        if len(new_dict) == 0:
            raise ValueError("The new dictionary cannot be empty")

        # Check If New Dictionary Contains Strings
        if set(type(word) for word in new_dict) != {str}:
            raise TypeError("The new dictionary must be a set of strings: a set of type {} was given"
                            .format(type(set(isinstance(word, str) for word in new_dict))))

        # Grab Dictionary Sizes
        new_dict_size = len(new_dict)
        old_dict_size = len(self.dictionary)

        word_freqs = {}
        label_word_counts = {}
        # Get Original Word Frequencies For Words That Are In Both Old And New Dictionary
        # Calculate Label Word Counts
        for label in self.likelihoods:

            label_word_count = (self.smoothing - self.empty_likelihoods[label] * self.smoothing * old_dict_size) \
                               / self.empty_likelihoods[label]

            word_freqs[label] = {}

            for word in self.likelihoods[label]:
                # Only Calculate Word Frequency If Words Are In New Dictionary
                if word in new_dict:
                    smoothed_likelihood = self.likelihoods[label][word]
                    word_freq = smoothed_likelihood * (label_word_count + self.smoothing * old_dict_size) \
                                       - self.smoothing

                    word_freqs[label][word] = word_freq

            label_word_counts[label] = sum(word_freqs[label].values())

        self.likelihoods = {}
        # Apply Smoothing To Words
        # Recalculate New Empty Likelihood Values
        for label in self.labels_used:
            self.likelihoods[label] = {}

            for word in word_freqs[label]:
                self.likelihoods[label][word] = (word_freqs[label][word] + self.smoothing) / \
                                                 (label_word_counts[label] + self.smoothing * new_dict_size)

            self.empty_likelihoods[label] = self.smoothing / (label_word_counts[label] + self.smoothing * new_dict_size)

        self.dictionary = new_dict
