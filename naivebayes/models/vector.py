import numpy as np


class Multinomial:
    """ A vector based multinomial naive bayes model.

    Uses additive smoothing in the likelihood calculations. This model is able to have its dictionary updated and it can
    be trained online by feeding it more training data after being initially trained.

    This model is implemented using numpy. This model expects documents to be represented as a vector. The vectors used
    represent the frequency which words appear in the document. The columns of the vector will represent a specific
    word, so ensure all vectors use a consistent word to column mapping.

    Attributes:
        smoothing (float): Additive smoothing factor used in likelihood calculations.
        priors (ndarray): A 1D array representing the priors calculated for each label.
        likelihoods (ndarray): A 2D array representing the smoothed likelihoods calculated for each label.
        empty_likelihoods (ndarray): A 1D array representing the empty likelihoods calculated for each label. Used when
            providing more training data and when the dictionary is updated.
        label_counts (ndarray):A 1D array representing the number of times each label appeared in the training data.

    """

    def __init__(self, smoothing=1.0):
        """ Create a vector based multinomial naive bayes model.

        Args:
            smoothing (float, optional): Additive smoothing factor used in likelihood calculations. Defaults to
                1.0(Laplace Smoothing).

        """

        self.smoothing = smoothing
        self.priors = np.empty(0)
        self.likelihoods = np.empty(0)
        self.empty_likelihoods = np.empty(0)
        self.label_counts = np.empty(0)  # Could Store Just The Number Of Training Instances

    def train(self, labels, train_data):
        """ Trains the naive bayes model using the supplied training data.

        Args:
            labels (ndarray): A 1D array of labels. Labels are represented by consecutive integers starting at 0(0, 1,
                2, ...).
            train_data (ndarray): A 2D array of training data. Rows represent documents and columns represent word
                counts. Each column represents a specific word.

        """

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


    def predict(self, test_data):
        """ Predicts the label for the supplied text document data.

        Args:
            test_data (ndarray): A 2D array of data which which will be used to make predictions. Rows represent
                documents and columns represent word counts. Each column represents a specific word. Ensure columns
                correspond with the data used in training. If columns do not correspond, use update_dictionary() to
                update the model before making predictions.

        Returns:
            (tuple): tuple containing:
                predictions (list): A 1D array of predicted labels.
                scores (:obj:`list` of :obj:`dict`): A 2D array of scores used to make predictions.

            A 1D array of predicted labels, each row represents a supplied document. If specified a 2D array of scores
            will be returned.

        """

        # Convert To Log Space
        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)

        # Find Product Between Every Test Vector and Likelihood Vector THen Add Correct Prior To Each
        log_space_sums = log_priors + test_data.dot(log_likelihoods.T)

        # Find Label With Max Log Space Sum For Each Test Vector
        predicted_labels = np.argmax(log_space_sums, axis=1)

        return predicted_labels, np.exp(log_space_sums)


    def update(self, new_labels, new_train_data):
        """ Updates the model using more training data.

        Takes a batch of new training data and updates the model. This allows the model to be used for online learning.

        Args:
            new_labels (ndarray): A 1D array of labels. Labels are represented by integers.
            new_train_data (ndarray): A 2D array of training data. Rows represent documents and columns represent word
                counts. Each column represents a specific word.

        """

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
        """ Changes the mapping between words to its representative column.

        This will change the order and dimensions of the data used by the model. If you want to introduce new words to
        your model this will be used to modify the vectors the model uses. You can also use this to remove words from
        your model. If you use this method, make sure that your vectors supplied to update() and predict() matches the
        new mapping.

        Args:
            old_map (:obj:`list` of :obj:`int`): The map from words to their representative column in the original
                training data.
            new_map (:obj:`list` of :obj:`int`): The new map from words to their representative column.

        """

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

        # If Word From Old Map Is Note Used Update Label Word Counts
        for word in old_map:
            if word not in new_map:
                label_word_count = label_word_count - self.likelihoods[:, old_map[word]]

        self.likelihoods = new_likelihoods

        # Get Size of The New Dictionary
        new_dictionary_size = self.likelihoods.shape[1]

        # Perform Likelihood Smoothing
        numerator = self.likelihoods + self.smoothing
        denominator = (label_word_count + self.smoothing * new_dictionary_size)[:, np.newaxis]
        self.likelihoods = numerator / denominator

        # Update Empty Likelihood Values
        self.empty_likelihoods = self.smoothing / (label_word_count + self.smoothing * new_dictionary_size)
