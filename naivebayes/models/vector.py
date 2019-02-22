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

        # Catch Invalid Smoothing Factor
        if smoothing < 0:
            raise ValueError("Smoothing factor cannot be less than 0: it was set to {}".format(smoothing))

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

        # Check Labels Is A Numpy Array
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a numpy array: it was a {}".format(type(labels)))

        # Check Labels Is 1D
        if len(labels.shape) != 1:
            raise ValueError("The labels array must be 1D: the shape was {}".format(labels.shape))

        # Check Train Data Is Numpy Array
        if not isinstance(train_data, np.ndarray):
            raise TypeError("Train data must be a numpy array: it was a {}".format(type(train_data)))

        # Check Train Data Is 2D
        if len(train_data.shape) != 2:
            raise ValueError("The train data array must be 2D: the shape was {}".format(train_data.shape))

        # Check That Every Document In train_data Has A Label Associated With It
        if labels.shape[0] != train_data.shape[0]:
            raise ValueError("The number of labels and documents are different: {} labels and {} documents"
                             .format(labels.shape[0], train_data.shape[0]))

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

        # Check Model Is Trained Before Performing Prediction
        if self.likelihoods.size == 0:
            raise ValueError("The model needs to be trained before making predictions")

        # Check Test Data Is A Numpy Array
        if not isinstance(test_data, np.ndarray):
            raise TypeError("Test data must be a numpy array: it was a {}".format(type(test_data)))

        # Check Test Data Is 2D
        if len(test_data.shape) != 2:
            raise ValueError("The test data array must be 2D: the shape was {}".format(test_data.shape))

        # Check Test Data Has The Same Number Of Columns As The Trained Model
        if test_data.shape[1] != self.likelihoods.shape[1]:
            raise ValueError("Test data must have the same number of columns as data used in training: test data" +
                             "cols {} and training data cols {}".format(test_data.shape[1],self.likelihoods.shape[1]))

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

        # Check Model Is Trained Before Updating The Model
        if self.likelihoods.size == 0:
            raise ValueError("The model needs to be trained before updating the model")

        # Check Labels Is A Numpy Array
        if not isinstance(new_labels, np.ndarray):
            raise TypeError("New labels must be a numpy array: it was a {}".format(type(new_labels)))

        # Check Labels Is 1D
        if len(new_labels.shape) != 1:
            raise ValueError("The new labels array must be 1D: the shape was {}".format(new_labels.shape))

        # Check Train Data Is Numpy Array
        if not isinstance(new_train_data, np.ndarray):
            raise TypeError("New train data must be a numpy array: it was a {}".format(type(new_train_data)))

        # Check Train Data Is 2D
        if len(new_train_data.shape) != 2:
            raise ValueError("The new train data array must be 2D: the shape was {}".format(new_train_data.shape))

        # Check That Every Document In train_data Has A Label Associated With It
        if new_labels.shape[0] != new_train_data.shape[0]:
            raise ValueError("The number of new labels and new documents are different: {} labels and {} documents"
                             .format(new_labels.shape[0], new_train_data.shape[0]))

        # Check New Train Data Has The Same Number Of Columns As The Trained Model
        if new_train_data.shape[1] != self.likelihoods.shape[1]:
            raise ValueError("New train data must have the same number of columns as data used in training: new " +
                             "training data cols {} and training data cols {}".format(new_train_data.shape[1],
                                                                                      self.likelihoods.shape[1]))

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

        # Check Model Is Trained Before Updating The Dictionary
        if self.likelihoods.size == 0:
            raise ValueError("The model needs to be trained before updating the dictionary")

        # Old Map Must Be A Dictionary
        if not isinstance(old_map, dict):
            raise TypeError("Old map must be of type dict: the type {} was given".format(type(old_map)))

        # Old Map Must Be A Dictionary With str Keys
        if set(map(type, old_map.keys())) != {str}:
            raise TypeError("Old maps keys must be of type str: the type {} was given"
                            .format(set(map(type, old_map.keys()))))

        # Old Map Must Be A Dictionary With int Values
        if set(map(type, old_map.values())) != {int}:
            raise TypeError("Old maps values must be of type int: the type {} was given"
                            .format(set(map(type, old_map.values()))))

        # Old Map Must Have The Same Number Of Keys As The Number Of Cols In Training Data
        if len(old_map.keys()) != self.likelihoods.shape[1]:
            raise ValueError("Old map should have the same numbers as keys as the number of cols in the training data:"
                             + "there were {} keys in old map and {} cols in the training data"
                             .format(len(old_map.keys()), self.likelihoods.shape[1]))

        # New Map Must Be A Dictionary
        if not isinstance(new_map, dict):
            raise TypeError("New map must be of type dict: the type {} was given".format(type(old_map)))

        # New Map Must Not Be Empty
        if len(new_map.keys()) == 0:
            raise ValueError("New map should not be empty")

        # New Map Must Be A Dictionary With str Keys
        if set(map(type, new_map.keys())) != {str}:
            raise TypeError("Old maps keys must be of type str: the type {} was given"
                            .format(set(map(type, old_map.keys()))))

        # New Map Must Be A Dictionary With int Values
        if set(map(type, new_map.values())) != {int}:
            raise TypeError("New maps values must be of type int: the type {} was given"
                            .format(set(map(type, new_map.keys()))))

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
