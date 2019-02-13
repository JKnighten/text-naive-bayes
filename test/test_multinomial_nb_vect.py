import unittest

from naivebayes.models.vector import Multinomial
import numpy as np


class TestDistanceMetrics(unittest.TestCase):
    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
        # Word Map
        # a - 0
        # great - 1
        # game - 2
        # very - 3
        # clean - 4
        # match - 5
        # but - 6
        # forgettable - 7
        # the - 8
        # election - 9
        # was - 10
        # over - 11
        # it - 12
        # close - 13
        self.train_data_sports = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
        self.labels_sports = np.array([0, 1, 0, 0, 1])

    # Sports Data
    def test_nb_vect_sports(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        prediction = model.predict(np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))

        self.assertEqual(prediction, 0)
