import unittest
import numpy as np
from naivebayes.models.vector import Multinomial


class TestMultinomialNaiveBayes(unittest.TestCase):

    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

        # Word -> Index Map
        self.sports_map = {"a": 0, "great": 1, "game": 2, "very": 3, "clean": 4,  "match": 5, "but": 6,
                           "forgettable": 7, "the": 8, "election": 9, "was": 10, "over": 11, "it": 12, "close": 13}

        self.train_data_sports = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],

                                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
        self.labels_sports = np.array([0, 1, 0, 0, 1])

        self.a_very_close_game = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    # Check Correct Prediction Is Made
    def test_nb_vect_prediction(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        prediction = model.predict(self.a_very_close_game)

        self.assertEqual(prediction, 0)

    # Check Score is Returned And Score Is Correct
    def test_nb_vect_scores(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        prediction, score = model.predict(self.a_very_close_game, return_scores=True)

        self.assertIsNotNone(score)
        np.testing.assert_array_almost_equal(score, np.array([[2.76e-05, 5.72e-06]]))
