import unittest
import numpy as np
from naivebayes.models.vector import Multinomial


class TestVectorNaiveBayes(unittest.TestCase):

    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

        # Base Sports Data As Vectors
        self.train_data_sports = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
        self.labels_sports = np.array([0, 1, 0, 0, 1])
        self.a_very_close_game = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        # Correct Model Estimated Model Parameters
        self.correct_likelihoods = np.array([[3/25, 2/25, 3/25, 2/25, 3/25, 2/25, 2/25, 2/25, 1/25, 1/25, 1/25, 1/25,
                                              1/25, 1/25],
                                             [2/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 2/23, 3/23, 3/23, 2/23,
                                              2/23, 2/23]])
        self.correct_empty_likelihoods = np.array([1/25, 1/23])
        self.correct_label_count = np.array([3, 2])
        self.correct_priors = np.array([3/5, 2/5])
        self.correct_a_very_close_game_score = np.array([(3/25) * (2/25) * (1/25) * (3/25) * (3/5),
                                                         (2/23) * (1/23) * (2/23) * (1/23) * (2/5)])

        # Data For Extending The Dictionary
        self.sports_map = {"a": 0, "great": 1, "game": 2, "very": 3, "clean": 4,  "match": 5, "but": 6,
                           "forgettable": 7, "the": 8, "election": 9, "was": 10, "over": 11, "it": 12, "close": 13}
        self.sports_map_extra = self.sports_map.copy()
        self.sports_map_extra["test"] = 14
        self.a_very_close_game_extra = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

        self.correct_extended_likelihoods = np.array([[3/26, 2/26, 3/26, 2/26, 3/26, 2/26, 2/26, 2/26, 1/26, 1/26, 1/26,
                                                       1/26, 1/26, 1/26, 1/26],
                                                      [2/24, 1/24, 1/24, 1/24, 1/24, 1/24, 1/24, 1/24, 2/24, 3/24, 3/24,
                                                       2/24, 2/24, 2/24, 1/24]])
        self.correct_extended_empty_likelihoods = np.array([1/26, 1/24])
        self.correct_extended_a_very_close_game_score = np.array([(3/26) * (2/26) * (1/26) * (3/26) * (3/5),
                                                                  (2/24) * (1/24) * (2/24) * (1/24) * (2/5)])

        # Data For Shortening The Dictionary
        self.sports_map_shortened = {"a": 0, "great": 1, "very": 2, "clean": 3,  "match": 4, "but": 5, "forgettable": 6,
                                     "the": 7, "election": 8, "was": 9, "over": 10, "it": 11, "close": 12}
        self.a_very_close_game_short = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.correct_shortened_likelihoods = np.array([[3/22, 2/22, 2/22, 3/22, 2/22, 2/22, 2/22, 1/22, 1/22, 1/22,
                                                       1/22, 1/22, 1/22],
                                                      [2/22, 1/22, 1/22, 1/22, 1/22, 1/22, 1/22, 2/22, 3/22, 3/22,
                                                       2/22, 2/22, 2/22]])
        self.correct_shortened_empty_likelihoods = np.array([1/22, 1/22])
        self.correct_shortened_a_very_close_game_score = np.array([(3/22) * (2/22) * (1/22) * (3/5),
                                                                  (2/22) * (1/22) * (2/22) * (2/5)])

    # Check Correct Correct Model and Prediction Is Made
    def test_nb_vect_correct_prediction_and_model(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        prediction, _ = model.predict(self.a_very_close_game)

        # Correct Model
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_likelihoods)

        # Correct Prediction Output
        self.assertEqual(prediction, 0)

    # Check The Correct Score Is Returned
    def test_nb_vect_correct_score_output(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Score Output
        self.assertIsNotNone(score)
        np.testing.assert_array_almost_equal(score[0], self.correct_a_very_close_game_score)

    # Check That The Model Parameters Are Updated After Adding More Training Data
    def test_nb_vect_correct_add_more_training_data(self):
        model = Multinomial()
        model.train(self.labels_sports[0:4], self.train_data_sports[0:4])
        model.update(self.labels_sports[4], self.train_data_sports[4])
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Model Parameter Updates
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_likelihoods)

        # Correct New Scores
        self.assertIsNotNone(score)
        np.testing.assert_array_almost_equal(score[0], self.correct_a_very_close_game_score)

        # Correct Prediction
        self.assertEqual(prediction[0], 0)

    # Check That The Model Parameters Are Updated After Updating The Dictionary
    def test_nb_vect_update_correct_extend_dictionary(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        model.update_dictionary(self.sports_map, self.sports_map_extra)
        prediction, score = model.predict(self.a_very_close_game_extra)

        # Correct Model Parameter Updates
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_extended_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_extended_likelihoods)

        # Correct Scores
        self.assertEqual(prediction[0], 0)
        self.assertIsNotNone(score)
        np.testing.assert_array_almost_equal(score[0], self.correct_extended_a_very_close_game_score)

        # Correct Prediction
        self.assertEqual(prediction[0], 0)

    def test_nb_vect_update_correct_shorten_dictionary(self):
        model = Multinomial()
        model.train(self.labels_sports, self.train_data_sports)
        model.update_dictionary(self.sports_map, self.sports_map_shortened)
        prediction, score = model.predict(self.a_very_close_game_short)

        # Correct Model Parameter Updates
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_shortened_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_shortened_likelihoods)

        # Correct Scores
        self.assertEqual(prediction[0], 0)
        self.assertIsNotNone(score)
        np.testing.assert_array_almost_equal(score[0], self.correct_shortened_a_very_close_game_score)

        # Correct Prediction
        self.assertEqual(prediction[0], 0)
