import unittest
import numpy as np
from naivebayes.models.vector import Multinomial


class TestVectorMultinomialNaiveBayes(unittest.TestCase):

    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

        # Base Sports Data As Vectors
        self.sports_data = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
        self.sports_labels = np.array([0, 1, 0, 0, 1])
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

    def test_constructor_invalid_smoothing_factor(self):
        self.assertRaises(ValueError, Multinomial, smoothing=-1.0)

    def test_train_number_of_labels_and_docs_differ(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.train, self.sports_labels[0:4], self.sports_data)

    def test_train_labels_not_numpy_array(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, list(self.sports_labels), self.sports_data)

    def test_train_labels_not_1D(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.train, self.sports_labels[:, np.newaxis], self.sports_data)

    def test_train_training_data_not_numpy_array(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, self.sports_labels, list(self.sports_data))

    def test_train_training_data_not_2D(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.train, self.sports_labels, self.sports_data[:, np.newaxis])

    def test_train_model_params(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)

        # Correct Model
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_likelihoods)

    def test_predict_test_data_not_numpy_array(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.predict, list(self.a_very_close_game))

    def test_predict_test_data_not_2D(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.predict, np.array(self.a_very_close_game[0]))

    def test_predict_test_data_not_same_dim_as_training_data(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.predict, self.a_very_close_game[:, 0:2])

    def test_predict_called_before_training(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.predict, self.a_very_close_game)

    def test_predict_prediction_and_score(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Prediction Output
        self.assertEqual(prediction, 0)

        # Correct Score Output
        np.testing.assert_array_almost_equal(score[0], self.correct_a_very_close_game_score)

    def test_update_number_of_labels_and_docs_differ(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.update, self.sports_labels[0:4], self.sports_data)

    def test_update_labels_not_numpy_array(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update, list(self.sports_labels), self.sports_data)

    def test_update_labels_not_1D(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.update, self.sports_labels[:, np.newaxis], self.sports_data)

    def test_update_new_train_data_not_numpy_array(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update, self.sports_labels, list(self.sports_data))

    def test_update_new_train_data_not_2D(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.update, self.sports_labels, self.sports_data[:, np.newaxis])

    def test__update_new_train_data_not_same_dim_as_training_data(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.update, self.sports_labels, self.sports_data[:, 0:2])

    def test_update_called_before_training(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.update, self.sports_labels, self.sports_data)

    def test_update_add_more_training_data(self):
        model = Multinomial()
        model.train(self.sports_labels[0:4], self.sports_data[0:4])
        model.update(np.array([self.sports_labels[4]]), np.array([self.sports_data[4]]))
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Model Parameter Updates
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_likelihoods)

        # Correct New Scores
        np.testing.assert_array_almost_equal(score[0], self.correct_a_very_close_game_score)

        # Correct Prediction
        self.assertEqual(prediction[0], 0)

    def test_update_dictionary_old_map_not_dict(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update_dictionary, list(self.sports_map), self.sports_map_shortened)

    def test_update_dictionary_old_map_wrong_keys(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.sports_map[0] = 1
        self.assertRaises(TypeError, model.update_dictionary, self.sports_map, self.sports_map_shortened)

    def test_update_dictionary_old_map_wrong_values(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.sports_map["test"] = "fun"
        self.assertRaises(TypeError, model.update_dictionary, self.sports_map, self.sports_map_shortened)

    def test_update_dictionary_old_map_not_same_size_as_training_data(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.sports_map["test"] = 15
        self.assertRaises(ValueError, model.update_dictionary, self.sports_map, self.sports_map_shortened)

    def test_update_dictionary_new_map_not_dict(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update_dictionary, self.sports_map, list(self.sports_map_shortened))

    def test_update_dictionary_new_map_wrong_keys(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.sports_map_shortened[0] = 1
        self.assertRaises(TypeError, model.update_dictionary, self.sports_map, self.sports_map_shortened)

    def test_update_dictionary_new_map_wrong_values(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.sports_map_shortened["test"] = "fun"
        self.assertRaises(TypeError, model.update_dictionary, self.sports_map, self.sports_map_shortened)

    def test_update_dictionary_new_map_is_empty(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.update_dictionary, self.sports_map, {})

    def test_update_dictionary_called_before_training(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.update_dictionary, self.sports_map, self.sports_map_extra)

    def test_update_dictionary_extend_dictionary(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        model.update_dictionary(self.sports_map, self.sports_map_extra)
        prediction, score = model.predict(self.a_very_close_game_extra)

        # Correct Model Parameter Updates
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_extended_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_extended_likelihoods)

        # Correct Scores
        np.testing.assert_array_almost_equal(score[0], self.correct_extended_a_very_close_game_score)

        # Correct Prediction
        self.assertEqual(prediction[0], 0)

    def test_update_dictionary_shorten_dictionary(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        model.update_dictionary(self.sports_map, self.sports_map_shortened)
        prediction, score = model.predict(self.a_very_close_game_short)

        # Correct Model Parameter Updates
        np.testing.assert_array_almost_equal(model.priors, self.correct_priors)
        np.testing.assert_array_almost_equal(model.label_counts, self.correct_label_count)
        np.testing.assert_array_almost_equal(model.empty_likelihoods, self.correct_shortened_empty_likelihoods)
        np.testing.assert_array_almost_equal(model.likelihoods, self.correct_shortened_likelihoods)

        # Correct Scores
        np.testing.assert_array_almost_equal(score[0], self.correct_shortened_a_very_close_game_score)

        # Correct Prediction
        self.assertEqual(prediction[0], 0)
