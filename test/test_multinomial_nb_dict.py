import unittest

from naivebayes.models.dictionary import Multinomial


class TestDictionaryNaiveBayes(unittest.TestCase):

    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

        # Base Sports Data As List Of Words
        self.sports_data = [["a", "great", "game"],
                            ["the", "election", "was", "over"],
                            ["very", "clean",  "match"],
                            ["a", "clean", "but", "forgettable", "game"],
                            ["it", "was", "a", "close", "election"]]
        self.sports_labels = ["sport", "not sport", "sport", "sport", "not sport"]
        self.a_very_close_game = [["a", "very", "close", "game"]]

        # Correct Model Estimated Model Parameters
        self.correct_likelihoods = {"not sport": {'election': 3/23, 'was': 3/23, 'over': 2/23, 'the': 2/23,
                                                  'close': 2/23, 'it': 2/23, 'a': 2/23},
                                    "sport": {'game': 3/25, 'a': 3/25, 'clean': 3/25, 'great': 2/25,
                                              'forgettable': 2/25, 'very': 2/25, 'match': 2/25, 'but': 2/25}}
        self.correct_empty_likelihoods = {'sport': 1/25, 'not sport': 1/23}
        self.correct_label_count = {'sport': 3, 'not sport': 2}
        self.correct_priors = {'sport': 3/5, 'not sport': 2/5}
        self.correct_a_very_close_game_score = {"not sport": (2/23) * (1/23) * (2/23) * (1/23) * (2/5),
                                                "sport": (3/25) * (2/25) * (1/25) * (3/25) * (3/5)}

        # Extended Data
        self.dictionary = {'game', 'was', 'a', 'very', 'the', 'great', 'forgettable', 'match', 'clean', 'but',
                           'election', 'it', 'close', 'over'}
        self.extended_dictionary = self.dictionary.copy()
        self.extended_dictionary.add("test")
        self.correct_extended_likelihoods = {"not sport": {'election': 3/24, 'was': 3/24, 'over': 2/24, 'the': 2/24,
                                                           'close': 2/24, 'it': 2/24, 'a': 2/24},
                                             "sport": {'game': 3/26, 'a': 3/26, 'clean': 3/26, 'great': 2/26,
                                                       'forgettable': 2/26, 'very': 2/26, 'match': 2/26, 'but': 2/26}}
        self.correct_extended_empty_likelihoods = {"sport": 1/26, "not sport":  1/24}
        self.correct_extended_a_very_close_game_score = {"sport": (3/26) * (2/26) * (1/26) * (3/26) * (3/5),
                                                         "not sport": (2/24) * (1/24) * (2/24) * (1/24) * (2/5)}

        # Shortened Data
        self.dictionary = {'game', 'was', 'a', 'very', 'the', 'great', 'forgettable', 'match', 'clean', 'but',
                           'election', 'it', 'close', 'over'}
        self.shortened_dictionary = self.dictionary.copy()
        self.shortened_dictionary.remove("game")
        self.correct_shortened_likelihoods = {"not sport": {'election': 3/22, 'was': 3/22, 'over': 2/22, 'the': 2/22,
                                                            'close': 2/22, 'it': 2/22, 'a': 2/22},
                                              "sport": {'a': 3/22, 'clean': 3/22, 'great': 2/22,
                                                        'forgettable': 2/22, 'very': 2/22, 'match': 2/22, 'but': 2/22}}
        self.correct_shortened_empty_likelihoods = {"sport": 1/22, "not sport":  1/22}
        self.correct_shortened_a_very_close_game_score = {"sport": (3/22) * (2/22) * (1/22) * (3/5),
                                                          "not sport": (2/22) * (1/22) * (2/22) * (2/5)}

    def test_constructor_invalid_smoothing_factor(self):
        self.assertRaises(ValueError, Multinomial, smoothing=-1.0)

    def test_train_labels_not_a_list(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, set(self.sports_labels), self.sports_data)

    def test_train_labels_are_not_str_or_int(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, list(map(lambda x: None, self.sports_labels)), self.sports_data)

    def test_train_training_data_is_not_in_a_list(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, self.sports_labels, set())

    def test_train_training_data_does_not_contains_lists(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, self.sports_labels, map(lambda x: set(), self.sports_data))

    def test_train_training_data_does_not_contains_lists_of_strs(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.train, self.sports_labels, map(lambda x: [0], self.sports_data))

    def test_train_number_of_labels_and_docs_differ(self):
        model = Multinomial()
        self.assertRaises(ValueError, model.train, self.sports_labels[0:4], self.sports_data)

    def test_train_model_params(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)

        # Correct Model
        self.assertEqual(model.priors, self.correct_priors)
        self.assertEqual(model.label_counts, self.correct_label_count)
        self.assertEqual(model.empty_likelihoods, self.correct_empty_likelihoods)
        self.assertDictEqual(model.likelihoods, self.correct_likelihoods)

    def test_predict_test_data_is_not_in_a_list(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.predict, set())

    def test_predict_test_data_does_not_contains_lists(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.predict, map(lambda x: set(), self.a_very_close_game))

    def test_predict_test_data_does_not_contains_lists_of_strs(self):
        model = Multinomial()
        self.assertRaises(TypeError, model.predict, map(lambda x: [0], self.sports_data))

    def test_predict_prediction_and_score(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Prediction Output
        self.assertEqual(prediction[0], "sport")

        # Correct Score Output
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score[0]["sport"], self.correct_a_very_close_game_score["sport"])
        self.assertAlmostEqual(score[0]["not sport"], self.correct_a_very_close_game_score["not sport"])

    def test_update_labels_not_a_list(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update, set(self.sports_labels), self.sports_data)

    def test_update_labels_are_not_str_or_int(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update, list(map(lambda x: None, self.sports_labels)), self.sports_data)

    def test_update_training_data_is_not_in_a_list(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.train, self.sports_labels, set())

    def test_update_training_data_does_not_contains_lists(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.train, self.sports_labels, map(lambda x: set(), self.sports_data))

    def test_update_training_data_does_not_contains_lists_of_strs(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.train, self.sports_labels, map(lambda x: [0], self.sports_data))

    def test_update_number_of_labels_and_docs_differ(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(ValueError, model.train, self.sports_labels[0:4], self.sports_data)

    def test_update_add_more_training_data(self):
        model = Multinomial()
        model.train(self.sports_labels[0:4], self.sports_data[0:4])
        model.update([self.sports_labels[4]], [self.sports_data[4]])
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Model
        self.assertEqual(model.priors, self.correct_priors)
        self.assertEqual(model.label_counts, self.correct_label_count)
        self.assertEqual(model.empty_likelihoods, self.correct_empty_likelihoods)
        self.assertDictEqual(model.likelihoods, self.correct_likelihoods)

        # Correct Score Output
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score[0]["sport"], self.correct_a_very_close_game_score["sport"])
        self.assertAlmostEqual(score[0]["not sport"], self.correct_a_very_close_game_score["not sport"])

        # Correct Prediction Output
        self.assertEqual(prediction[0], "sport")

    def test_update_dictionary_new_dictionary_is_not_set(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update_dictionary, list(self.extended_dictionary))

    def test_update_dictionary_new_dictionary_does_not_contain_strings(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update_dictionary, {0})

    def test_update_dictionary_new_dictionary_is_empty(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        self.assertRaises(TypeError, model.update_dictionary, {})

    def test_update_dictionary_correct_extend_dictionary(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        model.update_dictionary(self.extended_dictionary)
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Model Parameter Updates
        self.assertEqual(model.priors, self.correct_priors)
        self.assertEqual(model.label_counts, self.correct_label_count)
        self.assertEqual(model.empty_likelihoods, self.correct_extended_empty_likelihoods)
        self.assertDictEqual(model.likelihoods, self.correct_extended_likelihoods)

        # Correct Scores
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score[0]["sport"], self.correct_extended_a_very_close_game_score["sport"])
        self.assertAlmostEqual(score[0]["not sport"], self.correct_extended_a_very_close_game_score["not sport"])

        # Correct Prediction
        self.assertEqual(prediction[0], "sport")

    def test_update_dictionary_correct_shorten_dictionary(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        model.update_dictionary(self.shortened_dictionary)
        prediction, score = model.predict(self.a_very_close_game)

        # Correct Model Parameter Updates
        self.assertEqual(model.priors, self.correct_priors)
        self.assertEqual(model.label_counts, self.correct_label_count)
        self.assertEqual(model.empty_likelihoods, self.correct_shortened_empty_likelihoods)
        self.assertDictEqual(model.likelihoods, self.correct_shortened_likelihoods)

        # Correct Scores
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score[0]["sport"], self.correct_shortened_a_very_close_game_score["sport"])
        self.assertAlmostEqual(score[0]["not sport"], self.correct_shortened_a_very_close_game_score["not sport"])

        # Correct Prediction
        self.assertEqual(prediction[0], "sport")
