import unittest

from naivebayes.models.dictionary import Multinomial


class TestDictionaryNaiveBayes(unittest.TestCase):

    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
        self.sports_data = [["a", "great", "game"],
                            ["the", "election", "was", "over"],
                            ["very", "clean",  "match"],
                            ["a", "clean", "but", "forgettable", "game"],
                            ["it", "was", "a", "close", "election"]]
        self.sports_labels = ["sport", "not sport", "sport", "sport", "not sport"]

    # Check Correct Prediction Is Made
    def test_nb_dict_prediction(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        prediction = model.predict([["a", "very", "close", "game"]])[0]

        self.assertEqual(prediction, "sport")

    # Check Score is Returned And Score Is Correct
    def test_nb_dict_scores(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        prediction, score = model.predict([["a", "very", "close", "game"]], return_scores=True)

        self.assertIsNotNone(score)
        self.assertAlmostEqual(score[0]['sport'], 2.76e-05)
        self.assertAlmostEqual(score[0]['not sport'], 5.72e-06)
