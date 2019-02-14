import unittest

from naivebayes.models.dictionary import Multinomial


class TestDistanceMetrics(unittest.TestCase):
    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
        self.sports_data = [["a", "great", "game"],
                            ["the", "election", "was", "over"],
                            ["very", "clean",  "match"],
                            ["a", "clean", "but", "forgettable", "game"],
                            ["it", "was", "a", "close", "election"]]
        self.sports_labels = ["sport", "not sport", "sport", "sport", "not sport"]

    # Sports Data
    def test_nb_dict_sports(self):
        model = Multinomial()
        model.train(self.sports_labels, self.sports_data)
        prediction = model.predict([["a", "very", "close", "game"]])[0]

        self.assertEqual(prediction, "sport")
