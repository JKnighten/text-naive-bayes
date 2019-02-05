import unittest

from naivebayes.models.dictionary import Multinomial


class TestDistanceMetrics(unittest.TestCase):
    def setUp(self):
        # Sports Data Example
        # Example From: https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
        self.freq_sports_dict = {"a": 2, "great": 1, "game": 2, "very": 1, "clean": 2, "match": 1, "but": 1,
                                 "forgettable": 1}
        self.freq_not_sports_dict = {"the": 1, "election": 2, "was": 2, "over": 1, "it": 1, "close": 1, "a": 1}

        self.train_sports = {"sport": self.freq_sports_dict, "not sport": self.freq_not_sports_dict}
        self.label_sports = {"sport": .3, "not sport": 2}

    # Sports Data
    def test_nb_dict_sports(self):
        model = Multinomial()
        model.train(self.label_sports, self.train_sports)
        prediction = model.predict([["a", "very", "close", "game"]])[0]

        self.assertEqual(prediction, "sport")
