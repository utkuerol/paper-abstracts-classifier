import unittest
from preprocess import Preprocesser

class TestPreprocesser(unittest.TestCase):
    preprocesser = Preprocesser(data_path="data/mock/test_data.csv", remove_punct=True, lower_case=True,
                            remove_stop_words=False, stemming=False)
    test_data_shape_X = (10,)
    test_data_shape_y = (10,4)

    def test_get_data(self):
        X, y = self.preprocesser._get_data()
        self.assertTrue(X.shape == self.test_data_shape_X)
        self.assertTrue(y.shape == self.test_data_shape_y)

    def test_preprocess_sentence(self):
        input = "Test, 123 :; TeST-"
        output_should = ["test", "123", "test"]
        output_is = self.preprocesser.preprocess(input)
        self.assertEqual(output_is, output_should)

if __name__ == '__main__':
    unittest.main()