import unittest
import svm
import numpy as np 

class TestClassifier(unittest.TestCase):
    X=[["test11", "test12"], ["test21", "test22"], ["test31", "test32"]]
    X = np.asarray(X)
    y=[[1,0,1,0], [0,0,0,1], [0,0,0,1]]
    y = np.asarray(y)
    classifier = svm.SVMClassifier(X, y)

    def test_init(self):
        self.assertIsNotNone(self.classifier)
        self.assertTrue(len(self.classifier.classifiers) == 4)

    def test_train_test_split(self):
        X_train, X_test, y_train, y_test = self.classifier.train_test_split(self.X, self.y)
        self.assertTrue(len(X_train) == 2)
        self.assertTrue(len(X_test) == 1)
        self.assertTrue(len(y_train) == 2)
        self.assertTrue(len(y_test) == 1)

        
if __name__ == '__main__':
    unittest.main()