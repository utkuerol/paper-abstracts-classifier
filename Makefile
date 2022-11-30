# all preprocessing options except lemmatize and remove stop words
svm:
	python main.py svm 1 1 0 0 tfidf

randomforest:
	python main.py randomforest 1 1 0 0 tfidf
	
lstm-tfidf:
	python main.py lstm 1 1 0 0 tfidf

# all preprocessing options
svm-1111:
	python main.py svm 1 1 1 1 tfidf

randomforest-1111:
	python main.py randomforest 1 1 1 1 tfidf
	
lstm-1111-tfidf:
	python main.py lstm 1 1 1 1 tfidf

# run unit tests
test:
	python preprocessing/test_preprocess.py
	python model/test_classifier.py