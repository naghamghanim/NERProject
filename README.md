# NERProject

    data/
	- corpus.csv:  All details about the dataset, contains 562280 train + valid + test
	- corpus2.csv: contains 562280, only token and its flag
	- corpus.txt: contains all token with their flags ( we can you use for any processing)
	- train80.csv: contains 448556 (train + valid datasets)
	- test20.csv: contains 113726 (test data)
	- sample_tfidf.csv: contains the features for each token (12 features)
	- sample-word2vec.csv: ontains the features for each token (12 features)


    Features/
	- if_idf_features.csv: the final tf_idf extracted feaures used for ML methods.
	- word2vec_features.csv: the final word2vec extracted feaures used for ML methods.
	- ourfeatures.csv: the final surrounding word feature extracted feaures used for ML methods.

ExtractFeatures_tfidf.py: a file for extracing a features using tf-idf method.
ExtractFeatures_word2vec.py: a file for extracing a features using word2vec method.
ExtractionFeatures_surrounding.py: a file for extracing a features using surrounding word feature method.


sentences.txt: sentences for all corpus. number of sentences is 25134
labels.txt: the labels for each sentences.
			
