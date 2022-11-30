# Preprocessing

Applied preprocessing operations: 

- Remove punctuation
- Lowercase
- Tokenize

Other traditional preprocessing operations such as lemmatizing, stemming and stop word removal have not been applied due to lowered performance. But this is still completely configurable in the [source code](preprocessing/preprocess.py).

In the end we have a minimal preprocessing phase with 3 operations. The example below depicts a single sentence before and after preprocessing applied: 

![](img/preprocessing_example.png)
