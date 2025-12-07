# Introduction
This project implements a word predictor from the given which gives predictions to complete the given word by using a character based
Ngram model. Ngram model helps to estimate the probability of occurrence of a word or a character by taking into account the previous n-1 words or characters encountered so far.
Different sequences of characters or words have different occurrence probabilities depending on the usage, the grammar and the topics to which the text is related. The Ngram model in this
implementation can be trained on .txt files and folders containing .txt files.
The code also shows some metrics like  average number of tabs pressed per word that to analyze performance.
# Methodology
NGram model implementation and word prediction: Given a text corpus and value of
n, the Ngram implementation code first stores the frequencies of occurrence of characters
corresponding to group of previous (n-1) character using a python defaultdict of counter .
Essentially a two dimensional hash table where first dimension is the context of previous n-1
letters ,and second dimension is the n th character appearing after them. Removal of irrelevant
characters for cleaning data ,converting data to lower case and using padding to account for
words smaller than size n was also done to give better predictions.
While predicting words ,when we have the context of one or more previous characters, different
possible sequence of next characters are considered ,and probability of the corresponding
word is computed ,until adding further characters results in very low probability. Then the top k
words with highest computed probability are returned.

Data collection: Number of letters typed, tab presses, average number of letters typed
per word and average number of tab presses per word are calculated and displayed on the
screen.
