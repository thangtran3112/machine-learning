from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Apply CountVectorizer with max_features
# min_df=2: Keeps only terms that appear in at least 2 documents.
# max_df=0.75: Excludes terms that appear in more than 75% of the documents

vectorizer = CountVectorizer(max_features=200,ngram_range=(1, 3),binary=True)
X = vectorizer.fit_transform(corpus)

# print("Vocabulary:", vectorizer.get_feature_names_out())
# Convert vocabulary_ values to standard integers
vocabulary = {key: int(value) for key, value in vectorizer.vocabulary_.items()}

# Print the modified vocabulary
print("Vocabulary:", vocabulary)

