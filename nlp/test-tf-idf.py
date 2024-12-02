import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Sample documents
documents = [
    "The cats are chasing a mouse.",
    "A cat chased the mice yesterday.",
    "Cats love chasing after mice."
]

# Lemmatization function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def display_result(tfidf_matrix, vectorizer, label):
    # Display TF-IDF matrix
  print(f"\n{label} TF-IDF Matrix:")
  print(tfidf_matrix.toarray())

  # Display feature names
  print(f"\n{label} Feature Names:")
  print(vectorizer.get_feature_names_out())

# Function to preprocess text (lemmatization + stopword removal)
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

# Lemmatize the documents
lemmatized_docs = [preprocess_text(doc) for doc in documents]
print("Lemmatized Documents Without stopwords:")
print(lemmatized_docs)

# Compute TF-IDF
vectorizer = TfidfVectorizer(max_features=200)
tfidf_matrix = vectorizer.fit_transform(lemmatized_docs)
display_result(tfidf_matrix, vectorizer, 'No N-gram')

# Compute TF-IDF with n-gram
vectorizer = TfidfVectorizer(max_features=200, ngram_range=(2, 2))
tfidf_matrix = vectorizer.fit_transform(lemmatized_docs)
display_result(tfidf_matrix, vectorizer, 'With 2-gram or N-gram range (2,2)')
