# machine-learning

## Initializations

```bash
    conda create -p venv python=3.12
    conda activate venv/
```

## Tokenization

- Paragraph to sentences (Tokenization into sentence)
- Sentence to words/vocabolary (Tokenization into words)

## NLTK (Tokenization library)

- Alternative: `spacy`

```bash
  pip install ntlk
  pip install numpy
  pip install -r requirements.txt
```

- Or install those libraries, just for the Conda environment:

```bash
  conda install -p venv/ nltk
  conda install -p venv/ numpy
```

- [Install skykit-learn](https://scikit-learn.org/stable/install.html). This may install dependency from `numpy` and `scipy`

```bash
  conda uninstall numpy # just in case
  conda install -c conda-forge scikit-learn
```

- Or install with `pip` under conda environment. If there is stack error, uninstall `numpy`

```bash
  pip unstall numpy
  pip install --force-reinstall scikit-learn
```

- One time action to download punkt_tab, in python code:

```python
  import nltk
  nltk.download('punkt')  # Download the tokenizer models

  from nltk.tokenize import word_tokenize, sent_tokenize

  text = "Hello, world! This is NLTK's tokenizer."
  words = word_tokenize(text)  # Tokenizes into words
  sentences = sent_tokenize(text)  # Tokenizes into sentences
```

### Lemmatization

- Require to download: nltk.download('wordnet')

```python
  nltk.download('wordnet')
```
