import gensim
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

print(f"Most similar to cat is {wv.most_similar('cat')}")

vec = wv['king']-wv['man']+wv['woman']
print(f"Most similar to (king - man + woman) is {wv.most_similar(vec)}")