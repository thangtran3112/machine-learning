# PyTorch Seq2Seq

This repo contains train models to translate from German to English.

- For Nvidia GPU training, we may need to install `torch` for corresponding CUDA version, such as:

```bash
  # For CUDA 12.1
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  # 3. Verify installation in Python
  python -c "import torch; print(torch.cuda.is_available())"
```

```bash
  pip install -r requirements.txt
```

- Notes: `torchtext` is deprecated. It will require Python=3.11 at this moment, and it also require to install with `conda -c pytorch torchtext`, which will only work with CPU training.

## Getting Started

Install the required dependencies with: `pip install -r requirements.txt --upgrade`.

We'll also make use of [spaCy](https://spacy.io/) to tokenize our data which requires installing both the English and German models with:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 1. Sequence to Sequence Learning with LSTM

- 2-Layer LSTM in Encoder, each layer with 2 LSTM cells. [1 - Sequence to Sequence Learning.ipynb](./1%20-%20Sequence%20to%20Sequence%20Learning.ipynb)
  ![2-Layer encoder-decoder](./TwoLayerEachTwoLSTMEncoderDecoder.png)
- Context Vector of 2 Long term memory cells, and 2 Short-term memory cells will be passed to Decoder
- Alternative: Decode the sequence from EOS backwards, or from SOS (Start of Sentence) forward.
- The model training use `Teacher Enforcing` probability of 0.5

## 2. Learning Phrase Representations with encoder-decider

Now we have the basic workflow covered, this tutorial will focus on improving our results. Building on our knowledge of PyTorch, we'll implement a second model, which helps with the information compression problem faced by encoder-decoder models. This model will be based off an implementation of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), which uses GRUs.

## 3 - Neural Machine Translation by Jointly Learning to Align and Translate Open In Colab

Next, we learn about attention by implementing [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). This further allievates the information compression problem by allowing the decoder to "look back" at the input sentence by creating context vectors that are weighted sums of the encoder hidden states. The weights for this weighted sum are calculated via an attention mechanism, where the decoder learns to pay attention to the most relevant words in the input sentence.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- https://github.com/spro/practical-pytorch
- https://github.com/keon/seq2seq
- https://github.com/pengshuang/CNN-Seq2Seq
- https://github.com/pytorch/fairseq
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
