# Recurrent Neuron Network

## 128-Node RNN layer

- The training does not use a specific randonmized seed. The prediction training may require experimenting.
- It would likely be more accurate if we are scaling to 256-neuron RNN layer
- The training is only use a small 25000 data sets, it will need more real dataset to make it accurate
- Accurracy within the dataset is around 96%.

## Installations

- This testing package requires a different conda venv environement, compared to the root venv/
- The root venv/ has `gensim` library, which require `numpy` version 1.26, which is not compatible with `pandas`
- Note: `conda` does not install `scikeras` at the moment.

```zsh
    conda create -p venv python=3.12
    conda activate venv/
    conda install -p venv/ ipykernel tensorflow pandas numpy scikit-learn tensorboard matplotlib streamlit
    pip install scikeras
```

- Conda-Forge often has the most up-to-date packages, including TensorFlow. Add it to your configuration:
  powershell

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

```powershell
    conda create -p venv python=3.12
    conda activate .\venv\
    conda config --add channels conda-forge
    conda config --set channel_priority flexible
    conda install -p .\venv\ ipykernel=6.29.5 pandas=2.2.3 numpy=2.0.2 scikit-learn=1.5.1 matplotlib=3.9.2 streamlit==1.40.1
    pip install tensorflow==2.18 scikeras==0.13.0  tensorboard==2.18.0
```

- Or, but not preferable compared to using `conda install`:

```zsh
  pip install -r requirements.txt
```

## Streamlit Cloud

```bash
  streamlit run app.py
```

- Must set environment variable `ROOT_DIR = "./simple_rnn_imdb"`

To ensure that Keras and TensorFlow run on a GPU, you need to have the appropriate GPU drivers and libraries installed, such as CUDA and cuDNN. Here are the steps to set up and verify that TensorFlow is using the GPU:
