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
    conda config --set channel_priority strict
    conda install -p .\venv\ -c conda-forge ipykernel=6.29.5 pandas=2.2.3 numpy=2.0.2 scikit-learn=1.5.1 matplotlib=3.9.2 streamlit==1.40.1
    pip install tensorflow==2.18 scikeras==0.13.0  tensorboard==2.18.0
```

- In Ubuntu or MacOS, but not preferable compared to using `conda install`:

```zsh
  pip install -r requirements.txt
  export TF_CPP_MIN_LOG_LEVEL=2 # this is needed for GPU training, to filter out INFO and WARNING logs
  export TF_ENABLE_ONEDNN_OPTS=0
```

## Streamlit Cloud

- Must set environment variable `ROOT_DIR = "./annclassification"`

To ensure that Keras and TensorFlow run on a GPU, you need to have the appropriate GPU drivers and libraries installed, such as CUDA and cuDNN. Here are the steps to set up and verify that TensorFlow is using the GPU:
