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
    conda install -p .\venv\ -c conda-forge ipykernel=6.29.5 pandas=2.2.3 numpy=1.26.4 scikit-learn=1.5.1 matplotlib=3.9.2 streamlit==1.40.1
    pip install tensorflow==2.17 scikeras tensorboard==2.17.0
```

- Or, but not preferable compared to using `conda install`:

```zsh
  pip install -r requirements.txt
```

## Streamlit Cloud

- Must set environment variable `ROOT_DIR = "./annclassification"`

To ensure that Keras and TensorFlow run on a GPU, you need to have the appropriate GPU drivers and libraries installed, such as CUDA and cuDNN. Here are the steps to set up and verify that TensorFlow is using the GPU:

### [Install GPU Drivers and Libraries](https://medium.com/@gokulprasath100702/a-guide-to-enabling-cuda-and-cudnn-for-tensorflow-on-windows-11-a89ce11863f1)

1. [Read compatibility of Nvidia matrix for each GPU](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#support-matrix). At this moment, tensorflow 2.17 only works with `CUDA 11.8` and its corresponding `cuDNN`
2. **Install NVIDIA GPU Driver**: Ensure you have the latest NVIDIA GPU driver installed.
3. **Install CUDA Toolkit**: Download and install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
4. **Install cuDNN**: Download and install cuDNN from the [NVIDIA website](https://developer.nvidia.com/cudnn).
5. In order for CUDA toolkit to work, we must also install Visual Studio with `Desktop development with C++`. [See guideline here](https://medium.com/@gokulprasath100702/a-guide-to-enabling-cuda-and-cudnn-for-tensorflow-on-windows-11-a89ce11863f1)

### Step 2: Install TensorFlow with GPU Support

Install TensorFlow with GPU support using pip:

```sh
pip install tensorflow-gpu
```

### Step 3: Verify TensorFlow is Using the GPU

You can verify that TensorFlow is using the GPU by running the following code:

```python
import tensorflow as tf

# Check if TensorFlow is built with GPU support
print("Built with GPU support:", tf.test.is_built_with_cuda())

# List available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

# Check if a GPU is available
print("GPU available:", tf.test.is_gpu_available())
```

### Example Code to Run on GPU

Here is an example of how to ensure your Keras model runs on the GPU:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Ensure TensorFlow uses the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
```
