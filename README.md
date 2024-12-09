## [Install GPU Drivers and Libraries](https://medium.com/@gokulprasath100702/a-guide-to-enabling-cuda-and-cudnn-for-tensorflow-on-windows-11-a89ce11863f1)

1. [Windows tensorflow compatibility](https://www.tensorflow.org/install/source_windows). [Linux tensorflow compatibility](https://www.tensorflow.org/install/source#gpu)
2. **Install NVIDIA GPU Driver**: Ensure you have the latest NVIDIA GPU driver installed.
3. **Install CUDA Toolkit**: Download and install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
4. **Install cuDNN**: Download and install cuDNN from the [NVIDIA website](https://developer.nvidia.com/cudnn).
5. In order for CUDA toolkit to work, we must also install Visual Studio with `Desktop development with C++`. [See guideline here](https://medium.com/@gokulprasath100702/a-guide-to-enabling-cuda-and-cudnn-for-tensorflow-on-windows-11-a89ce11863f1)

## Step 2: Install TensorFlow with GPU Support

Install TensorFlow with GPU support using pip:

```sh
pip install tensorflow-gpu
```

## Step 3: Verify TensorFlow is Using the GPU

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

## Controlling fan in Ubuntu for Asus laptop:

- [Ask Ubuntu - throttle_thermal_policy set to 1](https://askubuntu.com/questions/1254364/how-to-control-fans-on-an-asus-laptop)
- [Control GPU fan through python script](https://blog.closex.org/posts/26a7c6ee/)
- [Script implementation in fan folder](./fan/fan.sh)
