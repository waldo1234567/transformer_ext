### Pytorch CUDA/C++ Extension

This repo demonstrates a PyTorch C++/CUDA extension that implements a fused matmul + GELU forward/backward kernel and integrates it into a small Transformer-style model. It includes build scripts, a minimal training script on a tiny text corpus, profiling guides.This project is intended to showcase systems-level ML engineering: CUDA kernel development, PyTorch extension integration, profiling, and optimization.

* Built a PyTorch C++/CUDA extension implementing a fused matmul+GELU kernel and integrated it into a Transformer-style model.

* Profiled and optimized training (AdaptiveSoftmax, pre-tokenization, DataLoader tuning) — reduced training time from ~10 min → ~2 min for 10 epochs on sample data.

* Tools: CUDA, C++, PyTorch, Python, PyTorch Profiler, TensorBoard.


### Prerequisites

* Python 3.8+
* PyTorch (matching with CUDA).
* CUDA Toolkit
* On Windows = Visual Studio Build Tools

### Build and install

```bash
cd custom_extension

python setup.py build

python -m pip install .

cd ..

```

### Running the training (with pure pytorch)

```bash
python train_model_profiler.py 
```

### Running the training (with cuda extension)

```bash
python ext_train_proper.py
```

### Benchmark the extenstion (saved to .csv)

```bash
python with_ext_class.py micro
```


