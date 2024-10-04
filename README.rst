Sparse Coding Library for Tensorflow
=================================

Installation
------------

To install the package, clone it to a local folder, then run `pip install -e .` from the folder to install the package in edit mode (so changes to the code are used).


Overview
------------

**Recommend running everything from the project root directory (ensures yolo will work).**

Main sparse coding file is found at:
    sparse_coding_torch/train_sparse_model.py

Sparse coding command example:
    python sparse_coding_torch/train_sparse_model.py --dataset pnb --train


Main classification model is found at:
    sparse_coding_torch/train_classifier.py

Classification command example:
    python sparse_coding_torch/train_classifier.py --sparse_checkpoint sparse.pt --dataset pnb --train

PTX Checkpoints can be found at:
    ptx_tensorflow/
