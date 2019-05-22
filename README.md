# Deep Compression

## Dependencies 

	- PyTorch 1.1
	- PIL
	- Numpy
	- PyPlot
	- Scipy

## Background

The goal of this project is to investigate the compression of deep learning neural networks, as discussed in Han, S., Mao, H. and Dally, W.J., 2015. *Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding*. This project focuses on AlexNet, but the method can be abstracted to any other network.

As the title suggests the project applies 3 compression methods to a model in an effort to reduce the size on disk. They are:

	1. Pruning: The process of finding and removing redundant weights.
	2. Quantisation: Limiting the number of effective weights.
	3. Huffman Encoding: Algorithmically finding the most efficient way to store the weights.

## How To Run

Ensure that all dependencies are installed. 

Simply run `DeepCompression.py`

## How To Change Variables

### Photo

Lion is provided as an example. To use your own image for testing, simply point to program to a different directory containing your photo(s). The program can test on more than one image, it will iterate over all images in the directory.

### Quality Parameter (QP)

This can be adjusted from the Threshold method

`Threshold(quality parameter for CONV layers, quality parameter for FC layers)`

### Quantisation

Similar to QP this can adjusted from the Quantise method

`Quantise(thresholded_model, number of bits per weight)`