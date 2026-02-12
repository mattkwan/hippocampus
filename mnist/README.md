A spiking neurons trained by a hippocampus
==========================================

This code implements a spiking neural network that uses a hippocampus to
train a cortex. Spiking neurons communicate via "spikes", whose timing
determines their intensity: frequent spikes encode higher values.

To test the effectiveness of the hippocampus, it's trained and tested against
the MNIST handwritten digit dataset with the following script:

`python main.py`

> `--train_count` _n_ The number of samples to train on. If not specified, uses all 60,000 samples.
>
> `--test_count` _n_ The number of samples to test. If not specified, uses all 10,000 samples.
>
> `--show_failures` Display incorrect predictions as they occur.

In this implementation, the MNIST images (28x28 greyscale) are first run
through horizontal and vertical edge detection filters. The positive and
negative values from each filter are then split into separate arrays and
downscaled by four (4x4 average pooling). This is similar to the steps carried
out by a visual cortex.

When trained with these filters, 599 neurons are added to the cortex. It can
then predict digits with 89.6% accuracy.

In previous tests, it was trained on images downscaled to 14x14 (196 raw
pixels). The resulting cortex predicted digits with 89.1% accuracy using 1603
neurons.
