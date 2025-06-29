# Spiking neuron hippocampus
A spiking neural network "brain" that learns using a hippocampus.

The included programs test different aspects of the model.

**codec** simply tests that 500-value embeddings can be encoded as neuron
spikes, then successfully decoded.

**pavlov** demonstrates basic Pavlovian cause-and-effect learning.

Three channels of neuron spikes simulate the ringing of a bell. After a pause,
three different channels then simulate the arrival of food. These spikes are
used to train the cortex.

When the trained cortex is stimulated by the ringing bell, it should output
both food and bell spikes.

**predict_self** tests that a cortex can be trained to produce outputs that
match its inputs.

The inputs encode a 500-value token embedding as neuron spikes. By default
the token is the word *American*, but **-R** will select a random token.

The program reports, over time:

- the number of neurons in the cortex;
- the correlation between the input and output;
- the volume of the output relative to the input; and
- the token that best matches the output.

At the end, it reports the ouput of the cortex when fed noise. Ideally, no
spikes should be output.

**sequence** tests the ability of a cortex with a feedback loop to learn a
sequence of outputs, essentially using repeated Pavlovian learning.

The cortex is trained with a sequence of eight two-channel spike patterns.
When it is later fed the first pattern in the sequence, it should iterate
through the patterns in order.

### Initialize the build directory

`cmake -S . -B build`

### Build everything

`cmake --build build`

### Run a binary

    build/codec
    build/pavlov
    build/predict_self [-R]
    build/sequence
