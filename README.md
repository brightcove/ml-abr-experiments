# ML ABR
Experiments using [TensorFlow](https://www.tensorflow.org/) to create a more accurate bandwidth-predictor for adaptive bitrate streaming.

# Building
Install TensorFlow using their [Getting Started docs](https://www.tensorflow.org/get_started/get_started). We recommend the virtualenv setup if you're on macOS.

# Dataset
The training and validation sets used for this project have been [made
available by
Brightcove](https://storage.googleapis.com/dmlap-experiments/bitrate_samples.csv.gz). Download
the set, unzip it, and remove the first line. Place it at
`data/bitrate_samples_0.validation` to see the neural network do
something. If you want to train the neural net further, place your
training data at `data/bitrate_samples_0.training`. If you want to get
serious, you should probably split the data set into training and
validation sets before getting started.

# Usage
You can run the neural net validation by running:

```sh
python lstm.py
```

If you'd like to train the neural net before validating, you'll need
to add another option:

```sh
python lstm.py --train
```

Or check out all the options by running `python lstm.py --help`.
