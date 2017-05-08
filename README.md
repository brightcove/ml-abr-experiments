# ML ABR
Experiments using [TensorFlow](https://www.tensorflow.org/) to create a more accurate bandwidth-predictor for adaptive bitrate streaming.

# Building
Install TensorFlow using their [Getting Started docs](https://www.tensorflow.org/get_started/get_started). We recommend the virtualenv setup if you're on macOS.

# Dataset
The training and validation sets used for this project are exported
from BigQuery into
[gs://dmlap-experiments](https://console.cloud.google.com/storage/browser/dmlap-experiments). At
minimum, you'll want to download a set and place it at
`data/bitrate_samples_0.validation` to see the neural network do
something. If you want to train the neural net further, place your
training data at `data/bitrate_samples_0.training`.

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
