# Triplet-based Person Re-Identification

Modification based on [In Defense of the Triplet Loss for Person Re-Identification] and some modification has been made suitable for DIVA project.

Code for reproducing the results of our [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737) paper.

The original author provided the following things:
- The exact pre-trained weights for the TriNet model as used in the paper, including some rudimentary example code for using it to compute embeddings.
  See section [Pretrained models](#pretrained-models).
- A clean re-implementation of the training code that can be used for training your own models/data.
  See section [Training your own models](#training-your-own-models).
- A script for evaluation which computes the CMC and mAP of embeddings in an HDF5 ("new .mat") file.
  See section [Evaluating embeddings](#evaluating-embeddings).
- A list of [independent re-implementations](#independent-re-implementations).

If you use any of the provided code, please cite:
```
@article{HermansBeyer2017Arxiv,
  title       = {{In Defense of the Triplet Loss for Person Re-Identification}},
  author      = {Hermans*, Alexander and Beyer*, Lucas and Leibe, Bastian},
  journal     = {arXiv preprint arXiv:1703.07737},
  year        = {2017}
}
```


# Pretrained TensorFlow models

For convenience, we provide the pretrained weights for our TriNet TensorFlow model, trained on Market-1501 using the code from this repository and the settings form our paper. The TensorFlow checkpoint can be downloaded in the [release section](https://github.com/VisualComputingInstitute/triplet-reid/releases/tag/250eb1).


# Pretrained Theano models

We provide the exact TriNet model used in the paper, which was implemented in
[Theano](http://deeplearning.net/software/theano/install.html)
and
[Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html).

As a first step, download either of these pre-trained models:
- [TriNet trained on MARS](https://omnomnom.vision.rwth-aachen.de/data/trinet-mars.npz) (md5sum: `72fafa2ee9aa3765f038d06e8dd8ef4b`)
- [TriNet trained on Market1501](https://omnomnom.vision.rwth-aachen.de/data/trinet-market1501.npz) (md5sum: `5353f95d1489536129ec14638aded3c7`)

Next, create a file (`files.txt`) which contains the full path to the image files you want to embed, one filename per line, like so:

```
/path/to/file1.png
/path/to/file2.jpg
```

Finally, run the `trinet_embed.py` script, passing both the above file and the pretrained model file you want to use, like so:

```
python trinet_embed.py files.txt /path/to/trinet-mars.npz
```

And it will output one comma-separated line for each file, containing the filename followed by the embedding, like so:

```
/path/to/file1.png,-1.234,5.678,...
/path/to/file2.jpg,9.876,-1.234,...
```

You could for example redirect it to a file for further processing:

```
python trinet_embed.py files.txt /path/to/trinet-market1501.npz >embeddings.csv
```

You can now do meaningful work by comparing these embeddings using the Euclidean distance, for example, try some K-means clustering!

A couple notes:
- The script depends on [Theano](http://deeplearning.net/software/theano/install.html), [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) and [OpenCV Python](http://opencv.org/) (`pip install opencv-python`) being correctly installed.
- The input files should be crops of a full person standing upright, and they will be resized to `288x144` before being passed to the network.


# Training your own models

If you want more flexibility, we now provide code for training your own models.
This is not the code that was used in the paper (which became a unusable mess),
but rather a clean re-implementation of it in [TensorFlow](https://www.tensorflow.org/),
achieving about the same performance.

- **This repository requires at least version 1.4 of TensorFlow.**
- **The TensorFlow code is Python 3 only and won't work in Python 2!**

:boom: :fire: :exclamation: **If you train on a very different dataset, don't forget to tune the learning-rate and schedule** :exclamation: :fire: :boom:

If the dataset is much larger, or much smaller, you might need to train much longer or much shorter.
Market1501, MARS (in tracklets) and DukeMTMC are all roughly similar in size, hence the same schedule works well for all.
CARS196, for example, is much smaller and thus needs a much shorter schedule.

## Defining a dataset

A dataset consists of two things:

1. An `image_root` folder which contains all images, possibly in sub-folders.
2. A dataset `.csv` file describing the dataset.

To create a dataset, you simply create a new `.csv` file for it of the following form:

```
identity,relative_path/to/image.jpg
```

Where the `identity` is also often called `PID` (`P`erson `ID`entity) and corresponds to the "class name",
it can be any arbitrary string, but should be the same for images belonging to the same identity.

The `relative_path/to/image.jpg` is relative to aforementioned `image_root`.

## Training

Given the dataset file, and the `image_root`, you can already train a model.
The minimal way of training a model is to just call `train.py` in the following way:

```
python train.py \
    --train_set data/market1501_train.csv \
    --image_root /absolute/image/root \
    --experiment_root ~/experiments/my_experiment
```

This will start training with all default parameters.
We recommend writing a script file similar to `market1501_train.sh` where you define all kinds of parameters,
it is **highly recommended** you tune hyperparameters such as `net_input_{height,width}`, `learning_rate`,
`decay_start_iteration`, and many more.
See the top of `train.py` for a list of all parameters.

As a convenience, we store all the parameters that were used for a run in `experiment_root/args.json`.

### Pre-trained initialization

If you want to initialize the model using pre-trained weights, such as done for TriNet,
you need to specify the location of the checkpoint file through `--initial_checkpoint`.

For most common models, you can download the [checkpoints provided by Google here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
For example, that's where we get our ResNet50 pre-trained weights from,
and what you should pass as second parameter to `market1501_train.sh`.

### Example training log

This is what a healthy training on Market1501 looks like, using the provided script:

![Screenshot of tensorboard of a healthy Market1501 run](healthy-market-run.png)

The `Histograms` tab in tensorboard also shows some interesting logs.

## Interrupting and resuming training

Since training can take quite a while, interrupting and resuming training is important.
You can interrupt training at any time by hitting `Ctrl+C` or sending `SIGINT (2)` or `SIGTERM (15)`
to the training process; it will finish the current batch, store the model and optimizer state,
and then terminate cleanly.
Because of the `args.json` file, you can later resume that run simply by running:

```
python train.py --experiment_root ~/experiments/my_experiment --resume
```

The last checkpoint is determined automatically by TensorFlow using the contents of the `checkpoint` file.


