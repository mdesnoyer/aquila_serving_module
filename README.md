# Overview

This repo provides the [Tensorflow Serving](https://tensorflow.github.io/serving/) module for the [Aquila](https://github.com/mdesnoyer/aquila) image ranking model. 

# System Setup

In order to run the serving module, it must be built using bazel. Please see <setup.sh> for details of how to setup your system. The odds are that it needs to be tweaked for your setup.

# Interface

The model can be queried using gRPC using the protocol buffer definition from aquila_inference.proto. As input, it takes a 299x299 image and returns a 1024 vector of abstract features. 

A Python client SDK is provided in python/client.py to query the model and convert the abstract features into valence scores for different demographics.


# Understanding the Output

## Demographic Predictions

## Predicted Lift
Using the final valence score, the estimated lift of image A over image B is defined by:

```
lift = exp(A) / exp(B) - 1
```

# Pre-Trained Model

A pre-trained version of the model is available [here](https://www.dropbox.com/s/3af8auuovksidm7/aquila_model.tar.gz?dl=0). This version is trained on the valence experiments performed by Neon Labs Inc.. It has been trained on approximately 3.6M video frames extracted from random videos on [YouTube](https://www.youtube.com) that have been rated ~25M times by US users of Mechanical Turk.

To convert the abstract features output from the pre-trained model, use the 20160713-aquilav20*.pkl files in the demographics directory. 