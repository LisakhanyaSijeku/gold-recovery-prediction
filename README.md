# Gold Recovery in Rougher Flotation

A portfolio adaptation of a chemical engineering project on predicting gold recovery from ore characteristics and controllable flotation settings. This demo generates its own synthetic observations. It contains no plant records, assessment materials, submission files, or original results.

## Run
```sh
python -m pip install -r requirements.txt
python demo.py
```

The demo screens invalid readings, splits observations chronologically, and fits all preprocessing on training data. It compares linear regression, K-nearest neighbours, and a neural network on validation mean absolute error. The selected model is evaluated on a held-out test period and interpreted with permutation importance.

A final screening step fixes a representative synthetic feed and compares the predicted recovery of jointly observed operating settings against typical settings. This is a demonstration of model-based decision support. The synthetic scores, estimated effects, and recommended settings have no plant-operating interpretation.
