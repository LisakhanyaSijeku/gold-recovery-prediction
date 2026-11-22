# Predicting gold recovery in rougher flotation

This project uses historical plant data to predict rougher flotation gold recovery from ore characteristics and operating conditions. The notebook cleans and inspects the data, splits it chronologically, and compares linear regression, K-nearest neighbours, and neural networks. It evaluates the selected linear regression model on held-out data, examines input importance, and explores operating settings for a fixed feed.

The weak held-out performance and physically impossible optimised recovery limit the model's usefulness for plant decisions. The original results and plots are saved in the notebook.

## Run locally

Install the dependencies with `python -m pip install -r requirements.txt`. Open `gold_recovery.ipynb` from the repository folder in Jupyter and run the cells in order. The CSV is included next to the notebook and the six fixed feed values are defined in the optimisation section.
