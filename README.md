# Predicting gold recovery in rougher flotation

This notebook contains my original project analysis of how ore characteristics and flotation settings relate to gold recovery. I cleaned the data, used a chronological train/validation/test split, and compared linear regression, K-nearest neighbours, and neural-network models. I evaluated the selected linear regression model on held-out data, used permutation importance to examine the inputs, and explored operating settings with a bounded optimisation.

## Running the notebook locally

Install the dependencies with `python -m pip install -r requirements.txt`, then open `gold_recovery.ipynb` in Jupyter. Running the full notebook requires an authorised local copy of the project dataset at `data/flotation.csv` and a JSON object mapping the feed column names to the scenario values at `data/fixed_feed.json`.

The repository contains the analysis source. The dataset, assignment materials, submitted report, specified feed values, and saved notebook outputs are excluded.
