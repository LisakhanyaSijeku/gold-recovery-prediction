# Gold Recovery Prediction in Rougher Flotation

This chemical engineering project investigates whether historical operating data can predict how much gold a rougher flotation circuit recovers. The goal was to assess whether predictions could inform reagent dosing and other operating decisions as feed conditions change.

## Approach

- Screened missing and physically invalid measurements and excluded downstream assays to reduce target leakage.
- Split the data chronologically into training, validation, and test periods. Fitted feature scaling on the training data only.
- Compared linear regression, k-nearest neighbours, and neural networks. Evaluated the selected model on held-out data and used permutation importance to examine which inputs influenced its predictions.
- Explored reagent doses, airflow, and pulp levels for a fixed feed, with bounds drawn from the training data.

## Results

| Measure | Result |
| --- | ---: |
| Linear regression validation R² | 0.450 |
| Linear regression test R² | −1.034 |
| Predicted recovery at typical settings | 76.02% |
| Predicted recovery at optimised settings | 117.94% |

Linear regression scored highest on validation, but its negative test R² showed poor generalisation to the later operating period. The optimiser also produced a prediction above the physical limit of 100%, even within the individual training bounds. These results rule out using the fitted model to recommend plant setpoints.

The [analysis notebook](gold_recovery.ipynb) contains the implementation, plots, and detailed results. The repository also includes the dataset and Python dependencies used to reproduce the analysis.
