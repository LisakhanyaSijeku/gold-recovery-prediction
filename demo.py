"""Gold recovery modelling and decision support using synthetic data."""
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "feed_grade", "particle_size", "solids_fraction",
    "collector_dose", "frother_dose", "airflow", "pulp_depth",
)


def synthetic_observations():
    rng = np.random.default_rng(42)
    inputs = rng.uniform(0.2, 0.8, size=(1600, len(FEATURES)))
    inputs[:, 0] += 0.07 * np.sin(np.arange(len(inputs)) / 120)
    grade, size, solids, collector, frother, air, depth = inputs.T
    recovery = (
        68 + 12 * grade - 8 * size + 4 * solids
        - 30 * (collector - 0.62) ** 2 + 5 * frother
        - 45 * (air - 0.57) ** 2 - 6 * depth
        + rng.normal(0, 1.5, len(inputs))
    )
    inputs[::109, 5] = -1
    inputs[::73, 1] = np.nan
    valid = np.isfinite(inputs).all(axis=1) & (inputs >= 0).all(axis=1)
    return inputs[valid], recovery[valid]


def main():
    inputs, recovery = synthetic_observations()
    split_points = [int(fraction * len(inputs)) for fraction in (0.70, 0.85)]
    train, validation, test = np.split(inputs, split_points)
    y_train, y_validation, y_test = np.split(recovery, split_points)

    candidates = {
        "Linear regression": LinearRegression(),
        "K-nearest neighbours": KNeighborsRegressor(n_neighbors=15),
        "Neural network": MLPRegressor(
            hidden_layer_sizes=(32, 16), random_state=42, max_iter=3000
        ),
    }
    fitted = {}
    validation_errors = {}
    for name, estimator in candidates.items():
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(train, y_train)
        fitted[name] = model
        error = mean_absolute_error(y_validation, model.predict(validation))
        validation_errors[name] = error
        print(f"{name} validation MAE = {error:.2f} percentage points")

    winner = min(validation_errors, key=validation_errors.get)
    model = fitted[winner]
    predictions = model.predict(test)
    print(f"Selected model = {winner}")
    print(f"Test MAE = {mean_absolute_error(y_test, predictions):.2f} percentage points")
    print(f"Test R² = {r2_score(y_test, predictions):.3f}")

    importance = permutation_importance(
        model, test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=42,
    )
    for index in np.argsort(importance.importances_mean)[::-1]:
        print(f"{FEATURES[index]} permutation importance = {importance.importances_mean[index]:.2f}")

    typical = np.median(train, axis=0, keepdims=True)
    settings = train.copy()
    settings[:, :3] = typical[0, :3]
    setting_predictions = model.predict(settings)
    best = np.argmax(setting_predictions)
    print(f"Typical predicted recovery = {model.predict(typical)[0]:.2f}%")
    print(f"Best screened predicted recovery = {setting_predictions[best]:.2f}%")
    for name, value in zip(FEATURES[3:], settings[best, 3:]):
        print(f"{name} = {value:.3f} (synthetic scale)")


if __name__ == "__main__":
    main()
