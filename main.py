import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class NeuralNet3:
    def __init__(self):
        # fetch dataset
        drug_consumption = fetch_ucirepo(id=373)

        # data
        X = drug_consumption.data.features
        y = drug_consumption.data.targets["ecstasy"]

        # group the target into people who have and haven't used ecstasy recently
        # 0 means never used or last used over a decade ago
        # 1 means used within the last decade
        y = y.isin(["CL2", "CL3", "CL4", "CL5", "CL6"]).astype(int)

        # keep the target with the features for quick inspection
        self.raw_input = pd.concat(
            [X, y.rename("ecstasy")], axis=1
        )

        print(self.raw_input.head())
        print(self.raw_input.shape)

    def preprocess(self, seed=42):
        # work on a copy so the raw data stays unchanged
        self.processed_data = self.raw_input.copy()

        # check the data before preprocessing
        print("Missing values:")
        print(self.processed_data.isnull().sum())

        print("Duplicate rows:")
        print(self.processed_data.duplicated().sum())

        print("Data types:")
        print(self.processed_data.dtypes)

        # separate the input columns from the value we want to predict
        X = self.processed_data.drop(columns=["ecstasy"])
        y = self.processed_data["ecstasy"]

        # 70% for training and split the rest into validation and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.30,
            random_state=seed,
            stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=seed,
            stratify=y_temp
        )

        # categorical columns need encoding; the rest are numeric
        categorical_cols = ["gender", "country", "ethnicity"]
        numerical_cols = [
            col for col in X.columns if col not in categorical_cols
        ]

        # turn missing categories into numeric indicator columns
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        # fill missing numeric values, then scale
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # apply preprocessing to each type of columns
        self.preprocessor = ColumnTransformer([
            ("categorical", categorical_pipeline, categorical_cols),
            ("numerical", numerical_pipeline, numerical_cols)
        ])

        # learn encoding and scaling from training data
        self.X_train = self.preprocessor.fit_transform(X_train)

        # reuse those learned steps on val and test data
        self.X_val = self.preprocessor.transform(X_val)
        self.X_test = self.preprocessor.transform(X_test)

        # store the targets as arrays 
        self.y_train = y_train.to_numpy()
        self.y_val = y_val.to_numpy()
        self.y_test = y_test.to_numpy()

        # check split and the training labels
        print("\nPreprocessing Results:")
        print("Training samples:", len(self.X_train))
        print("Validation samples:", len(self.X_val))
        print("Testing samples:", len(self.X_test))

        print("Training shape:", self.X_train.shape)
        print("Validation shape:", self.X_val.shape)
        print("Testing shape:", self.X_test.shape)

        print("Number of input features:", self.X_train.shape[1])

        print("Training class distribution:",
              np.unique(self.y_train, return_counts=True))

        return 0

    def initialize_parameters(self, hidden=None, activation="relu", seed=42):
        if hidden is None:
            hidden = [64]

        self.hidden = hidden
        self.activation = activation

        # add the input and output sizes around the hidden layer sizes
        layer_sizes = [self.X_train.shape[1]] + hidden + [1]

        self.weights = []
        self.biases = []

        # random num generator
        rng = np.random.default_rng(seed)

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            # choose an initialization that fits the hidden activation
            if i == len(layer_sizes) - 2:
                # sigmoid for this binary classification task
                limit = np.sqrt(6 / (input_size + output_size))
                W = rng.uniform(
                    -limit, limit, (input_size, output_size)
                )

            elif activation == "relu":
                # He initialization works well with relu
                W = rng.standard_normal((input_size, output_size)) * np.sqrt(2 / input_size)

            else:
                # Xavier initialization works well with sigmoid and tanh
                limit = np.sqrt(6 / (input_size + output_size))
                W = rng.uniform(-limit, limit, (input_size, output_size))

            b = np.zeros((1, output_size))

            self.weights.append(W)
            self.biases.append(b)

        # start momentum with zero velocity for each weight and bias
        self.velocity_w = [
            np.zeros_like(W) for W in self.weights
        ]

        self.velocity_b = [
            np.zeros_like(b) for b in self.biases
        ]

    def activate(self, Z):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

        elif self.activation == "tanh":
            return np.tanh(Z)

        elif self.activation == "relu":
            return np.maximum(0, Z)

    def activation_derivative(self, Z):
        if self.activation == "sigmoid":
            A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            return A * (1 - A)

        elif self.activation == "tanh":
            return 1 - np.tanh(Z) ** 2

        elif self.activation == "relu":
            derivative = np.zeros_like(Z, dtype=float)
            derivative[Z > 0] = 1.0
            return derivative

    def forward(self, X):
        activations = [X]
        z_values = []

        A = X

        # run each hidden layer in order
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.activate(Z)

            z_values.append(Z)
            activations.append(A)

        # use sigmoid at the output to get a probability between 0 and 1
        Z = A @ self.weights[-1] + self.biases[-1]
        A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

        z_values.append(Z)
        activations.append(A)

        return A, activations, z_values

    def compute_loss(self, y, predictions):
        y = np.asarray(y).reshape(-1, 1)

        # clip probabilities so the logarithms stay finite
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)

        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

        return loss

    def backward(self, y, activations, z_values):
        y = np.asarray(y).reshape(-1, 1)
        m = len(y)

        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        # for sigmoid with binary cross-entropy, this is the output gradient
        dZ = activations[-1] - y

        # work backward through the layers to calculate each gradient
        for i in reversed(range(len(self.weights))):
            weight_gradients[i] = (activations[i].T @ dZ) / m

            bias_gradients[i] = np.sum(dZ, axis=0, keepdims=True) / m

            # pass the gradient back through the previous hidden activation
            if i > 0:
                dA = dZ @ self.weights[i].T
                dZ = dA * self.activation_derivative(z_values[i - 1])

        return weight_gradients, bias_gradients

    def update_parameters(self, weight_gradients, bias_gradients, learning_rate=0.01, optimizer="sgd", momentum=0.9):

        for i in range(len(self.weights)):

            if optimizer == "sgd":
                # plain mini-batch gradient descent
                self.weights[i] -= learning_rate * weight_gradients[i]
                self.biases[i] -= learning_rate * bias_gradients[i]

            elif optimizer == "momentum":
                # momentum keeps part of the previous update
                self.velocity_w[i] = (momentum * self.velocity_w[i] - learning_rate * weight_gradients[i])

                self.velocity_b[i] = (momentum * self.velocity_b[i] - learning_rate * bias_gradients[i])

                self.weights[i] += self.velocity_w[i]
                self.biases[i] += self.velocity_b[i]

    def train(self, epochs=100, batch_size=32, learning_rate=0.01, optimizer="sgd", momentum=0.9, seed=42, verbose=True):

        n_samples = len(self.X_train)

        self.train_losses = []
        self.val_losses = []

        rng = np.random.default_rng(seed)

        for epoch in range(epochs):

            # shuffle examples so batches differ from epoch to epoch
            indices = rng.permutation(n_samples)

            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            # update the network one mini-batch at a time
            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # get predictions and values needed for backpropagation
                predictions, activations, z_values = self.forward(X_batch)

                weight_gradients, bias_gradients = self.backward(y_batch, activations, z_values)

                # apply the gradients using the selected optimizer
                self.update_parameters(weight_gradients, bias_gradients, learning_rate=learning_rate, optimizer=optimizer, momentum=momentum)

            # track training loss after each full pass through the data
            train_predictions, _, _ = self.forward(self.X_train)
            train_loss = self.compute_loss(self.y_train, train_predictions)

            # track validation loss without using it to update the weights
            val_predictions, _, _ = self.forward(self.X_val)
            val_loss = self.compute_loss(self.y_val, val_predictions)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Train Loss: {train_loss:.4f} "
                    f"- Val Loss: {val_loss:.4f}"
                )

    def predict(self, X, threshold=0.5):
        # turn the predicted probabilities into class labels based on the threshold
        probabilities, _, _ = self.forward(X)
        return (probabilities.ravel() >= threshold).astype(int)

    def evaluate(self, X, y):
        # use macro f1 in case of class imbalance
        predictions = self.predict(X)
        return {
            "accuracy": accuracy_score(y, predictions),
            "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        }

    def compare_optimizers(self, activation, hidden, learning_rate, epochs, batch_size, momentum, seed, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        losses = {}

        # reset the weights and shuffle order so the optimizer is the only difference
        for optimizer in ("sgd", "momentum"):
            self.initialize_parameters(
                hidden=hidden, activation=activation, seed=seed
            )
            self.train(epochs=epochs,batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer, momentum=momentum, seed=seed, verbose=False)
            losses[optimizer] = self.val_losses.copy()

        # keep each epoch's validation loss for the csv and comparison plot
        results = pd.DataFrame({
            "epoch": np.arange(1, epochs + 1),
            "sgd_validation_loss": losses["sgd"],
            "momentum_validation_loss": losses["momentum"],
        })
        results_path = output_dir / "optimizer_comparison.csv"
        results.to_csv(results_path, index=False)

        plt.figure()
        for optimizer in ("sgd", "momentum"):
            # mark the lowest validation loss and when it happened
            best_epoch = int(np.argmin(losses[optimizer])) + 1
            best_loss = min(losses[optimizer])
            plt.plot(results["epoch"], losses[optimizer], label=f"{optimizer} (min {best_loss:.4f}, epoch {best_epoch})")
            print(
                f"{optimizer}: minimum validation loss {best_loss:.4f} "
                f"at epoch {best_epoch}"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        plt.title(
            f"SGD vs. momentum — {activation}, hidden={hidden}, "
            f"learning rate={learning_rate}"
        )
        plt.legend()
        plt.tight_layout()
        plot_path = output_dir / "optimizer_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Per-epoch validation losses saved to {results_path}")
        print(f"Optimizer comparison plot saved to {plot_path}")
        return results

    # run all experiment combinations and save the results
    def run_experiments(self, activations, learning_rates, architectures, optimizers, epochs, batch_size, momentum, seed, output_dir, logistic_max_iter):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        best_by_activation = {}

        # one run for every hyperparameter combo
        configurations = product(
            activations, learning_rates, architectures, optimizers
        )
        total = (
            len(activations)
            * len(learning_rates)
            * len(architectures)
            * len(optimizers)
        )

        for run_number, (activation, learning_rate, hidden, optimizer) in enumerate(
            configurations, start=1
        ):
            # print specifications for this experiment, then setup and train
            print(
                f"Experiment {run_number}/{total}: activation={activation}, "
                f"learning_rate={learning_rate}, hidden={list(hidden)}, "
                f"optimizer={optimizer}"
            )
            self.initialize_parameters(
                hidden=list(hidden), activation=activation, seed=seed
            )
            self.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer, momentum=momentum, seed=seed, verbose=False)

            train_metrics = self.evaluate(self.X_train, self.y_train)
            val_metrics = self.evaluate(self.X_val, self.y_val)
            result = {
                "activation": activation,
                "hidden_layers": str(list(hidden)),
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "momentum": momentum if optimizer == "momentum" else "",
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
            results.append(result)

            # choose activation model based on the best validation results
            current_best = best_by_activation.get(activation)
            if (
                current_best is None
                or val_metrics["macro_f1"] > current_best["validation_macro_f1"]
            ):
                # keep a copy since the next run would replace the network parameters
                best_by_activation[activation] = {
                    **result,
                    "weights": [weight.copy() for weight in self.weights],
                    "biases": [bias.copy() for bias in self.biases],
                    "train_losses": self.train_losses.copy(),
                    "val_losses": self.val_losses.copy(),
                }

        # save final metrics
        results_frame = pd.DataFrame(results)
        results_path = output_dir / "experiment_results.csv"
        results_frame.to_csv(results_path, index=False)
        print("\nExperiment results (sorted by activation and validation macro-F1):")
        print(
            results_frame.sort_values(
                ["activation", "validation_macro_f1"],
                ascending=[True, False],
            ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print(f"\nFull results saved to {results_path}")

        test_results = []
        for activation, best in best_by_activation.items():
            # now run on test set as the final evaluation
            self.activation = activation
            self.weights = [weight.copy() for weight in best["weights"]]
            self.biases = [bias.copy() for bias in best["biases"]]
            test_metrics = self.evaluate(self.X_test, self.y_test)

            test_results.append({
                "model": f"Neural network ({activation})",
                "activation": activation,
                "hidden_layers": best["hidden_layers"],
                "optimizer": best["optimizer"],
                "learning_rate": best["learning_rate"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            })

            # generate the learning curve plots for the best combo for each activation function
            plt.figure()
            plt.plot(best["train_losses"], label="Training loss")
            plt.plot(best["val_losses"], label="Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Binary cross-entropy loss")
            plt.title(
                f"{activation} — best validation macro-F1 "
                f"({best['validation_macro_f1']:.3f})"
            )
            plt.legend()
            plt.tight_layout()
            plot_path = output_dir / f"learning_curve_{activation}.png"
            plt.savefig(plot_path)
            plt.close()
            print(
                f"Best {activation} configuration by validation macro-F1: "
                f"hidden={best['hidden_layers']}, optimizer={best['optimizer']}, "
                f"learning_rate={best['learning_rate']}"
            )

        # use logistic regression to create a baseline on the same training split and then evaluate on the test set
        baseline = LogisticRegression(
            max_iter=logistic_max_iter,
            random_state=seed,
        )
        baseline.fit(self.X_train, self.y_train)
        baseline_predictions = baseline.predict(self.X_test)
        test_results.append({
            "model": "Logistic regression baseline",
            "activation": "",
            "hidden_layers": "",
            "optimizer": "",
            "learning_rate": "",
            "test_accuracy": accuracy_score(self.y_test, baseline_predictions),
            "test_macro_f1": f1_score(
                self.y_test,
                baseline_predictions,
                average="macro",
                zero_division=0,
            ),
        })

        test_frame = pd.DataFrame(test_results)
        test_path = output_dir / "best_test_results.csv"
        test_frame.to_csv(test_path, index=False)
        print("\nTest results for selected configurations and baseline:")
        print(test_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"Test results saved to {test_path}")
        print(f"Learning curves saved in {output_dir}")
        return results_frame, test_frame


if __name__ == "__main__":
    # user input for the specifications of the network/experiments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run the required hyperparameter comparison and baseline.",
    )
    parser.add_argument(
        "--compare-optimizers",
        action="store_true",
        help="Compare SGD and momentum on the same model configuration.",
    )
    parser.add_argument("--hidden", nargs="+", type=int, default=[64])
    parser.add_argument("--activation", choices=["sigmoid", "tanh", "relu"], default="relu")
    parser.add_argument("--optimizer", choices=["sgd", "momentum"], default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-activations", nargs="+", choices=["sigmoid", "tanh", "relu"], default=["sigmoid", "tanh", "relu"])
    parser.add_argument("--experiment-learning-rates", nargs="+", type=float, default=[0.01, 0.001])
    parser.add_argument("--experiment-architectures", nargs="+", default=["64", "64,32"], help="Comma-separated hidden-layer sizes, e.g. 64 64,32.")
    parser.add_argument("--experiment-optimizers", nargs="+", choices=["sgd", "momentum"], default=["sgd", "momentum"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--logistic-max-iter", type=int, default=1000)
    args = parser.parse_args()

    # error checking for invalid user inputs
    if args.run_experiments and args.compare_optimizers:
        parser.error("--run-experiments and --compare-optimizers cannot be used together.")
    if len(args.hidden) not in [1, 2] or any(size <= 0 for size in args.hidden):
        parser.error("State one or two hidden layers with positive sizes.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size have to be positive.")
    if args.learning_rate <= 0 or any(rate <= 0 for rate in args.experiment_learning_rates):
        parser.error("Learning rates have to be positive.")
    if args.logistic_max_iter <= 0:
        parser.error("--logistic-max-iter have to be positive.")

    try:
        experiment_architectures = [
            tuple(int(size) for size in architecture.split(","))
            for architecture in args.experiment_architectures
        ]
    except ValueError:
        parser.error("Experiment architectures have to contain comma-separated integers")
    if any(
        not 1 <= len(architecture) <= 2
        or any(size <= 0 for size in architecture)
        for architecture in experiment_architectures
    ):
        parser.error(
            "Experiment architectures have to specify one or two positive hidden-layer sizes"
        )

    neural_network = NeuralNet3()
    neural_network.preprocess(seed=args.seed)

    print(f"Random seed: {args.seed}")

    # run experiments, compare optimizers, or train a single model
    if args.run_experiments:
        neural_network.run_experiments(
            activations=args.experiment_activations,
            learning_rates=args.experiment_learning_rates,
            architectures=experiment_architectures,
            optimizers=args.experiment_optimizers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            momentum=args.momentum,
            seed=args.seed,
            output_dir=args.output_dir,
            logistic_max_iter=args.logistic_max_iter,
        )
    elif args.compare_optimizers:
        neural_network.compare_optimizers(
            activation=args.activation,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            momentum=args.momentum,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        neural_network.initialize_parameters(
            hidden=args.hidden,
            activation=args.activation,
            seed=args.seed,
        )
        neural_network.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            momentum=args.momentum,
            seed=args.seed,
        )

        train_metrics = neural_network.evaluate(
            neural_network.X_train, neural_network.y_train
        )
        val_metrics = neural_network.evaluate(
            neural_network.X_val, neural_network.y_val
        )
        print(f"\n{args.optimizer.upper()} Results:")
        print(f"Final Training Loss: {neural_network.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {neural_network.val_losses[-1]:.4f}")
        print(
            f"Training Accuracy: {train_metrics['accuracy']:.4f} "
            f"(Macro-F1: {train_metrics['macro_f1']:.4f})"
        )
        print(
            f"Validation Accuracy: {val_metrics['accuracy']:.4f} "
            f"(Macro-F1: {val_metrics['macro_f1']:.4f})"
        )