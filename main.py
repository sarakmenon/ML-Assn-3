import argparse
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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

    def initialize_parameters(self, hidden=[64], activation="relu", seed=42):
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

        loss = -np.mean(
            y * np.log(predictions)
            + (1 - y) * np.log(1 - predictions)
        )

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
                self.velocity_w[i] = (
                    momentum * self.velocity_w[i]
                    - learning_rate * weight_gradients[i]
                )

                self.velocity_b[i] = (
                    momentum * self.velocity_b[i]
                    - learning_rate * bias_gradients[i]
                )

                self.weights[i] += self.velocity_w[i]
                self.biases[i] += self.velocity_b[i]

    def train(self, epochs=100, batch_size=32, learning_rate=0.01, optimizer="sgd", momentum=0.9, seed=42):

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

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Train Loss: {train_loss:.4f} "
                    f"- Val Loss: {val_loss:.4f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", nargs="+", type=int, default=[64])
    parser.add_argument(
        "--activation",
        choices=["sigmoid", "tanh", "relu"],
        default="relu"
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "momentum"],
        default="sgd"
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if len(args.hidden) not in [1, 2] or any(size <= 0 for size in args.hidden):
        parser.error("Specify one or two hidden layers with positive sizes.")

    neural_network = NeuralNet3()
    neural_network.preprocess(seed=args.seed)

    neural_network.initialize_parameters(hidden=args.hidden, activation=args.activation, seed=args.seed)

    neural_network.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, optimizer=args.optimizer, momentum=args.momentum, seed=args.seed)

    print(f"\n{args.optimizer.upper()} Results:")
    print(f"Final Training Loss: {neural_network.train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {neural_network.val_losses[-1]:.4f}")