import PureAttention as pa
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearSanityCheck(pa.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = pa.Linear(input_dim, 1)

    def forward(self, x):
        return self.l1.forward(x)

    def parameters(self):
        return self.l1.parameters()

def solve_least_squares_numpy(X, y):
    m = X.shape[0]
    X_b = np.c_[X, np.ones((m, 1))]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    weights = theta_best[:-1].T
    bias = theta_best[-1]
    return weights, bias

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    BATCH_SIZE = len(X_train)

    X_batch = X_train[:BATCH_SIZE]
    y_batch = y_train[:BATCH_SIZE]

    B = X_batch.shape[0]
    IN = X_batch.shape[1]
    OUT = 1

    true_weights, true_bias = solve_least_squares_numpy(X_batch, y_batch)

    print(f"Ideal Bias: {true_bias[0]:.5f}")

    input_tensor = pa.Tensor([B, IN], False)
    input_tensor.to_device(X_batch.flatten())

    target_tensor = pa.Tensor([B, OUT], False)
    target_tensor.to_device(y_batch.flatten())

    model = LinearSanityCheck(input_dim=IN)

    optimizer = pa.Adam(model.parameters(), lr=0.0001)
    criterion = pa.MSE()

    epochs = 30000

    for i in range(epochs):
        optimizer.zero_grad()
        pred = model.forward(input_tensor)
        loss = criterion.forward(pred, target_tensor)
        loss.backward(True)

        if i % 1000 == 0:
            val_loss = loss.to_host()[0]
            print(f"Epoch {i}, MSE Loss: {val_loss:.6f}")

            if np.isnan(val_loss) or np.isinf(val_loss):
                break

        optimizer.step()

    cuda_weights = np.array(model.l1.weight.to_host()).reshape(IN, 1)
    cuda_bias = np.array(model.l1.bias.to_host())

    weights_diff = np.abs(cuda_weights - true_weights.T)
    bias_diff = np.abs(cuda_bias - true_bias)

    print("-" * 60)
    print(f"{'Param':<15} | {'PureAttn':<15} | {'LstSquares':<15} | {'Diff':<15}")
    print("-" * 60)

    print(f"{'Bias':<15} | {cuda_bias[0]:<15.5f} | {true_bias[0]:<15.5f} | {bias_diff[0]:<15.5f}")

    for k in range(IN):
        w_cuda = cuda_weights[k][0]
        w_np = true_weights[0][k]
        diff = weights_diff[k][0]
        print(f"{f'W[{k}]':<15} | {w_cuda:<15.5f} | {w_np:<15.5f} | {diff:<15.5f}")
    print("-" * 60)

if __name__ == "__main__":
    main()

