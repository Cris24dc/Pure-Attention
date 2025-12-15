import PureAttention as pa
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def log_gradient_stats(epoch, model):
    print(f"\n--- [DEBUG] Epoch {epoch} Gradient Stats ---")
    
    w1_stats = model.l1.weight.get_grad_stats()
    b1_stats = model.l1.bias.get_grad_stats()
    print(f"L1 Weights | Mean: {w1_stats['mean']:.5e} | Std: {w1_stats['std']:.5e} | Min: {w1_stats['min']:.5e} | Max: {w1_stats['max']:.5e}")
    print(f"L1 Bias    | Mean: {b1_stats['mean']:.5e} | Std: {b1_stats['std']:.5e} | Min: {b1_stats['min']:.5e} | Max: {b1_stats['max']:.5e}")

    w2_stats = model.l2.weight.get_grad_stats()
    b2_stats = model.l2.bias.get_grad_stats()
    print(f"L2 Weights | Mean: {w2_stats['mean']:.5e} | Std: {w2_stats['std']:.5e} | Min: {w2_stats['min']:.5e} | Max: {w2_stats['max']:.5e}")
    print("-" * 50)


class HousingModel(pa.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = pa.Linear(input_dim, 128)
        self.relu = pa.ReLU()
        self.l2 = pa.Linear(128, 1)

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.relu.forward(x)
        x = self.l2.forward(x)
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


def main():
    print("Loading data...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.reshape(-1, 1)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    BATCH_SIZE = 4000
    if BATCH_SIZE > len(X_train): BATCH_SIZE = len(X_train)
    
    X_batch = X_train[:BATCH_SIZE]
    y_batch = y_train[:BATCH_SIZE]

    B = X_batch.shape[0]
    IN = X_batch.shape[1]
    OUT = 1

    print("Transfer data to GPU...")
    input_tensor = pa.Tensor([B, IN], False)
    input_tensor.to_device(X_batch.flatten())

    target_tensor = pa.Tensor([B, OUT], False)
    target_tensor.to_device(y_batch.flatten())


    model = HousingModel(input_dim=IN)
    
    optimizer = pa.Adam(model.parameters(), lr=0.001)
    criterion = pa.MSE()

    print(f"Start training on {B} samples...")


    epochs = 5000
    for i in range(epochs):
        optimizer.zero_grad()

        pred = model.forward(input_tensor)
        
        loss = criterion.forward(pred, target_tensor)

        loss.backward(True)

        # debug
        if i % 50 == 0:
            val_loss = loss.to_host()[0]
            print(f"Epoch {i}, MSE Loss: {val_loss:.6f}")
            log_gradient_stats(i, model)

        optimizer.step()

    print("Done.")
    

    preds = pred.to_host()
    targets = y_batch.flatten()
    
    print("\nVerification")
    print(f"{'Prediction':<15} | {'Real':<15} | {'Diference':<15}")
    print("-" * 50)
    for k in range(5):
        p = preds[k]
        t = targets[k]
        print(f"{p:<15.4f} | {t:<15.4f} | {abs(p-t):<15.4f}")


if __name__ == "__main__":
    main()