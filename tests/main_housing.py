import PureAttention as pa
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

class HousingModel(pa.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l1 = pa.Linear(input_dim, 128)
        self.relu = pa.ReLU()
        self.l2 = pa.Linear(128, 256)
        self.relu2 = pa.ReLU()
        self.l3 = pa.Linear(256, 64)
        self.relu3 = pa.ReLU()
        self.l4 = pa.Linear(64, 1)

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.relu.forward(x)
        x = self.l2.forward(x)
        x = self.relu2.forward(x)
        x = self.l3.forward(x)
        x = self.relu3.forward(x)
        x = self.l4.forward(x)
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters() + self.l3.parameters()

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

    TOTAL_SAMPLES = len(X_train)
    BATCH_SIZE = 2048
    N_BATCHES = TOTAL_SAMPLES // BATCH_SIZE

    print(f"Total train samples: {TOTAL_SAMPLES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Batches per epoch: {N_BATCHES}")

    IN = X_train.shape[1]
    OUT = 1

    input_tensor = pa.Tensor([BATCH_SIZE, IN], False)
    target_tensor = pa.Tensor([BATCH_SIZE, OUT], False)

    model = HousingModel(input_dim=IN)

    optimizer = pa.Adam(model.parameters(), lr=7.5e-8)
    criterion = pa.MSE()

    print("Start training...")
    epochs = 2000

    start_time = time.time()
    loss_history = []
    epoch_history = []

    for epoch in range(epochs):
        perm_indices = np.random.permutation(TOTAL_SAMPLES)
        X_train_shuffled = X_train[perm_indices]
        y_train_shuffled = y_train[perm_indices]

        epoch_loss = 0.0

        for b in range(N_BATCHES):
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE

            x_batch_cpu = X_train_shuffled[start:end]
            y_batch_cpu = y_train_shuffled[start:end]

            input_tensor.to_device(x_batch_cpu.flatten())
            target_tensor.to_device(y_batch_cpu.flatten())

            optimizer.zero_grad()

            pred = model.forward(input_tensor)
            loss = criterion.forward(pred, target_tensor)

            loss.backward(True)
            optimizer.step()

            batch_loss = loss.to_host()[0]
            epoch_loss += batch_loss

        avg_loss = epoch_loss / N_BATCHES
        loss_history.append(avg_loss)
        epoch_history.append(epoch)
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    X_val = X_test[:BATCH_SIZE]
    y_val = y_test[:BATCH_SIZE]

    input_tensor.to_device(X_val.flatten())

    pred_final = model.forward(input_tensor)
    preds_host = pred_final.to_host()
    targets_host = y_val.flatten()

    print("\nValidation")
    print(f"{'Prediction':<15} | {'Real':<15} | {'Difference':<15}")
    print("-" * 50)
    for k in range(10):
        p = preds_host[k]
        t = targets_host[k]
        print(f"{p:<15.4f} | {t:<15.4f} | {abs(p-t):<15.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_history, loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('California Housing')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_housing.pdf')
    print("\nLoss plot saved to loss_plot_housing.pdf")

if __name__ == "__main__":
    main()