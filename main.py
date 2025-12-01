import PureAttention as pa
import numpy as np

class Network(pa.Module):
    def __init__(self):
        super().__init__()
        self.l1 = pa.Linear(5, 6)
        self.relu = pa.ReLU()
        self.l2 = pa.Linear(6, 3)

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.relu.forward(x)
        x = self.l2.forward(x)
        return x

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

def main():
    B = 3
    IN = 5
    H = 6
    OUT = 3

    input_data = np.array([
        0.5, -0.3, 100.2, -0.89587, 0.72678,
        0.1, 0.9, -0.5, 0.759687, 0.256784,
        0.32, 0.13, -0.9, -0.01, 0.3687
    ], dtype=np.float32)

    target_data = np.array([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ], dtype=np.float32)

    input_tensor = pa.Tensor([B, IN], False)
    input_tensor.to_device(input_data)

    target_tensor = pa.Tensor([B, OUT], False)
    target_tensor.to_device(target_data)

    model = Network()

    # 1. Instantiere
    criterion = pa.MSELoss()

    for i in range(10):
        pred = model.forward(input_tensor)

        # 2. Apelare Concreta (Explicit .forward)
        loss = criterion.forward(pred, target_tensor)

        print(f"Epoch {i+1}, Loss: {loss.to_host()[0]:.6f}")

        loss.backward()

if __name__ == "__main__":
    main()