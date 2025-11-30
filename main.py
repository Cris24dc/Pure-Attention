import PureAttention as pa
import numpy as np

class Network(pa.Module):
    def __init__(self):
        super().__init__()
        self.l1 = pa.Linear(10, 10000)
        self.relu = pa.ReLU()
        self.l2 = pa.Linear(10000,1000)

    def forward(self, x): 
        return self.l2.forward((self.relu.forward((self.l1.forward(x)))))
        
    def parameters(self):
        return self.l1.parameters()



def print_tensor(title, tensor, cols):
    data = tensor.to_host()
    print(f"\n{title}:")
    for i, val in enumerate(data):
        print(f"{val:.5f} ", end="")
        if (i + 1) % cols == 0:
            print()

def main():
    B = 3
    IN = 5
    H = 6
    OUT = 3

    input_data = [
        # Batch 0
        0.5, -0.3, 100.2, -0.89587, 0.72678,
        # Batch 1
        0.1, 0.9, -0.5, 0.759687, 0.256784,
        # Batch 2
        0.32, 0.13, -0.9, -0.01, 0.3687
    ]

    input_np = np.array(input_data, dtype=np.float32)

    model = Network()

    input_tensor = pa.Tensor([B, IN], False) 
    input_tensor.to_device(input_np)

    for i in range(10000):
        out_tensor=model.forward(input_tensor)

    l1 = pa.Linear(IN, H)
    relu = pa.ReLU()
    l2 = pa.Linear(H, OUT)

    # h = l1.forward(input_tensor)
    # a = relu.forward(h)
    # y = l2.forward(a)



    # print("Input:")
    # for i, val in enumerate(input_data):
    #     print(f"{val} ", end="")
    #     if (i + 1) % IN == 0: print()

    # print_tensor("After L1 (h)", h, H)
    # print_tensor("After ReLU (a)", a, H)
    # print_tensor("Final Output (y)", y, OUT)

    # y.backward()

if __name__ == "__main__":
    main()
