import sys
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import csv

sys.path.append(os.path.abspath("build"))
import PureAttention as pa

class FeedForward(pa.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = pa.Linear(embed_dim, hidden_dim)
        self.relu = pa.ReLU()
        self.fc2 = pa.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        out = self.fc1.forward(x)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        return out

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

class TransformerLayer(pa.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.mha = pa.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = pa.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.ln2 = pa.LayerNorm(embed_dim)
        self.ln_final = pa.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.ln1.forward(x)
        attn_out = self.mha.forward(x_norm, x_norm, x_norm)
        x = pa.add(x, attn_out)

        x_norm = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        x = pa.add(x, ffn_out)

        x = self.ln_final.forward(x)

        return x

    def parameters(self):
        return (self.mha.parameters() +
                self.ln1.parameters() +
                self.ffn.parameters() +
                self.ln2.parameters() +
                self.ln_final.parameters())

def main():
    print("--- Denoising Task: Clean Signal Recovery (CSV Logging) ---")
    np.random.seed(42)

    BATCH = 256
    SEQ_LEN = 64
    EMBED = 128
    HEADS = 4
    FFN_DIM = 128
    STEPS = 2000
    LR = 5e-5

    model = TransformerLayer(EMBED, HEADS, FFN_DIM)

    phase = np.random.uniform(0, 2 * np.pi, size=(BATCH, 1, 1))
    s_indices = np.arange(SEQ_LEN)[None, :, None]
    e_indices = np.arange(EMBED)[None, None, :]
    
    clean_signal = np.sin(s_indices * 0.1 + e_indices * 0.05 + phase).astype(np.float32)
    noise = np.random.uniform(-0.5, 0.5, size=(BATCH, SEQ_LEN, EMBED)).astype(np.float32)
    
    input_array = clean_signal + noise
    target_array = clean_signal
    
    input_list = input_array.flatten().tolist()
    target_list = target_array.flatten().tolist()

    input_data = pa.Tensor([BATCH, SEQ_LEN, EMBED], False)
    target_data = pa.Tensor([BATCH, SEQ_LEN, EMBED], False)

    input_data.to_device(input_list)
    target_data.to_device(target_list)

    optim = pa.Adam(model.parameters(), lr=LR)
    criterion = pa.MSE()

    start_time = time.time()
    loss_history = []
    steps_history = []

    csv_filename = 'denoise_loss.csv'
    print(f"Logging loss to {csv_filename}...")
    
    fieldnames = ['step', 'loss_pa', 'loss_to']
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    buffer = []

    for step in range(STEPS):
        optim.zero_grad()
        output = model.forward(input_data)
        loss = criterion.forward(output, target_data)

        loss_val = loss.to_host()[0]

        buffer.append({'step': step, 'loss_pa': loss_val, 'loss_to': ''})

        if len(buffer) >= 50:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(buffer)
            buffer = []

        if not math.isfinite(loss_val) or loss_val > 1e6:
            print(f"!!! Loss explosion at step {step}: {loss_val}")
            break

        loss.backward()
        optim.step()

        if step % 50 == 0:
            loss_history.append(loss_val)
            steps_history.append(step)
            print(f"Step {step:03d} | Loss: {loss_val:.6f}")

    if buffer:
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(buffer)

    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f}s")
    print(f"Final Loss: {loss_val:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(steps_history, loss_history, label='Denoising Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Denoising Task Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_denoise_csv.pdf')
    print("Plot saved to loss_plot_denoise_csv.pdf")

if __name__ == "__main__":
    main()
