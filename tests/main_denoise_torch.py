import sys
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln_final = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        x = self.ln_final(x)
        return x

def main():
    print("--- Denoising Task: Clean Signal Recovery (Torch Comparison) ---")
    np.random.seed(42)
    torch.manual_seed(42)

    BATCH = 256
    SEQ_LEN = 64
    EMBED = 128
    HEADS = 4
    FFN_DIM = 128
    STEPS = 2000
    LR = 5e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerLayer(EMBED, HEADS, FFN_DIM).to(device)

    phase = np.random.uniform(0, 2 * np.pi, size=(BATCH, 1, 1))
    s_indices = np.arange(SEQ_LEN)[None, :, None]
    e_indices = np.arange(EMBED)[None, None, :]
    
    clean_signal = np.sin(s_indices * 0.1 + e_indices * 0.05 + phase).astype(np.float32)
    noise = np.random.uniform(-0.5, 0.5, size=(BATCH, SEQ_LEN, EMBED)).astype(np.float32)
    
    input_array = clean_signal + noise
    target_array = clean_signal
    
    input_data = torch.from_numpy(input_array).to(device)
    target_data = torch.from_numpy(target_array).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    start_time = time.time()
    loss_history = []
    steps_history = []
    
    csv_filename = 'denoise_loss.csv'
    torch_csv_filename = 'denoise_loss_torch.csv'

    with open(torch_csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'loss_to'])
        writer.writeheader()

    print("Starting Torch Training Loop...")
    
    buffer = []

    for step in range(STEPS):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        
        loss_val = loss.item()

        buffer.append({'step': step, 'loss_to': f"{loss_val:.6f}"})

        if len(buffer) >= 50:
            with open(torch_csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['step', 'loss_to'])
                writer.writerows(buffer)
            buffer = []
        
        if not math.isfinite(loss_val) or loss_val > 1e6:
            print(f"!!! Loss explosion at step {step}: {loss_val}")
            break

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            loss_history.append(loss_val)
            steps_history.append(step)
            print(f"Step {step:03d} | Torch Loss: {loss_val:.6f}")

    if buffer:
        with open(torch_csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'loss_to'])
            writer.writerows(buffer)

    print(f"Merging {torch_csv_filename} into {csv_filename}...")
    
    pa_losses_map = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'loss_pa' in row and row['loss_pa'].strip():
                    pa_losses_map[int(row['step'])] = row['loss_pa']
    
    to_losses_map = {}
    with open(torch_csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            to_losses_map[int(row['step'])] = row['loss_to']

    all_steps = sorted(set(pa_losses_map.keys()) | set(to_losses_map.keys()))
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'loss_pa', 'loss_to'])
        writer.writeheader()
        
        rows = []
        for s in all_steps:
            row = {
                'step': s,
                'loss_pa': pa_losses_map.get(s, ''),
                'loss_to': to_losses_map.get(s, '')
            }
            rows.append(row)
        writer.writerows(rows)
    
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f}s")
    print(f"Final Loss: {loss_val:.6f}")

    try:
        pa_losses = []
        to_losses = []
        plot_steps = []
        
        for r in rows:
            s = int(r['step'])
            if r['loss_pa'] and r['loss_to'] and s % 50 == 0:
                pa_val = float(r['loss_pa'])
                to_val = float(r['loss_to'])
                pa_losses.append(pa_val)
                to_losses.append(to_val)
                plot_steps.append(s)

        if len(pa_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(plot_steps, pa_losses, label='PureAttention Loss')
            plt.plot(plot_steps, to_losses, label='PyTorch Loss', linestyle='--')
            plt.xlabel('Step')
            plt.ylabel('MSE Loss')
            plt.title('Denoising Task Training Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig('loss_comparison.pdf')
            print("Comparison plot saved to loss_comparison.pdf")
    except Exception as e:
        print(f"Could not creat comparison plot: {e}")

if __name__ == "__main__":
    main()
