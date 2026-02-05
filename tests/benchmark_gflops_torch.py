import sys
import os
import time
import math
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

def calculate_transformer_flops(batch, seq_len, embed, ffn_dim):
    term1 = 4 * batch * seq_len * (embed ** 2)
    term2 = 2 * batch * (seq_len ** 2) * embed
    term3 = 2 * batch * seq_len * embed * ffn_dim
    
    forward_flops = 2 * (term1 + term2 + term3)
    backward_flops = 2 * forward_flops
    
    total_step_flops = forward_flops + backward_flops
    return total_step_flops

def main():
    print("--- PyTorch GFLOPS Benchmark (Block Timing) ---")
    
    np.random.seed(42)
    torch.manual_seed(42)

    BATCH = 256
    SEQ_LEN = 64
    EMBED = 128
    HEADS = 4
    FFN_DIM = 128
    STEPS = 500
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

    flops_per_step = calculate_transformer_flops(BATCH, SEQ_LEN, EMBED, FFN_DIM)
    print(f"Theoretical GFLOPS per step target: {flops_per_step / 1e9:.4f}")

    csv_filename = 'gflops_torch.csv'
    print(f"Logging GFLOPS to {csv_filename}...")

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['step', 'gflops']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        buffer = []

        print("Warming up...")
        for _ in range(10):
            optimizer.zero_grad()
            out = model(input_data)
            loss = criterion(out, target_data)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        print("Benchmarking...")
        
        block_start_time = time.time()

        for step in range(STEPS):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target_data)
            
            loss_val = loss.item() 
            
            if not math.isfinite(loss_val) or loss_val > 1e6:
                print(f"!!! Loss explosion at step {step}: {loss_val}")
                break

            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                torch.cuda.synchronize()
                block_end_time = time.time()
                block_duration = block_end_time - block_start_time
                
                avg_throughput = (50 * flops_per_step) / block_duration / 1e9
                
                print(f"Step {step:03d} | Loss: {loss_val:.6f} | GFLOPS: {avg_throughput:.2f}")
                
                buffer.append({'step': step, 'gflops': f"{avg_throughput:.4f}"})
                
                block_start_time = time.time()

        if buffer:
            writer.writerows(buffer)

    print("Finished.")

if __name__ == "__main__":
    main()
