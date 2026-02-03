import sys
import os
import time
import math
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

def calculate_transformer_flops(batch, seq_len, embed, ffn_dim):
    term1 = 4 * batch * seq_len * (embed ** 2)
    term2 = 2 * batch * (seq_len ** 2) * embed
    term3 = 2 * batch * seq_len * embed * ffn_dim
    
    forward_flops = 2 * (term1 + term2 + term3)
    backward_flops = 2 * forward_flops
    return forward_flops + backward_flops

def main():
    print("--- PureAttention GFLOPS Benchmark (Block Timing) ---")
    np.random.seed(42)

    BATCH = 256
    SEQ_LEN = 64
    EMBED = 128
    HEADS = 4
    FFN_DIM = 128
    STEPS = 500
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

    flops_per_step = calculate_transformer_flops(BATCH, SEQ_LEN, EMBED, FFN_DIM)
    print(f"Theoretical GFLOPS per step target: {flops_per_step / 1e9:.4f}")

    csv_filename = 'gflops_pa.csv'
    print(f"Logging GFLOPS to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['step', 'gflops']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        buffer = []
        
        block_start_time = time.time()
        
        for step in range(STEPS):
            optim.zero_grad()
            output = model.forward(input_data)
            loss = criterion.forward(output, target_data)

            loss_val = loss.to_host()[0]

            if not math.isfinite(loss_val) or loss_val > 1e6:
                print(f"!!! Loss explosion at step {step}: {loss_val}")
                break

            loss.backward()
            optim.step()

            if (step + 1) % 50 == 0:
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
