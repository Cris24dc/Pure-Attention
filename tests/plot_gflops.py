import matplotlib.pyplot as plt
import csv
import sys
import os

def read_gflops_csv(filename):
    steps = []
    gflops = []
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return steps, gflops
        
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(row['step'])
                gf = float(row['gflops'])
                steps.append(s)
                gflops.append(gf)
            except ValueError:
                continue
    return steps, gflops

def main():
    pa_file = 'gflops_pa.csv'
    to_file = 'gflops_torch.csv'
    
    steps_pa, gflops_pa = read_gflops_csv(pa_file)
    steps_to, gflops_to = read_gflops_csv(to_file)

    if not steps_pa and not steps_to:
        print("No data found to plot. Run benchmarks first.")
        sys.exit(1)

    print(f"Found {len(steps_pa)} steps for PureAttention")
    print(f"Found {len(steps_to)} steps for PyTorch")

    plt.figure(figsize=(12, 7))

    if steps_pa:
        avg_pa = sum(gflops_pa) / len(gflops_pa)
        plt.plot(steps_pa, gflops_pa, label=f'PureAttention (Avg: {avg_pa:.2f} GFLOPS)', color='blue', alpha=0.8)
        plt.axhline(y=avg_pa, color='blue', linestyle=':', alpha=0.5)

    if steps_to:
        avg_to = sum(gflops_to) / len(gflops_to)
        plt.plot(steps_to, gflops_to, label=f'PyTorch (Avg: {avg_to:.2f} GFLOPS)', color='red', linestyle='--', alpha=0.8)
        plt.axhline(y=avg_to, color='red', linestyle=':', alpha=0.5)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Throughput (GFLOPS)', fontsize=12)
    plt.title('Performance Benchmark: GFLOPS Comparison', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_file = 'gflops_comparison.pdf'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.savefig('gflops_comparison.png')

if __name__ == "__main__":
    main()
