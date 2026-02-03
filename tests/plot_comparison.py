import matplotlib.pyplot as plt
import csv
import sys
import os

def main():
    csv_filename = 'denoise_loss.csv'
    
    if not os.path.exists(csv_filename):
        print(f"Error: {csv_filename} not found.")
        sys.exit(1)

    steps = []
    loss_pa = []
    loss_to = []

    print(f"Reading {csv_filename}...")
    
    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(row['step'])
                
                pa_val = None
                if row.get('loss_pa') and row['loss_pa'].strip():
                    pa_val = float(row['loss_pa'])
                
                to_val = None
                if row.get('loss_to') and row['loss_to'].strip():
                    to_val = float(row['loss_to'])

                if pa_val is not None:
                    loss_pa.append((s, pa_val))
                
                if to_val is not None:
                    loss_to.append((s, to_val))
                    
            except ValueError:
                continue

    if not loss_pa and not loss_to:
        print("No valid data found to plot.")
        sys.exit(0)

    plt.figure(figsize=(12, 7))

    if loss_pa:
        xs, ys = zip(*loss_pa)
        plt.plot(xs, ys, label='PureAttention', color='blue', alpha=0.8, linewidth=1.5)

    if loss_to:
        xs, ys = zip(*loss_to)
        plt.plot(xs, ys, label='PyTorch', color='red', linestyle='--', alpha=0.8, linewidth=1.5)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Denoising Task: PureAttention vs PyTorch', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_file = 'comparison_plot_standalone.pdf'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    plt.savefig('comparison_plot_standalone.png')

if __name__ == "__main__":
    main()
