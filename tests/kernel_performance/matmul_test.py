import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

df = pd.read_csv('results.csv')

plt.figure(figsize=(12, 7))

kernels = df['Kernel'].unique()
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']

label_map = {
    "0_Naive": "Naive (Global Memory)",
    "1_Tiled": "Shared Memory Tiling",
    "2_Vectorized_A": "Tiled + Vectorized Mem",
    "3_Coarse": "Tiled + Thread Coarsening",
    "4_Ultimate_Optimized": "Ultimate (Tiled + Coarse + Vec)"
}

for i, kernel_name in enumerate(kernels):
    data = df[df['Kernel'] == kernel_name]
    
    x = data['MatrixSize'].values
    y = data['GFLOPS'].values
    
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    display_name = label_map.get(kernel_name, kernel_name)

    if len(x) > 3:
        try:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=2) 
            y_smooth = spl(x_smooth)
            y_smooth = np.clip(y_smooth, 0, None)
            
            plt.plot(x_smooth, y_smooth, label=display_name, linewidth=2.5, color=colors[i % len(colors)])
        except:
            plt.plot(x, y, label=display_name, linewidth=2.5, color=colors[i % len(colors)])
    else:
        plt.plot(x, y, label=display_name, linewidth=2.5, color=colors[i % len(colors)])

    plt.scatter(x, y, color=colors[i % len(colors)], alpha=0.4, s=30)

plt.xlabel('Matrix Size (N)', fontsize=12)
plt.ylabel('Throughput (GFLOPS)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(title='Optimization Techniques', fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()

plt.savefig('matmul_performance.pdf', format='pdf', dpi=300)
plt.show()
