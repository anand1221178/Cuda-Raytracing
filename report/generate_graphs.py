import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style for professional graphs
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Data from terminal output
# Main performance comparison
global_time = 74944.8  # ms
const_time = 105289   # ms
global_rays_sec = 186.762  # million rays/sec
const_rays_sec = 132.937   # million rays/sec
global_bandwidth = 7.00572  # GB/s
const_bandwidth = 4.98666   # GB/s
theoretical_bandwidth = 256.256  # GB/s

# Resolution scaling data
resolutions = ['360p', '720p', '1080p', '4K']
resolution_pixels = [640*360, 1280*720, 1920*1080, 3840*2160]
estimated_times = [8327.2, 33308.8, 74944.8, 299779]  # ms

# Benchmark scaling data
benchmark_data = {
    'config': ['360p Low', '360p Medium', '360p High', '720p Low', '720p Medium', 
               '720p High', '1080p Low', '1080p Medium', '1080p High', '1080p Ultra'],
    'resolution': ['640x360', '640x360', '640x360', '1280x720', '1280x720', 
                   '1280x720', '1920x1080', '1920x1080', '1920x1080', '1920x1080'],
    'samples': [10, 50, 100, 10, 30, 100, 10, 50, 100, 300],
    'depth': [5, 10, 15, 5, 10, 20, 5, 15, 20, 30],
    'time_ms': [92.54, 596.87, 1745.63, 333.99, 1258.26, 6240.56, 704.68, 6056.02, 14172.45, 52582.96],
    'mrays_sec': [93.36, 144.76, 148.48, 103.47, 164.80, 221.52, 110.35, 192.60, 219.47, 266.19]
}

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# Figure 1: Memory Strategy Performance Comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Render time comparison
strategies = ['Global Memory', 'Constant Memory']
times = [global_time/1000, const_time/1000]  # Convert to seconds
colors = ['#2E86AB', '#A23B72']
bars1 = ax1.bar(strategies, times, color=colors, alpha=0.8)
ax1.set_ylabel('Render Time (seconds)')
ax1.set_title('Render Time Comparison')
ax1.grid(axis='y', alpha=0.3)
for bar, time in zip(bars1, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{time:.1f}s', ha='center', va='bottom')

# Ray throughput comparison
rays = [global_rays_sec, const_rays_sec]
bars2 = ax2.bar(strategies, rays, color=colors, alpha=0.8)
ax2.set_ylabel('Million Rays/Second')
ax2.set_title('Ray Throughput')
ax2.grid(axis='y', alpha=0.3)
for bar, ray in zip(bars2, rays):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{ray:.1f}', ha='center', va='bottom')

# Bandwidth utilization
bandwidth_util = [(global_bandwidth/theoretical_bandwidth)*100, 
                  (const_bandwidth/theoretical_bandwidth)*100]
bars3 = ax3.bar(strategies, bandwidth_util, color=colors, alpha=0.8)
ax3.set_ylabel('Bandwidth Utilization (%)')
ax3.set_title('Memory Bandwidth Efficiency')
ax3.set_ylim(0, 10)
ax3.grid(axis='y', alpha=0.3)
for bar, util in zip(bars3, bandwidth_util):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{util:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/memory_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/memory_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Resolution Scaling Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Time vs Resolution
ax1.plot(resolution_pixels, estimated_times, 'o-', color='#2E86AB', linewidth=2, markersize=8)
ax1.plot(resolution_pixels, estimated_times, '--', color='#A23B72', alpha=0.5, linewidth=2)
ax1.set_xlabel('Resolution (pixels)')
ax1.set_ylabel('Render Time (ms)')
ax1.set_title('Render Time Scaling with Resolution')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
for i, (res, time) in enumerate(zip(resolutions, estimated_times)):
    ax1.annotate(res, (resolution_pixels[i], time), 
                 textcoords="offset points", xytext=(0,10), ha='center')

# Pixels per second
pixels_per_sec = [pixels/(time/1000) for pixels, time in zip(resolution_pixels, estimated_times)]
ax2.bar(resolutions, [p/1e6 for p in pixels_per_sec], color='#F18F01', alpha=0.8)
ax2.set_xlabel('Resolution')
ax2.set_ylabel('Million Pixels/Second')
ax2.set_title('Rendering Throughput by Resolution')
ax2.grid(axis='y', alpha=0.3)
for i, (res, pps) in enumerate(zip(resolutions, pixels_per_sec)):
    ax2.text(i, pps/1e6 + 0.005, f'{pps/1e6:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/resolution_scaling.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/resolution_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Detailed Benchmark Analysis
df = pd.DataFrame(benchmark_data)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Ray throughput by configuration
x_pos = np.arange(len(df['config']))
bars = ax1.bar(x_pos, df['mrays_sec'], color='#2E86AB', alpha=0.8)
ax1.set_xlabel('Configuration')
ax1.set_ylabel('Million Rays/Second')
ax1.set_title('Ray Throughput by Configuration')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['config'], rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Sample scaling for 720p
df_720p = df[df['resolution'] == '1280x720']
ax2.plot(df_720p['samples'], df_720p['time_ms'], 'o-', color='#A23B72', linewidth=2, markersize=8)
ax2.set_xlabel('Samples per Pixel')
ax2.set_ylabel('Render Time (ms)')
ax2.set_title('Sample Scaling (720p Resolution)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Resolution impact on throughput
resolution_groups = df.groupby('resolution')['mrays_sec'].mean()
res_labels = ['360p', '720p', '1080p']
ax3.plot(resolution_pixels[:3], resolution_groups.values, 'o-', color='#F18F01', linewidth=2, markersize=8)
ax3.set_xlabel('Resolution (pixels)')
ax3.set_ylabel('Average MRays/Second')
ax3.set_title('Average Ray Throughput vs Resolution')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)
for i, (pixels, throughput) in enumerate(zip(resolution_pixels[:3], resolution_groups.values)):
    ax3.annotate(res_labels[i], (pixels, throughput), 
                 textcoords="offset points", xytext=(0,10), ha='center')

# Efficiency heatmap
# Create a matrix of efficiency (MRays/sec) for different samples and resolutions
unique_res = df['resolution'].unique()
unique_samples = sorted(df['samples'].unique())
efficiency_matrix = np.zeros((len(unique_res), len(unique_samples)))

for i, res in enumerate(unique_res):
    for j, samp in enumerate(unique_samples):
        data = df[(df['resolution'] == res) & (df['samples'] == samp)]
        if not data.empty:
            efficiency_matrix[i, j] = data['mrays_sec'].values[0]
        else:
            efficiency_matrix[i, j] = np.nan

im = ax4.imshow(efficiency_matrix, cmap='viridis', aspect='auto')
ax4.set_xticks(range(len(unique_samples)))
ax4.set_xticklabels(unique_samples)
ax4.set_yticks(range(len(unique_res)))
ax4.set_yticklabels(['360p', '720p', '1080p'])
ax4.set_xlabel('Samples per Pixel')
ax4.set_ylabel('Resolution')
ax4.set_title('Ray Throughput Efficiency Heatmap')
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('MRays/Second')

# Add text annotations
for i in range(len(unique_res)):
    for j in range(len(unique_samples)):
        if not np.isnan(efficiency_matrix[i, j]):
            text = ax4.text(j, i, f'{efficiency_matrix[i, j]:.0f}',
                           ha="center", va="center", color="white" if efficiency_matrix[i, j] < 150 else "black")

plt.tight_layout()
plt.savefig('figures/benchmark_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/benchmark_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Performance Breakdown
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Speedup comparison
speedup = global_time / const_time
categories = ['Global\n(Baseline)', 'Constant\nMemory']
speedups = [1.0, speedup]
colors = ['#2E86AB', '#A23B72']
bars = ax1.bar(categories, speedups, color=colors, alpha=0.8)
ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
ax1.set_ylabel('Speedup Factor')
ax1.set_title('Performance Speedup Comparison')
ax1.set_ylim(0, 1.5)
ax1.grid(axis='y', alpha=0.3)
for bar, sp in zip(bars, speedups):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{sp:.2f}×', ha='center', va='bottom')

# Memory bandwidth breakdown
bandwidth_data = {
    'Global Memory': {
        'Used': global_bandwidth,
        'Unused': theoretical_bandwidth - global_bandwidth
    },
    'Constant Memory': {
        'Used': const_bandwidth,
        'Unused': theoretical_bandwidth - const_bandwidth
    }
}

x = np.arange(len(bandwidth_data))
width = 0.35

used = [bandwidth_data[k]['Used'] for k in bandwidth_data]
unused = [bandwidth_data[k]['Unused'] for k in bandwidth_data]

p1 = ax2.bar(x, used, width, label='Used', color='#2E86AB', alpha=0.8)
p2 = ax2.bar(x, unused, width, bottom=used, label='Unused', color='#E0E0E0', alpha=0.8)

ax2.set_ylabel('Bandwidth (GB/s)')
ax2.set_title('Memory Bandwidth Utilization')
ax2.set_xticks(x)
ax2.set_xticklabels(bandwidth_data.keys())
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, theoretical_bandwidth * 1.1)

# Add percentage labels
for i, (u, total) in enumerate(zip(used, [theoretical_bandwidth] * 2)):
    percentage = (u / total) * 100
    ax2.text(i, u/2, f'{percentage:.1f}%', ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/performance_breakdown.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/performance_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

print("All graphs generated successfully!")
print("Generated files:")
print("- figures/memory_comparison.pdf/png")
print("- figures/resolution_scaling.pdf/png")
print("- figures/benchmark_analysis.pdf/png")
print("- figures/performance_breakdown.pdf/png")