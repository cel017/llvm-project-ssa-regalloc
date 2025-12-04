import csv
import matplotlib.pyplot as plt
import sys
import os

INPUT_CSV = "benchmark_results.csv"
OUTPUT_IMG = "riscv_regalloc_dashboard.png"

def plot_cdf(ax, data, label, color, linestyle='-'):
    """Helper to plot a Cumulative Distribution Function."""
    sorted_data = sorted(data)
    y_vals = [(i + 1) / len(sorted_data) for i in range(len(sorted_data))]
    ax.plot(sorted_data, y_vals, label=label, color=color, linestyle=linestyle, linewidth=2)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run the collector script first.")
        sys.exit(1)

    data = []
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    if not data:
        print("CSV is empty.")
        sys.exit(1)

    # Extract columns
    b_spills = [int(d['basic_spills']) for d in data]
    g_spills = [int(d['greedy_spills']) for d in data]
    b_regs = [int(d['basic_regs']) for d in data]
    g_regs = [int(d['greedy_regs']) for d in data]
    
    count = len(data)
    print(f"Plotting data for {count} tests...")

    # Set up 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'RISC-V (RV32E) Allocator Comparison ({count} Tests)', fontsize=16)

    # 1. Total Spills
    total_b = sum(b_spills)
    total_g = sum(g_spills)
    bars = axs[0, 0].bar(['Basic', 'Greedy'], [total_b, total_g], color=['#3498db', '#2ecc71'])
    axs[0, 0].set_title('Total Spills (Lower is Better)')
    axs[0, 0].set_ylabel('Count')
    for bar in bars:
        axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')

    # 2. Perfect Allocation Rate
    zeros_b = b_spills.count(0)
    zeros_g = g_spills.count(0)
    bars2 = axs[0, 1].bar(['Basic', 'Greedy'], [zeros_b, zeros_g], color=['#3498db', '#2ecc71'])
    axs[0, 1].set_title('Tests with ZERO Spills (Higher is Better)')
    for bar in bars2:
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')

    # 3. CDF of Spill Counts (Log Scale)
    plot_cdf(axs[1, 0], b_spills, 'Basic', '#3498db')
    plot_cdf(axs[1, 0], g_spills, 'Greedy', '#2ecc71')
    
    axs[1, 0].set_title('Spill Count CDF (Log Scale)')
    axs[1, 0].set_xlabel('Number of Spills (Left is Better)')
    axs[1, 0].set_ylabel('Fraction of Tests')
    axs[1, 0].set_xscale('symlog') 
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. CDF of Register Usage
    plot_cdf(axs[1, 1], b_regs, 'Basic', '#3498db')
    plot_cdf(axs[1, 1], g_regs, 'Greedy', '#2ecc71')
    
    axs[1, 1].set_title('Register Usage CDF')
    axs[1, 1].set_xlabel('Registers Used (Left is Better)')
    axs[1, 1].set_ylabel('Fraction of Tests')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG)
    print(f"Graph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()