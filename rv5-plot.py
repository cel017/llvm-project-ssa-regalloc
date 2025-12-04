import csv
import matplotlib.pyplot as plt
import sys
import os

INPUT_CSV = "benchmark_results.csv"
OUTPUT_IMG = "riscv_regalloc_dashboard.png"

def plot_cdf(ax, data, label, color, linestyle='-'):
    """Helper to plot a Cumulative Distribution Function."""
    if not data:
        return
    sorted_data = sorted(data)
    y_vals = [(i + 1) / len(sorted_data) for i in range(len(sorted_data))]
    ax.plot(sorted_data, y_vals, label=label, color=color, linestyle=linestyle, linewidth=2)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Run the collector script first.")
        sys.exit(1)

    raw_data = []
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_data.append(row)
            
    if not raw_data:
        print("CSV is empty.")
        sys.exit(1)

    # --- FILTERING STEP ---
    data = []
    skipped_count = 0
    
    for row in raw_data:
        try:
            ssa_val = int(row.get('ssa_regs', 0))
            # Only keep rows where SSA reported valid register usage (> 0)
            if ssa_val > 0:
                data.append(row)
            else:
                skipped_count += 1
        except ValueError:
            skipped_count += 1
            
    if not data:
        print(f"No valid data found after filtering! (Skipped {skipped_count} rows where ssa_regs=0)")
        sys.exit(1)

    # Extract columns from filtered data
    b_spills = [int(d['basic_spills']) for d in data]
    g_spills = [int(d['greedy_spills']) for d in data]
    s_spills = [int(d.get('ssa_spills', 0)) for d in data]

    b_regs = [int(d['basic_regs']) for d in data]
    g_regs = [int(d['greedy_regs']) for d in data]
    s_regs = [int(d.get('ssa_regs', 0)) for d in data]
    
    count = len(data)
    print(f"Plotting data for {count} tests (Filtered out {skipped_count} zeros)...")

    # Set up 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'RISC-V Allocator Comparison: Basic vs Greedy vs SSA ({count} Tests)', fontsize=16)

    # Colors
    c_basic = '#3498db'  # Blue
    c_greedy = '#2ecc71' # Green
    c_ssa = '#e74c3c'    # Red

    # 1. Total Spills
    total_b = sum(b_spills)
    total_g = sum(g_spills)
    total_s = sum(s_spills)
    
    labels = ['Basic', 'Greedy', 'SSA']
    values = [total_b, total_g, total_s]
    colors = [c_basic, c_greedy, c_ssa]

    bars = axs[0, 0].bar(labels, values, color=colors)
    axs[0, 0].set_title('Total Spills (Lower is Better)')
    axs[0, 0].set_ylabel('Count')
    for bar in bars:
        axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')

    # 2. Perfect Allocation Rate (Zero Spills)
    zeros_b = b_spills.count(0)
    zeros_g = g_spills.count(0)
    zeros_s = s_spills.count(0)
    
    z_values = [zeros_b, zeros_g, zeros_s]

    bars2 = axs[0, 1].bar(labels, z_values, color=colors)
    axs[0, 1].set_title('Tests with ZERO Spills (Higher is Better)')
    for bar in bars2:
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')

    # 3. CDF of Spill Counts (Log Scale)
    plot_cdf(axs[1, 0], b_spills, 'Basic', c_basic)
    plot_cdf(axs[1, 0], g_spills, 'Greedy', c_greedy)
    plot_cdf(axs[1, 0], s_spills, 'SSA (Theoretical)', c_ssa, linestyle='--')
    
    axs[1, 0].set_title('Spill Count CDF (Log Scale)')
    axs[1, 0].set_xlabel('Number of Spills (Left is Better)')
    axs[1, 0].set_ylabel('Fraction of Tests')
    axs[1, 0].set_xscale('symlog') 
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. CDF of Register Usage / Pressure
    # Note: For SSA, 'regs' represents Max Pressure, for others it is actual PhysRegs used.
    plot_cdf(axs[1, 1], b_regs, 'Basic (Used)', c_basic)
    plot_cdf(axs[1, 1], g_regs, 'Greedy (Used)', c_greedy)
    plot_cdf(axs[1, 1], s_regs, 'SSA (Max Pressure)', c_ssa, linestyle='--')
    
    axs[1, 1].set_title('Register Usage / Pressure CDF')
    axs[1, 1].set_xlabel('Registers (Left is Better)')
    axs[1, 1].set_ylabel('Fraction of Tests')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG)
    print(f"Graph saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()