import os
import re
import glob
import subprocess
import matplotlib.pyplot as plt
import sys

# Configuration
LLC_CMD = "llc"
TEST_DIR = "497"
TEMP_ASM = "temp_output.s"
TARGET_COUNT = 500  # Stop after this many successful tests

# Regex for x86 registers
REG_PATTERN = re.compile(r'%(?:[er]?[a-d]x|[er]?[bs]p|[er]?[sd]i|r\d+[dwb]?|[xy]mm\d+|st(?:\(\d\))?)')

def parse_requirements(file_path):
    """Check if the file is suitable for x86_64-linux execution."""
    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Check REQUIRES
    requires_line = re.search(r';\s*REQUIRES:\s*(.*)', content)
    if requires_line:
        reqs = requires_line.group(1).lower()
        # Forbidden architectures for this machine
        forbidden = ['aarch64', 'arm', 'mips', 'powerpc', 'hexagon', 'nvptx', 'amdgpu', 'systemz', 'wasm']
        for arch in forbidden:
            if arch in reqs:
                return False, None
    
    # Extract RUN command
    run_lines = re.findall(r';\s*RUN:\s*(.*)', content)
    llc_run = None
    for line in run_lines:
        if 'llc' in line:
            llc_run = line
            break
            
    return True, llc_run

def extract_clean_command(run_line, file_path):
    """Clean the RUN line to get a standalone llc command."""
    match = re.search(r'(llc.*?)(?:\s*[|>].*)?$', run_line)
    if not match:
        return None
    cmd_str = match.group(1)
    cmd_str = cmd_str.replace('%s', f'"{file_path}"')
    return cmd_str.strip()

def count_unique_registers(asm_file):
    """Parse .s file and count unique physical registers used."""
    if not os.path.exists(asm_file):
        return 0
    unique_regs = set()
    with open(asm_file, 'r', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('.') or line.strip().startswith('#'):
                continue
            matches = REG_PATTERN.findall(line)
            for m in matches:
                unique_regs.add(m)
    return len(unique_regs)

def run_benchmark(file_path, run_cmd, alloc_mode):
    """Run llc with specific allocator and gather stats."""
    if "-regalloc=" in run_cmd:
        cmd = re.sub(r'-regalloc=\S+', f'-regalloc={alloc_mode}', run_cmd)
    else:
        cmd = f"{run_cmd} -regalloc={alloc_mode}"
        
    cmd = f"{cmd} -stats -o {TEMP_ASM}"
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.DEVNULL,
            timeout=5 
        )
    except subprocess.TimeoutExpired:
        return None, 0

    if result.returncode != 0:
        return None, 0

    stderr_output = result.stderr.decode('utf-8', errors='ignore')
    spill_match = re.search(r'(\d+)\s+.*- Number of spills inserted', stderr_output)
    spills = int(spill_match.group(1)) if spill_match else 0
    regs = count_unique_registers(TEMP_ASM)
    
    return spills, regs

def plot_cdf(ax, data, label, color, linestyle='-'):
    """Helper to plot a Cumulative Distribution Function."""
    sorted_data = sorted(data)
    y_vals = [(i + 1) / len(sorted_data) for i in range(len(sorted_data))]
    ax.plot(sorted_data, y_vals, label=label, color=color, linestyle=linestyle, linewidth=2)

def main():
    files = glob.glob(os.path.join(TEST_DIR, "*.ll"))
    
    # --- CHANGED: SORT IN REVERSE ORDER ---
    files.sort(reverse=True)
    # --------------------------------------

    total_files = len(files)
    print(f"Found {total_files} .ll files in {TEST_DIR}. Processing in REVERSE (Z->A). Targeting {TARGET_COUNT} valid runs.")
    
    data = [] 
    
    for i, fpath in enumerate(files):
        # Stop if we hit the target
        if len(data) >= TARGET_COUNT:
            break

        # Dynamic progress update
        sys.stdout.write(f"\r[Collected: {len(data)}/{TARGET_COUNT}] Scanning file {i+1}/{total_files} ({os.path.basename(fpath)})...")
        sys.stdout.flush()
        
        valid, run_line = parse_requirements(fpath)
        if not valid or not run_line:
            continue
            
        clean_cmd = extract_clean_command(run_line, fpath)
        if not clean_cmd:
            continue
            
        # Run Basic
        b_spill, b_reg = run_benchmark(fpath, clean_cmd, "basic")
        if b_reg is None: continue 
        
        # Run Greedy
        g_spill, g_reg = run_benchmark(fpath, clean_cmd, "greedy")
        if g_reg is None: continue 
        
        # Filter trivial files
        if b_reg == 0 and g_reg == 0:
            continue
            
        data.append({
            'file': os.path.basename(fpath),
            'basic_spill': b_spill,
            'basic_reg': b_reg,
            'greedy_spill': g_spill,
            'greedy_reg': g_reg
        })
        
    print(f"\n\nCompleted! Collected valid data for {len(data)} files.")
    
    if not data:
        print("No valid data found to plot.")
        return

    # --- PLOTTING ---
    b_spills = [d['basic_spill'] for d in data]
    g_spills = [d['greedy_spill'] for d in data]
    b_regs = [d['basic_reg'] for d in data]
    g_regs = [d['greedy_reg'] for d in data]
    
    # Set up a 2x2 grid for clearer metrics
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Register Allocator Comparison ({len(data)} Tests)', fontsize=16)

    # 1. Total Spills (The "Global Cost")
    total_b = sum(b_spills)
    total_g = sum(g_spills)
    bars = axs[0, 0].bar(['Basic', 'Greedy'], [total_b, total_g], color=['#3498db', '#2ecc71'])
    axs[0, 0].set_title('Total Spills (Lower is Better)')
    axs[0, 0].set_ylabel('Count')
    for bar in bars:
        axs[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')

    # 2. Perfect Allocation Rate (The "Success" metric)
    zeros_b = b_spills.count(0)
    zeros_g = g_spills.count(0)
    bars2 = axs[0, 1].bar(['Basic', 'Greedy'], [zeros_b, zeros_g], color=['#3498db', '#2ecc71'])
    axs[0, 1].set_title('Tests with ZERO Spills (Higher is Better)')
    for bar in bars2:
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{int(bar.get_height()):,}', ha='center', va='bottom', fontweight='bold')

    # 3. Cumulative Distribution Function (CDF) of Spill Counts
    # SCALABLE: Works for any number of allocators.
    # Shows: "What % of tests have <= X spills?"
    # Note: Using symlog to handle the massive number of 0s gracefully while showing large outliers.
    plot_cdf(axs[1, 0], b_spills, 'Basic', '#3498db')
    plot_cdf(axs[1, 0], g_spills, 'Greedy', '#2ecc71')
    
    axs[1, 0].set_title('Spill Count CDF (Log Scale)')
    axs[1, 0].set_xlabel('Number of Spills (Left is Better)')
    axs[1, 0].set_ylabel('Fraction of Tests')
    axs[1, 0].set_xscale('symlog') # symlog handles 0s better than log
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. Cumulative Distribution Function (CDF) of Register Usage
    # SCALABLE: Works for any number of allocators.
    # Shows: "What % of tests fit in <= X registers?"
    plot_cdf(axs[1, 1], b_regs, 'Basic', '#3498db')
    plot_cdf(axs[1, 1], g_regs, 'Greedy', '#2ecc71')
    
    axs[1, 1].set_title('Register Usage CDF')
    axs[1, 1].set_xlabel('Registers Used (Left is Better)')
    axs[1, 1].set_ylabel('Fraction of Tests')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle
    plt.savefig('regalloc_comparison.png')
    print("Graph saved to regalloc_comparison.png")
    
    if os.path.exists(TEMP_ASM):
        os.remove(TEMP_ASM)

if __name__ == "__main__":
    main()