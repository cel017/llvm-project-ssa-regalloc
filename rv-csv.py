import os
import re
import glob
import subprocess
import sys
import csv

# Configuration
LLC_CMD = "llc"
TEST_DIR = "497"
TEMP_ASM = "temp_output.s"
OUTPUT_CSV = "benchmark_results.csv"
TARGET_COUNT = 500  # Stop after this many successful tests

# Regex for RISC-V registers (ABI names and Architectural names)
# Matches: x0-x31, f0-f31, a0-a7, s0-s11, t0-t6, sp, ra, gp, tp, zero
REG_PATTERN = re.compile(r'\b(?:x[0-9]|[1-2][0-9]|3[0-1]|f[0-9]|[1-2][0-9]|3[0-1]|zero|ra|sp|gp|tp|t[0-6]|s[0-1]?[0-9]|a[0-7]|ft[0-7]|fs[0-1]?[0-9]|fa[0-7])\b')

def parse_requirements(file_path):
    """Check if the file is suitable for RISC-V execution."""
    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Check REQUIRES
    # We strictly look for riscv support or generic tests
    requires_line = re.search(r';\s*REQUIRES:\s*(.*)', content)
    if requires_line:
        reqs = requires_line.group(1).lower()
        # If it explicitly requires another arch, skip it
        forbidden = ['x86', 'aarch64', 'arm', 'mips', 'powerpc', 'hexagon', 'nvptx', 'amdgpu', 'systemz', 'wasm']
        for arch in forbidden:
            if arch in reqs:
                return False, None
        # Optional: Force it to have 'riscv' if your folder is mixed
        # if 'riscv' not in reqs: return False, None
    
    # Extract RUN command
    run_lines = re.findall(r';\s*RUN:\s*(.*)', content)
    llc_run = None
    for line in run_lines:
        if 'llc' in line:
            llc_run = line
            break
            
    return True, llc_run

def extract_clean_command(run_line, file_path):
    """Clean the RUN line to get a standalone llc command for RISC-V."""
    match = re.search(r'(llc.*?)(?:\s*[|>].*)?$', run_line)
    if not match:
        return None
    cmd_str = match.group(1)
    
    # --- CHANGED: Native Mode ---
    # Since the host is RISC-V, we trust the file's original command entirely.
    # We do NOT strip -mtriple or inject anything.
    
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

def main():
    files = glob.glob(os.path.join(TEST_DIR, "*.ll"))
    
    # Sort A-Z
    files.sort()
    
    # Rotate list to start at 'r'
    start_index = 0
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath).lower()
        if fname >= 'r':
            start_index = i
            break
            
    # Rotate list: [r..., z, a..., q]
    files = files[start_index:] + files[:start_index]

    total_files = len(files)
    print(f"Found {total_files} .ll files. Processing for Native RISC-V.")
    
    valid_count = 0
    
    # Open CSV for writing
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'basic_spills', 'basic_regs', 'greedy_spills', 'greedy_regs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, fpath in enumerate(files):
            if valid_count >= TARGET_COUNT:
                break

            sys.stdout.write(f"\r[Collected: {valid_count}/{TARGET_COUNT}] Scanning {i+1}/{total_files} ({os.path.basename(fpath)})...")
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
                
            # Write row immediately
            writer.writerow({
                'filename': os.path.basename(fpath),
                'basic_spills': b_spill,
                'basic_regs': b_reg,
                'greedy_spills': g_spill,
                'greedy_regs': g_reg
            })
            csvfile.flush() # Ensure data is saved if script crashes
            valid_count += 1
            
    print(f"\n\nDone! Saved {valid_count} records to {OUTPUT_CSV}")
    
    if os.path.exists(TEMP_ASM):
        os.remove(TEMP_ASM)

if __name__ == "__main__":
    main()