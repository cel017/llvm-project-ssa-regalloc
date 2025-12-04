import os
import re
import glob
import subprocess
import sys
import csv
import shutil

# Configuration
# Path to the specific custom build of llc
LLC_CMD = os.path.abspath("build_rv1/bin/llc")
TEST_DIR = "497"
TEMP_ASM = "temp_output.s"
OUTPUT_CSV = "benchmark_results.csv"
TARGET_COUNT = 500  # Stop after this many successful tests

# Regex for RISC-V registers (ABI names and Architectural names)
REG_PATTERN = re.compile(r'\b(?:[xf](?:[1-2][0-9]|3[0-1]|[0-9])|zero|ra|sp|gp|tp|t[0-6]|s[0-1]?[0-9]|a[0-7]|ft[0-7]|fs[0-1]?[0-9]|fa[0-7])\b')

def parse_requirements(file_path):
    """Check if the file is suitable for RISC-V execution."""
    filename = os.path.basename(file_path)
    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # 1. Check REQUIRES line
    requires_line = re.search(r';\s*REQUIRES:\s*(.*)', content)
    if requires_line:
        reqs = requires_line.group(1).lower()
        forbidden = ['x86', 'aarch64', 'arm', 'mips', 'powerpc', 'hexagon', 'nvptx', 'amdgpu', 'systemz', 'wasm']
        for arch in forbidden:
            if arch in reqs:
                return False, None
    
    # 2. Extract RUN command and check -mtriple
    run_lines = re.findall(r';\s*RUN:\s*(.*)', content)
    llc_run = None
    for line in run_lines:
        if 'llc' in line:
            # Check for explicitly forbidden architectures in -mtriple
            triple_match = re.search(r'-mtriple=([a-zA-Z0-9_]+)', line)
            if triple_match:
                arch = triple_match.group(1).lower()
                # If explicit triple exists, it MUST contain 'riscv'
                if 'riscv' not in arch:
                    continue 
            
            llc_run = line
            break
            
    if not llc_run:
        return False, None
            
    return True, llc_run

def extract_clean_command(run_line, file_path):
    """Clean the RUN line to get a standalone llc command for RISC-V."""
    filename = os.path.basename(file_path)
    # 1. Stop at pipes (|) or redirects (>)
    match = re.search(r'(llc.*?)(?:\s*[|>].*)?$', run_line)
    if not match:
        return None
    cmd_str = match.group(1)

    # 2. Replace generic 'llc' with the absolute path found on system
    if cmd_str.startswith("llc"):
        cmd_str = LLC_CMD + cmd_str[3:]
    
    # 3. CRITICAL FIX: Remove trailing backslashes that cause shell to hang
    cmd_str = cmd_str.replace('\\', ' ')
    
    # 4. Handle input file replacement
    # Some tests use "< %s". llc accepts "llc file.ll". 
    # We replace "< %s" with just the filename, effectively converting redirect to arg.
    if '< %s' in cmd_str:
        cmd_str = cmd_str.replace('< %s', f'"{file_path}"')
    else:
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
    filename = os.path.basename(file_path)
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
    if not os.path.exists(LLC_CMD):
        print(f"Error: Custom 'llc' not found at: {LLC_CMD}")
        print("Please check the path to your build_rv1 folder.")
        sys.exit(1)
        
    files = glob.glob(os.path.join(TEST_DIR, "*.ll"))
    
    # Sort files in Reverse Alphabetical order (Z -> A)
    files.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)
    
    # Find the first file starting with 'r'
    start_index = 0
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath).lower()
        if fname.startswith('r'):
            start_index = i
            break
            
    # Rotate list to start at R
    # Sequence will be: [R... -> A... -> Z... -> S...]
    files = files[start_index:] + files[:start_index]

    total_files = len(files)
    start_file_name = os.path.basename(files[0]) if files else "None"
    print(f"Found {total_files} .ll files. Processing for Native RISC-V.")
    print(f"Using LLC: {LLC_CMD}")
    print(f"Starting at file: {start_file_name}")
    print("Processing order: Reverse Alphabetical (Z->A), starting at 'R'")
    
    valid_count = 0
    skipped_count = 0
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'basic_spills', 'basic_regs', 'greedy_spills', 'greedy_regs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, fpath in enumerate(files):
            if valid_count >= TARGET_COUNT:
                break

            sys.stdout.write(f"\r[Collected: {valid_count}/{TARGET_COUNT} | Skipped: {skipped_count}] Scanning {i+1}/{total_files} ({os.path.basename(fpath)})...")
            sys.stdout.flush()
            
            valid, run_line = parse_requirements(fpath)
            if not valid or not run_line:
                skipped_count += 1
                continue
                
            clean_cmd = extract_clean_command(run_line, fpath)
            if not clean_cmd:
                skipped_count += 1
                continue
                
            # Run Basic
            b_spill, b_reg = run_benchmark(fpath, clean_cmd, "basic")
            if b_reg is None: 
                skipped_count += 1
                continue 
            
            # Run Greedy
            g_spill, g_reg = run_benchmark(fpath, clean_cmd, "greedy")
            if g_reg is None: 
                skipped_count += 1
                continue 
            
            if b_reg == 0 and g_reg == 0:
                skipped_count += 1
                continue
                
            writer.writerow({
                'filename': os.path.basename(fpath),
                'basic_spills': b_spill,
                'basic_regs': b_reg,
                'greedy_spills': g_spill,
                'greedy_regs': g_reg
            })
            csvfile.flush() 
            valid_count += 1
            
    print(f"\n\nDone! Saved {valid_count} records to {OUTPUT_CSV}")
    
    if os.path.exists(TEMP_ASM):
        os.remove(TEMP_ASM)

if __name__ == "__main__":
    main()