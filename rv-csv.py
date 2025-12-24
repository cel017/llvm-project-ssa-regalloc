import os
import re
import glob
import subprocess
import sys
import csv

# --- CONFIGURATION ---
LLC_CMD = os.path.abspath("build_rv1/bin/llc")
TEST_DIR = "497"
TEMP_ASM = "temp_output.s"
OUTPUT_CSV = "benchmark_results.csv"
TARGET_COUNT = 500  

# --- MATTR=+E (16 Registers) ---
LLC_MATTR = "-mattr=+e -debug-pass=Structure" 

def parse_requirements(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()
    
    requires_line = re.search(r';\s*REQUIRES:\s*(.*)', content)
    if requires_line:
        reqs = requires_line.group(1).lower()
        forbidden = [
            'x86', 'aarch64', 'arm', 'mips', 'powerpc', 'hexagon',
            'nvptx', 'amdgpu', 'systemz', 'wasm'
        ]
        for arch in forbidden:
            if arch in reqs:
                return False, None
    
    run_lines = re.findall(r';\s*RUN:\s*(.*)', content)
    llc_run = None
    for line in run_lines:
        if 'llc' in line:
            triple_match = re.search(r'-mtriple=([a-zA-Z0-9_]+)', line)
            if triple_match:
                arch = triple_match.group(1).lower()
                if 'riscv' not in arch:
                    continue
            llc_run = line
            break
            
    if not llc_run:
        return False, None
    return True, llc_run

def extract_clean_command(run_line, file_path):
    match = re.search(r'(llc.*?)(?:\s*[|>].*)?$', run_line)
    if not match:
        return None
    cmd_str = match.group(1)
    if cmd_str.startswith("llc"):
        cmd_str = LLC_CMD + cmd_str[3:]
    cmd_str = cmd_str.replace('\\', ' ')
    if '< %s' in cmd_str:
        cmd_str = cmd_str.replace('< %s', f'"{file_path}"')
    else:
        cmd_str = cmd_str.replace('%s', f'"{file_path}"')
    return cmd_str.strip()

def count_ops(asm_file):
    """
    Parses the generated assembly to count Stores and Loads.
    """
    stores = 0
    loads = 0
    
    if not os.path.exists(asm_file):
        return 0, 0

    try:
        with open(asm_file, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # Skip labels and comments
                if line.endswith(':') or line.startswith('.') or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts: continue
                opcode = parts[0]

                # 1. STORES
                if re.match(r'^(sd|fsd|sw|fsw|sh|sb)$', opcode):
                    stores += 1
                
                # 2. LOADS
                elif re.match(r'^(ld|fld|lw|flw|lh|lb)$', opcode):
                    loads += 1
                    
    except Exception:
        return 0, 0
        
    return stores, loads

def run_benchmark(file_path, run_cmd, alloc_mode):
    # Prepare command with allocator
    if "-regalloc=" in run_cmd:
        cmd = re.sub(r'-regalloc=\S+', f'-regalloc={alloc_mode}', run_cmd)
    else:
        cmd = f"{run_cmd} -regalloc={alloc_mode}"
    
    # Inject the MATTR flag
    cmd = f"{cmd} {LLC_MATTR} -o {TEMP_ASM}"
    
    try:
        # Capture stderr so we can print it on failure
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore',
            timeout=5,
            check=True
        )
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT EXPIRED"
    except subprocess.CalledProcessError as e:
        return None, None, e.stderr # Return the actual error message

    # Count Ops from the generated asm
    stores, loads = count_ops(TEMP_ASM)
    return stores, loads, None

def main():
    files = glob.glob(os.path.join(TEST_DIR, "*.ll"))
    files.sort(key=lambda x: os.path.basename(x).lower())
    
    # Rotate so that we start at the first file whose name is >= 'r'
    start_index = 0
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath).lower()
        if fname >= 'r':
            start_index = i
            break
    files = files[start_index:] + files[:start_index]

    print(f"Processing {len(files)} files with MATTR=+E (16 Regs)...")
    
    valid_count = 0
    skipped_count = 0
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = [
            'filename',
            'basic_stores', 'basic_loads',
            'greedy_stores', 'greedy_loads',
            'ssa_stores', 'ssa_loads'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, fpath in enumerate(files):
            if valid_count >= TARGET_COUNT:
                break

            fname = os.path.basename(fpath)
            
            valid, run_line = parse_requirements(fpath)
            if not valid or not run_line:
                skipped_count += 1
                print(f"Skipped: {fname} (Invalid Requirements or Missing RUN line)")
                continue
                
            clean_cmd = extract_clean_command(run_line, fpath)
            if not clean_cmd:
                skipped_count += 1
                print(f"Skipped: {fname} (Could not extract valid LLC command)")
                continue
                
            # 1. Basic
            b_stores, b_loads, b_err = run_benchmark(fpath, clean_cmd, "basic")
            if b_stores is None:
                skipped_count += 1
                print(f"Skipped: {fname} (Basic Allocator Failed)")
                continue 
            
            # 2. Greedy
            g_stores, g_loads, g_err = run_benchmark(fpath, clean_cmd, "greedy")
            if g_stores is None:
                skipped_count += 1
                print(f"Skipped: {fname} (Greedy Allocator Failed)")
                continue 
            
            # 3. SSA
            ssa_stores, ssa_loads, ssa_err = run_benchmark(fpath, clean_cmd, "ssa")
            if ssa_stores is None:
                skipped_count += 1
                print(f"\n!!! Skipped: {fname} (SSA Allocator Failed) !!!")
                print("="*40)
                print(f"ERROR LOG FOR {fname} (SSA):")
                print(ssa_err)
                print("="*40 + "\n")
                continue

            # Filter trivial files
            if b_stores == 0 and g_stores == 0 and ssa_stores == 0 and \
               b_loads == 0 and g_loads == 0 and ssa_loads == 0:
                skipped_count += 1
                print(f"Skipped: {fname} (Trivial: 0 Stores/Loads across all allocators)")
                continue
                
            # Print status to console
            print(f"[{valid_count+1}] {fname} | "
                  f"Basic(S:{b_stores} L:{b_loads}) "
                  f"Greedy(S:{g_stores} L:{g_loads}) "
                  f"SSA(S:{ssa_stores} L:{ssa_loads})")

            writer.writerow({
                'filename': fname,
                'basic_stores': b_stores,
                'basic_loads': b_loads,
                'greedy_stores': g_stores,
                'greedy_loads': g_loads,
                'ssa_stores': ssa_stores,
                'ssa_loads': ssa_loads
            })
            csvfile.flush()
            valid_count += 1
            
    print(f"\n\nDone! Saved to {OUTPUT_CSV}")
    if os.path.exists(TEMP_ASM):
        os.remove(TEMP_ASM)

if __name__ == "__main__":
    main()