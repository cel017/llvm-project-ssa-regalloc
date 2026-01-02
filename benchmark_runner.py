import os
import subprocess
import csv
import re
import time

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_spec_metrics.csv"

ALLOCATORS = ["ssa", "greedy"]

# The "SPEC Equivalent" Suite
BENCHMARKS = [
    "linear-algebra/kernels/2mm/2mm.c",
    "stencils/fdtd-2d/fdtd-2d.c",
    "medley/floyd-warshall/floyd-warshall.c",
    "stencils/jacobi-2d/jacobi-2d.c",
    "linear-algebra/solvers/ludcmp/ludcmp.c",
    "medley/deriche/deriche.c",
    "linear-algebra/blas/syrk/syrk.c"
    ]

# --- STARVATION (5 Registers: a0-a4) ---

reserved_regs = []
for r in range(5, 8): reserved_regs.append(f"+reserve-x{r}")   # t0-t2
for r in range(8, 10): reserved_regs.append(f"+reserve-x{r}")  # s0-s1
for r in range(15, 18): reserved_regs.append(f"+reserve-x{r}") # a5-a7
for r in range(18, 32): reserved_regs.append(f"+reserve-x{r}") # s2-s11, t3-t6
STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

# --- DATASET ---
# Standard Dataset for correctness, Heap arrays for stability
SIZE_FLAGS = "-DSTANDARD_DATASET"

def get_static_metrics(asm_file):
    """
    Parses RISC-V assembly to count specific instruction categories.
    """
    metrics = {'store': 0, 'load': 0, 'move': 0, 'xor': 0}
    
    try:
        with open(asm_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip labels and comments
                if line.endswith(':') or line.startswith('.') or line.startswith('#'):
                    continue
                
                # Tokenize the opcode (first word)
                parts = line.split()
                if not parts: continue
                opcode = parts[0]

                # 1. STORES (Stack spills + Data saves)
                if re.match(r'^(sd|fsd|sw|fsw|sh|sb)$', opcode):
                    metrics['store'] += 1
                
                # 2. LOADS (Stack reloads + Data fetches)
                elif re.match(r'^(ld|fld|lw|flw|lh|lb)$', opcode):
                    metrics['load'] += 1
                
                # 3. MOVES (Register Shuffling)
                # LLVM RISC-V emits 'mv' or 'fmv'. 
                # Sometimes 'addi rd, rs, 0' is used, but 'mv' is standard alias.
                elif re.match(r'^(mv|fmv\.d|fmv\.w|fmv\.x\.w|fmv\.w\.x)$', opcode):
                    metrics['move'] += 1
                    
                # 4. XOR (Logic/Checksums)
                elif re.match(r'^(xor|xori)$', opcode):
                    metrics['xor'] += 1
                    
    except Exception as e:
        print(f"Error parsing ASM: {e}")
    
    return metrics

def run_command(cmd):
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return True

def get_exec_time(exe_path):
    try:
        times = []
        # Run 3 times
        for _ in range(3): 
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val: times.append(float(val))
        if not times: return None
        return sum(times) / len(times)
    except Exception:
        return None

def main():
    print(f"Starting SPEC-Like Metric Collection (5 Regs)")
    
    # Write CSV Header
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Allocator", "Store", "Load", "Move", "Xor", "Time(s)"])

    for bench_path in BENCHMARKS:
        bench_name = os.path.basename(bench_path).replace(".c", "")
        full_src_path = os.path.join(POLYBENCH_ROOT, bench_path)
        bench_dir = os.path.dirname(full_src_path)
        
        print(f"--- Processing: {bench_name} ---")

        # 1. EMIT IR 
        ir_file = f"{bench_name}.ll"
        cmd_ir = (
            f"{CLANG_PATH} -O1 -S -emit-llvm {full_src_path} -o {ir_file} "
            f"-I {POLYBENCH_ROOT}/utilities "
            f"-I {bench_dir} "
            f"-DPOLYBENCH_TIME {SIZE_FLAGS}"
        )
        
        if not run_command(cmd_ir):
            print("Skipping (IR Gen failed)")
            continue

        for alloc in ALLOCATORS:
            asm_file = f"{bench_name}_{alloc}.s"
            obj_file = f"{bench_name}_{alloc}.o"
            exe_file = f"./{bench_name}_{alloc}"
            
            # 2. COMPILE TO ASM (To count instructions)
            cmd_asm = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=asm "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {asm_file}"
            )
            if not run_command(cmd_asm):
                continue

            # 3. COMPILE TO OBJ (For Linking)
            cmd_obj = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=obj "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {obj_file}"
            )
            if not run_command(cmd_obj):
                continue

            # 4. GATHER STATIC METRICS
            stats = get_static_metrics(asm_file)

            # 5. LINK & RUN (For Time)
            poly_util = os.path.join(POLYBENCH_ROOT, "utilities/polybench.c")
            cmd_link = f"{CLANG_PATH} {obj_file} {poly_util} -DPOLYBENCH_TIME -o {exe_file} -lm"
            
            runtime = "Fail"
            if run_command(cmd_link):
                print(f"   Testing {alloc}...", end="", flush=True)
                t = get_exec_time(exe_file)
                if t is not None:
                    print(f" {t:.4f}s")
                    runtime = f"{t:.6f}"
                else:
                    print(" Error")

            # 6. WRITE ROW
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    bench_name, 
                    alloc, 
                    stats['store'], 
                    stats['load'], 
                    stats['move'], 
                    stats['xor'], 
                    runtime
                ])

            # Cleanup
            if os.path.exists(asm_file): os.remove(asm_file)
            if os.path.exists(obj_file): os.remove(obj_file)
            if os.path.exists(exe_file): os.remove(exe_file)

        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Data collected in {RESULTS_FILE}")

if __name__ == "__main__":
    main()