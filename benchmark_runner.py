import os
import subprocess
import csv
import time

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_starved.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

BENCHMARKS = [
    # BLAS
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/syrk/syrk.c",

    # Complex Kernels
    "linear-algebra/kernels/atax/atax.c",
    "linear-algebra/kernels/bicg/bicg.c",
    "linear-algebra/kernels/2mm/2mm.c",
    "linear-algebra/kernels/3mm/3mm.c",
]

# ---  STARVATION FLAGS (RISC-V) ---
# Valid Allocatable Regs in RISC-V are usually x5-x31.
reserved_regs = []

# 1. Reserve Temps t0-t2 (x5-x7)
for r in range(5, 8): reserved_regs.append(f"+reserve-x{r}")

# 2. Reserve Saved s0-s1 (x8-x9)
for r in range(8, 10): reserved_regs.append(f"+reserve-x{r}")

# 3. Reserve High Args a5-a7 (x15-x17)
# WE LEAVE a0-a4 (x10-x14) AVAILABLE. (5 Registers total)
for r in range(15, 18): reserved_regs.append(f"+reserve-x{r}")

# 4. Reserve High Saved/Temps (x18-x31)
for r in range(18, 32): reserved_regs.append(f"+reserve-x{r}")

STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

def run_command(cmd):
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"\n!! Command Failed: {cmd}")
        return False
    return True

def get_exec_time(exe_path):
    try:
        times = []
        for _ in range(5): # Run 5 times for Standard Dataset (it's faster)
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val: times.append(float(val))
        if not times: return None
        return sum(times) / len(times)
    except Exception:
        return None

def main():
    print(f"Starting Benchmark Suite using LLC: {LLC_PATH}")
    print(f"--- MODE: STANDARD DATASET + STARVED REGS (5 Available) ---")
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark"] + ALLOCATORS)

    for bench_path in BENCHMARKS:
        bench_name = os.path.basename(bench_path).replace(".c", "")
        full_src_path = os.path.join(POLYBENCH_ROOT, bench_path)
        bench_dir = os.path.dirname(full_src_path)
        
        print(f"--- Processing: {bench_name} ---")
        row_data = [bench_name]

        # 1. EMIT IR (System Clang)
        # Changed to STANDARD_DATASET (Safe for stack)
        ir_file = f"{bench_name}.ll"
        cmd_ir = (
            f"{CLANG_PATH} -O1 -S -emit-llvm {full_src_path} -o {ir_file} "
            f"-I {POLYBENCH_ROOT}/utilities "
            f"-I {bench_dir} "
            f"-DPOLYBENCH_TIME -DPOLYBENCH_STACK_ARRAYS -DSTANDARD_DATASET"
        )
        
        if not run_command(cmd_ir):
            print("Skipping (IR Gen failed)")
            continue

        for alloc in ALLOCATORS:
            obj_file = f"{bench_name}_{alloc}.o"
            exe_file = f"./{bench_name}_{alloc}"
            
            # 2. COMPILE (Custom LLC)
            # Injecting STARVE_FLAGS
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=obj "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {obj_file}"
            )
            
            if not run_command(cmd_llc):
                # Basic might fail with "ran out of registers" if 5 is too tight
                row_data.append("RegsExhausted")
                continue

            # 3. LINK
            poly_util = os.path.join(POLYBENCH_ROOT, "utilities/polybench.c")
            cmd_link = f"{CLANG_PATH} {obj_file} {poly_util} -DPOLYBENCH_TIME -o {exe_file} -lm"
            
            if not run_command(cmd_link):
                row_data.append("LinkFail")
                continue

            # 4. RUN
            print(f"   Testing {alloc}...", end="", flush=True)
            avg_time = get_exec_time(exe_file)
            
            if avg_time is not None:
                print(f" {avg_time:.4f}s")
                row_data.append(f"{avg_time:.6f}")
            else:
                print(" Error")
                row_data.append("RunFail")
            
            if os.path.exists(exe_file): os.remove(exe_file)
            if os.path.exists(obj_file): os.remove(obj_file)

        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()