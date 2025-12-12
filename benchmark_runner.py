import os
import subprocess
import csv
import time

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_rv32e.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

# We focus on the benchmarks that showed sensitivity earlier
BENCHMARKS = [
    # Control Flow Heavy (Expect Greedy/SSA to win)
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",
    
    # Solvers (Complex Liveness)
    "linear-algebra/solvers/lu/lu.c",
    "linear-algebra/solvers/cholesky/cholesky.c",
    
    # Stencils (Many live neighbors)
    "medley/floyd-warshall/floyd-warshall.c",
    "medley/deriche/deriche.c",

    # Dense (Baseline)
    "linear-algebra/blas/gemm/gemm.c",
]

# --- RV32E SIMULATION FLAGS ---
# RV32E only has registers x0 through x15.
# We reserve x16 through x31 to force the compiler to work with half the register file.
reserved_regs = []
for r in range(16, 32): 
    reserved_regs.append(f"+reserve-x{r}")

# Combine into -mattr flag
STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

def run_command(cmd):
    try:
        # stdout is hidden, but we let stderr show up if something crashes
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"\n!! Command Failed: {cmd}")
        return False
    return True

def get_exec_time(exe_path):
    try:
        times = []
        # Standard Dataset is slower, so 3 runs is enough
        for _ in range(3): 
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val: times.append(float(val))
        if not times: return None
        return sum(times) / len(times)
    except Exception:
        return None

def main():
    print(f"Starting RV32E Suite (16 Regs) using LLC: {LLC_PATH}")
    print(f"--- MODE: STANDARD DATASET + x16-x31 RESERVED ---")
    
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
        # Standard Dataset + O1 (Keep loops intact)
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
            # Injecting RV32E Flags
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=obj "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {obj_file}"
            )
            
            if not run_command(cmd_llc):
                row_data.append("Fail")
                continue

            # 3. LINK
            poly_util = os.path.join(POLYBENCH_ROOT, "utilities/polybench.c")
            cmd_link = f"{CLANG_PATH} {obj_file} {poly_util} -DPOLYBENCH_TIME -o {exe_file} -lm"
            
            if not run_command(cmd_link):
                row_data.append("Fail")
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
            
            # Cleanup
            if os.path.exists(exe_file): os.remove(exe_file)
            if os.path.exists(obj_file): os.remove(obj_file)

        # Save immediately
        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()