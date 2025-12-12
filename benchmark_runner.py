import os
import subprocess
import csv

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_medium_starved.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

BENCHMARKS = [
    # The "Winners" (Control Flow / Triangular)
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",
    # Solvers
    "linear-algebra/solvers/lu/lu.c",
    "linear-algebra/solvers/cholesky/cholesky.c",
    # Dense
    "linear-algebra/blas/gemm/gemm.c",
]

# --- NUCLEAR STARVATION (5 Regs) ---
reserved_regs = []
for r in range(5, 8): reserved_regs.append(f"+reserve-x{r}")   # t0-t2
for r in range(8, 10): reserved_regs.append(f"+reserve-x{r}")  # s0-s1
for r in range(15, 18): reserved_regs.append(f"+reserve-x{r}") # a5-a7
for r in range(18, 32): reserved_regs.append(f"+reserve-x{r}") # s2-s11, t3-t6
STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

# --- CUSTOM SIZE (Medium) ---
# Standard is usually 1024. Small is 32. 
# We set 256 to get ~1 second runtime.
# We define ALL common dimension names to be safe.
SIZE_FLAGS = "-DNI=256 -DNJ=256 -DNK=256 -DNL=256 -DNM=256 -DN=256"

def run_command(cmd):
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return True

def get_exec_time(exe_path):
    try:
        times = []
        # Run 3 times. Should be fast enough now.
        for _ in range(3): 
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val: times.append(float(val))
        if not times: return None
        return sum(times) / len(times)
    except Exception:
        return None

def main():
    print(f"Starting Medium Suite (N=256) + STARVATION")
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark"] + ALLOCATORS)

    for bench_path in BENCHMARKS:
        bench_name = os.path.basename(bench_path).replace(".c", "")
        full_src_path = os.path.join(POLYBENCH_ROOT, bench_path)
        bench_dir = os.path.dirname(full_src_path)
        
        print(f"--- Processing: {bench_name} ---")
        row_data = [bench_name]

        # 1. EMIT IR 
        # INJECT SIZE FLAGS HERE instead of -D..._DATASET
        ir_file = f"{bench_name}.ll"
        cmd_ir = (
            f"{CLANG_PATH} -O1 -S -emit-llvm {full_src_path} -o {ir_file} "
            f"-I {POLYBENCH_ROOT}/utilities "
            f"-I {bench_dir} "
            f"-DPOLYBENCH_TIME -DPOLYBENCH_STACK_ARRAYS {SIZE_FLAGS}"
        )
        
        if not run_command(cmd_ir):
            print("Skipping (IR Gen failed)")
            continue

        for alloc in ALLOCATORS:
            obj_file = f"{bench_name}_{alloc}.o"
            exe_file = f"./{bench_name}_{alloc}"
            
            # 2. COMPILE (Custom LLC + Starvation)
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
            
            if os.path.exists(exe_file): os.remove(exe_file)
            if os.path.exists(obj_file): os.remove(obj_file)

        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()