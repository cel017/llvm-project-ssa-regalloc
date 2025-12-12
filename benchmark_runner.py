import os
import subprocess
import csv
import time

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_runtime_heap_starved.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

BENCHMARKS = [
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",
    "linear-algebra/solvers/lu/lu.c",
    "medley/floyd-warshall/floyd-warshall.c",
    "medley/deriche/deriche.c",
    "linear-algebra/blas/gemm/gemm.c",
]

# --- 1. NUCLEAR STARVATION (5 Regs) ---
reserved_regs = []
for r in range(5, 8): reserved_regs.append(f"+reserve-x{r}")
for r in range(8, 10): reserved_regs.append(f"+reserve-x{r}")
for r in range(15, 18): reserved_regs.append(f"+reserve-x{r}")
for r in range(18, 32): reserved_regs.append(f"+reserve-x{r}")
STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

# --- 2. DATASET (Standard + Heap) ---
# REMOVED: -DPOLYBENCH_STACK_ARRAYS
# Now using default malloc() behavior.
SIZE_FLAGS = "-DSTANDARD_DATASET"

def run_command(cmd):
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return True

def get_exec_time(exe_path):
    try:
        times = []
        for _ in range(3): 
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val: times.append(float(val))
        if not times: return None
        return sum(times) / len(times)
    except Exception:
        return None

def main():
    print(f"Starting RUNTIME Analysis (Heap Arrays + 5 Regs)")
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Alloc", "Time(s)"])

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
            obj_file = f"{bench_name}_{alloc}.o"
            exe_file = f"./{bench_name}_{alloc}"
            
            # 2. COMPILE (Custom LLC)
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=obj "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {obj_file}"
            )
            
            if not run_command(cmd_llc):
                print(f"  {alloc}: Failed compilation")
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([bench_name, alloc, "CompileFail"])
                continue

            # 3. LINK
            poly_util = os.path.join(POLYBENCH_ROOT, "utilities/polybench.c")
            cmd_link = f"{CLANG_PATH} {obj_file} {poly_util} -DPOLYBENCH_TIME -o {exe_file} -lm"
            
            if not run_command(cmd_link):
                print(f"  {alloc}: Failed linking")
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([bench_name, alloc, "LinkFail"])
                continue

            # 4. RUN
            print(f"   Testing {alloc}...", end="", flush=True)
            avg_time = get_exec_time(exe_file)
            
            if avg_time is not None:
                print(f" {avg_time:.4f}s")
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([bench_name, alloc, f"{avg_time:.6f}"])
            else:
                print(" Error")
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([bench_name, alloc, "RunFail"])
            
            if os.path.exists(exe_file): os.remove(exe_file)
            if os.path.exists(obj_file): os.remove(obj_file)

        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()