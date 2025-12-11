import os
import subprocess
import csv
import time

LLC_PATH = "./build_rv1/bin/llc" 

CLANG_PATH = "clang"          

POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

# Selected benchmarks (Mix of BLAS, Kernels, and Solvers)
BENCHMARKS = [
    # Dense Linear Algebra (High Register Pressure)
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",

    # Kernels (Complex loop nests)
    "linear-algebra/kernels/2mm/2mm.c",
    "linear-algebra/kernels/3mm/3mm.c",
    "linear-algebra/kernels/atax/atax.c",
    "linear-algebra/kernels/bicg/bicg.c",
    
    # Solvers (Data dependencies)
    "linear-algebra/solvers/lu/lu.c",
    "linear-algebra/solvers/gramschmidt/gramschmidt.c",
]

def run_command(cmd):
    """Helper to run shell commands silently"""
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # If it fails, print the command so you can debug it manually
        return False
    return True

def get_exec_time(exe_path):
    """Runs the benchmark and parses the output for time"""
    try:
        times = []
        for _ in range(3):
            # Run the executable
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            # PolyBench with -DPOLYBENCH_TIME outputs ONLY the time (e.g. "0.042")
            val = result.stdout.strip()
            if val:
                times.append(float(val))
        
        if not times: return None
        return sum(times) / len(times)
    except Exception as e:
        print(f"!! Runtime Error: {e}")
        return None

def main():
    print(f"Starting Benchmark Suite using LLC: {LLC_PATH}")
    
    # Initialize CSV with headers
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark"] + ALLOCATORS)

    for bench_path in BENCHMARKS:
        # Extract name (e.g., "gemm")
        bench_name = os.path.basename(bench_path).replace(".c", "")
        
        full_src_path = os.path.join(POLYBENCH_ROOT, bench_path)
        bench_dir = os.path.dirname(full_src_path)
        
        print(f"--- Processing: {bench_name} ---")
        
        row_data = [bench_name]

        # 1. EMIT IR (System Clang)
        # Use -O3 for optimization, Stack Arrays for pressure, and Large Dataset for stress
        ir_file = f"{bench_name}.ll"
        cmd_ir = (
            f"{CLANG_PATH} -O3 -S -emit-llvm {full_src_path} -o {ir_file} "
            f"-I {POLYBENCH_ROOT}/utilities "
            f"-I {bench_dir} "
            f"-DPOLYBENCH_TIME -DPOLYBENCH_STACK_ARRAYS -DLARGE_DATASET"
        )
        
        if not run_command(cmd_ir):
            print(f"Skipping {bench_name} (IR Gen failed)")
            continue

        # Loop through allocators
        for alloc in ALLOCATORS:
            obj_file = f"{bench_name}_{alloc}.o"
            exe_file = f"./{bench_name}_{alloc}"
            
            # 2. COMPILE IR -> OBJ (Custom LLC)
            # This is where your code runs!
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=obj "
                f"{ir_file} -o {obj_file}"
            )
            
            if not run_command(cmd_llc):
                row_data.append("CompileFail")
                continue

            # 3. LINK OBJ -> EXE (System Clang)
            poly_util = os.path.join(POLYBENCH_ROOT, "utilities/polybench.c")
            cmd_link = f"{CLANG_PATH} {obj_file} {poly_util} -o {exe_file} -lm"
            
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

            # Clean up binary to save space
            if os.path.exists(exe_file): os.remove(exe_file)
            if os.path.exists(obj_file): os.remove(obj_file)

        # Write result immediately to CSV (so you don't lose data if script crashes)
        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Cleanup IR
        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()