import os
import subprocess
import csv
import time

LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"          # System clang
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

# Selected benchmarks
BENCHMARKS = [
    # Complex Control Flow (Kernels)
    "linear-algebra/kernels/atax/atax.c",
    "linear-algebra/kernels/bicg/bicg.c",
    "linear-algebra/kernels/2mm/2mm.c",
    "linear-algebra/kernels/3mm/3mm.c",
    
    # Solvers
    "linear-algebra/solvers/gramschmidt/gramschmidt.c",
    "linear-algebra/solvers/lu/lu.c",

    # High Register Pressure (BLAS)
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",
]

def run_command(cmd):
    """Helper to run shell commands silently"""
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"\n!! Command Failed: {cmd}")
        return False
    return True

def get_exec_time(exe_path):
    """Runs the benchmark and parses the output for time"""
    try:
        times = []
        # Run 3 times to average out noise
        for _ in range(3):
            result = subprocess.run(exe_path, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val:
                times.append(float(val))
        
        if not times: return None
        return sum(times) / len(times)
    except Exception as e:
        return None

def main():
    print(f"Starting Benchmark Suite using LLC: {LLC_PATH}")
    print("REMINDER: Ensure you ran 'ulimit -s unlimited' or large arrays will segfault.")

    # Initialize CSV
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
        # -O1: clean IR but keep loops intact
        # -DLARGE_DATASET: ensure run time is measurable
        # -DPOLYBENCH_STACK_ARRAYS: put data on stack to stress register allocation
        ir_file = f"{bench_name}.ll"
        cmd_ir = (
            f"{CLANG_PATH} -O1 -S -emit-llvm {full_src_path} -o {ir_file} "
            f"-I {POLYBENCH_ROOT}/utilities "
            f"-I {bench_dir} "
            f"-DPOLYBENCH_TIME -DPOLYBENCH_STACK_ARRAYS -DLARGE_DATASET"
        )
        
        if not run_command(cmd_ir):
            print(f"Skipping {bench_name} (IR Gen failed)")
            continue

        for alloc in ALLOCATORS:
            obj_file = f"{bench_name}_{alloc}.o"
            exe_file = f"./{bench_name}_{alloc}"
            
            # 2. COMPILE TO OBJECT (Custom LLC)
            # -O3: Force the backend/allocator to optimize hard
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=obj "
                f"{ir_file} -o {obj_file}"
            )
            
            if not run_command(cmd_llc):
                row_data.append("CompileFail")
                continue

            # 3. LINK (System Clang)
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
                print(" Error/Segfault")
                row_data.append("RunFail")

            # Cleanup binaries
            if os.path.exists(exe_file): os.remove(exe_file)
            if os.path.exists(obj_file): os.remove(obj_file)

        # Save results after every benchmark
        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # Cleanup IR
        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()