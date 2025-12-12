import os
import subprocess
import csv
import re

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_stats.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

BENCHMARKS = [
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/solvers/lu/lu.c",
    "medley/floyd-warshall/floyd-warshall.c",
]

# Starvation Flags (5 Regs)
reserved_regs = []
for r in range(5, 8): reserved_regs.append(f"+reserve-x{r}")
for r in range(8, 10): reserved_regs.append(f"+reserve-x{r}")
for r in range(15, 18): reserved_regs.append(f"+reserve-x{r}")
for r in range(18, 32): reserved_regs.append(f"+reserve-x{r}")
STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

SIZE_FLAGS = "-DSTANDARD_DATASET -Wno-macro-redefined -DNI=256 -DNJ=256 -DNK=256 -DNL=256 -DNM=256 -DN=256"

def get_llvm_spill_count(stderr_output):
    """
    Parses LLVM -stats output for "Number of spills inserted"
    """
    # Look for: "  55 regalloc - Number of spills inserted"
    match = re.search(r'\s+(\d+)\s+regalloc\s+-\s+Number of spills inserted', stderr_output)
    if match:
        return int(match.group(1))
    return 0

def run_command(cmd):
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return True

def main():
    print(f"Starting SPILL COUNT Analysis (Using -stats)")
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Alloc", "SpillCount"])

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
            f"-DPOLYBENCH_TIME -DPOLYBENCH_STACK_ARRAYS {SIZE_FLAGS}"
        )
        
        if not run_command(cmd_ir):
            print("Skipping (IR Gen failed)")
            continue

        for alloc in ALLOCATORS:
            # 2. COMPILE with -stats
            # We capture stderr because that's where stats are printed
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -stats "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o /dev/null" # Throw away the object file, we only want stats
            )
            
            try:
                # Run and capture stderr
                result = subprocess.run(cmd_llc, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"  {alloc}: Failed")
                    continue

                spills = get_llvm_spill_count(result.stderr)
                print(f"  {alloc}: {spills} spills")

                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([bench_name, alloc, spills])
            
            except Exception as e:
                print(f"  {alloc}: Error running llc")

        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()