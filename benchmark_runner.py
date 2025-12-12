import os
import subprocess
import csv
import re

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_spill_counts_starved.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

BENCHMARKS = [
    # Control Flow Heavy (Expect Greedy to win on spills)
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",
    
    # Solvers (Complex Liveness)
    "linear-algebra/solvers/lu/lu.c",
    
    # Stencils
    "medley/floyd-warshall/floyd-warshall.c",
    "medley/deriche/deriche.c",

    # Dense Baseline
    "linear-algebra/blas/gemm/gemm.c",
]

# --- 1. NUCLEAR STARVATION (5 Regs: a0-a4) ---
reserved_regs = []
# Reserve Temps t0-t2 (x5-x7)
for r in range(5, 8): reserved_regs.append(f"+reserve-x{r}")
# Reserve Saved s0-s1 (x8-x9)
for r in range(8, 10): reserved_regs.append(f"+reserve-x{r}")
# Reserve High Args a5-a7 (x15-x17)
for r in range(15, 18): reserved_regs.append(f"+reserve-x{r}")
# Reserve High Saved/Temps (x18-x31)
for r in range(18, 32): reserved_regs.append(f"+reserve-x{r}")

STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

# --- 2. DATASET (Medium Size) ---
# -DSTANDARD_DATASET: Sets correct data types (double/float)
# -Wno-macro-redefined: Allows us to overwrite the size macros without error
# -DNI=256...: Sets size to 256 (Fast runtime, identical pressure pattern)
SIZE_FLAGS = "-DSTANDARD_DATASET -Wno-macro-redefined -DNI=256 -DNJ=256 -DNK=256 -DNL=256 -DNM=256 -DN=256"

def count_spills(asm_file):
    try:
        with open(asm_file, 'r') as f:
            content = f.read()
            # RISC-V Stack Stores: sd/fsd/sw/fsw ... offset(sp)
            spills = re.findall(r'(sd|fsd|sw|fsw)\s+.*,.*\d*\(sp\)', content)
            return len(spills)
    except:
        return 0

def run_command(cmd):
    try:
        # Silencing warnings
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return True

def main():
    print(f"Starting SPILL COUNT Analysis (Starved 5 Regs)")
    print(f"Metric: Number of Store instructions to Stack (Lower is Better)")
    
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
            asm_file = f"{bench_name}_{alloc}.s"
            
            # 2. COMPILE TO ASM
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=asm "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {asm_file}"
            )
            
            if not run_command(cmd_llc):
                # Basic is likely to crash here due to running out of registers
                print(f"  {alloc}: Failed (Ran out of registers?)")
                with open(RESULTS_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([bench_name, alloc, "Failed"])
                continue

            # 3. COUNT SPILLS
            spill_count = count_spills(asm_file)
            print(f"  {alloc}: {spill_count} stores to stack")
            
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([bench_name, alloc, spill_count])
            
            if os.path.exists(asm_file): os.remove(asm_file)

        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()