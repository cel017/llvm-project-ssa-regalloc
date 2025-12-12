import os
import subprocess
import csv
import re

# --- CONFIGURATION ---
LLC_PATH = "./build_rv1/bin/llc"
CLANG_PATH = "clang"
POLYBENCH_ROOT = "./polybench-c-4.2.1"
RESULTS_FILE = "results_spill_counts.csv"

ALLOCATORS = ["basic", "greedy", "ssa"]

BENCHMARKS = [
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/solvers/lu/lu.c",
    "medley/floyd-warshall/floyd-warshall.c",
    "medley/deriche/deriche.c",
]

# --- 1. RV32E SIMULATION (16 Registers) ---
reserved_regs = []
for r in range(16, 32): reserved_regs.append(f"+reserve-x{r}")
STARVE_FLAGS = f"-mattr={','.join(reserved_regs)}"

# --- 2. DATASET ---
# Medium size + standard types
SIZE_FLAGS = "-DSTANDARD_DATASET -DNI=256 -DNJ=256 -DNK=256 -DNL=256 -DNM=256 -DN=256 -Wno-macro-redefined"

def count_spills(asm_file):
    """
    Counts store instructions that target the stack pointer (sp).
    RISC-V pattern: 'sd  reg, offset(sp)' or 'fsd reg, offset(sp)'
    """
    try:
        with open(asm_file, 'r') as f:
            content = f.read()
            # Regex to find stores to the stack:
            # (sd|fsd|sw|fsw) -> Store Double/Float/Word
            # \s+ -> whitespace
            # .*, -> register operand
            # .*\d*\(sp\) -> offset(sp)
            spills = re.findall(r'(sd|fsd|sw|fsw)\s+.*,.*\d*\(sp\)', content)
            return len(spills)
    except:
        return 0

def run_command(cmd):
    try:
        # UPDATED: stderr=subprocess.DEVNULL silences the warnings.
        # check_call will still raise an error if the command crashes (non-zero exit code).
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return True

def main():
    print(f"Starting SPILL COUNT Analysis (RV32E + Unroll)")
    print(f"Metric: Number of Store instructions to Stack (Lower is Better)")
    print(f"Warnings are silenced.")
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Alloc", "SpillCount"])

    for bench_path in BENCHMARKS:
        bench_name = os.path.basename(bench_path).replace(".c", "")
        full_src_path = os.path.join(POLYBENCH_ROOT, bench_path)
        bench_dir = os.path.dirname(full_src_path)
        
        print(f"--- Processing: {bench_name} ---")

        # 1. EMIT IR (With Unrolling to force pressure)
        ir_file = f"{bench_name}.ll"
        cmd_ir = (
            f"{CLANG_PATH} -O1 -S -emit-llvm {full_src_path} -o {ir_file} "
            f"-funroll-loops -mllvm -unroll-count=4 " # Force pressure up
            f"-I {POLYBENCH_ROOT}/utilities "
            f"-I {bench_dir} "
            f"-DPOLYBENCH_TIME -DPOLYBENCH_STACK_ARRAYS {SIZE_FLAGS}"
        )
        
        if not run_command(cmd_ir):
            print("Skipping (IR Gen failed)")
            continue

        for alloc in ALLOCATORS:
            asm_file = f"{bench_name}_{alloc}.s" # Generate Assembly text
            
            # 2. COMPILE TO ASM (Not Object) so we can count spills
            cmd_llc = (
                f"{LLC_PATH} -O3 -regalloc={alloc} -filetype=asm "
                f"{STARVE_FLAGS} "
                f"{ir_file} -o {asm_file}"
            )
            
            if not run_command(cmd_llc):
                print(f"  {alloc}: Failed (Crash)")
                continue

            # 3. COUNT SPILLS
            spill_count = count_spills(asm_file)
            print(f"  {alloc}: {spill_count} stores to stack")
            
            # Save
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([bench_name, alloc, spill_count])
            
            if os.path.exists(asm_file): os.remove(asm_file)

        if os.path.exists(ir_file): os.remove(ir_file)

    print(f"\nDone! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()