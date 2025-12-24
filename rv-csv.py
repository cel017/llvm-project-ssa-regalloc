def run_benchmark(file_path, run_cmd, alloc_mode):
    # Prepare command with allocator
    if "-regalloc=" in run_cmd:
        cmd = re.sub(r'-regalloc=\S+', f'-regalloc={alloc_mode}', run_cmd)
    else:
        cmd = f"{run_cmd} -regalloc={alloc_mode}"
    
    # 1. Inject the MATTR flag
    # 2. Explicitly disable machine instruction verification
    cmd = f"{cmd} {LLC_MATTR} -verify-machineinstrs=0 -o {TEMP_ASM}"
    
    try:
        # Capture stderr so we can print it on failure
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='ignore',
            timeout=5,
            check=True
        )
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT EXPIRED"
    except subprocess.CalledProcessError as e:
        return None, None, e.stderr # Return the actual error message

    # Count Ops from the generated asm
    stores, loads = count_ops(TEMP_ASM)
    return stores, loads, None