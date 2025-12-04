import os

# Configuration
TEST_DIR = "497"
PREFIXES = ['x86', 'aarch64', 'arm', 'mips', 'powerpc', 'hexagon', 'nvptx', 'amdgpu', 'systemz', 'wasm']

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Directory '{TEST_DIR}' not found.")
        return

    print(f"Scanning '{TEST_DIR}' for files starting with: {', '.join(PREFIXES)}")
    
    files = os.listdir(TEST_DIR)
    removed_count = 0

    for filename in files:
        # Check if file starts with any of the forbidden prefixes
        # distinct from "contains" - purely checks the start of the filename
        if any(filename.lower().startswith(prefix) for prefix in PREFIXES):
            file_path = os.path.join(TEST_DIR, filename)
            try:
                os.remove(file_path)
                removed_count += 1
            except OSError as e:
                print(f"Error removing {filename}: {e}")

    print(f"Done. Removed {removed_count} files.")

if __name__ == "__main__":
    main()