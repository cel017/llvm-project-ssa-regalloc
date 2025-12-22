// no_spill.c
// Pure C, no headers required.
// "long long" is 64-bit on RISC-V.

long long no_spill(long long a, long long b) {
    long long v1 = a + b;
    long long v2 = v1 - 10;
    long long v3 = v2 * 2;
    long long v4 = v3 & 255;
    long long v5 = v4 | 1024;
    long long v6 = v5 ^ a;
    
    // Create dependencies to keep variables alive
    long long v7 = v6 + v1;  // Keeps v1 alive
    long long v8 = v7 + v2;  // Keeps v2 alive
    long long v9 = v8 + v3;  // Keeps v3 alive
    long long v10 = v9 + v4; // Keeps v4 alive

    // Final result uses v10 and v5, ensuring v5 stayed alive
    return v10 + v5;
}