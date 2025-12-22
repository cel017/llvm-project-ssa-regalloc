#include <cstdint>

extern "C" {

// A function that creates 10 live values.
// We use simple arithmetic to ensure dependencies keep variables alive.
int64_t no_spill(int64_t a, int64_t b) {
    int64_t v1 = a + b;
    int64_t v2 = v1 - 10;
    int64_t v3 = v2 * 2;
    int64_t v4 = v3 & 255;
    int64_t v5 = v4 | 1024;
    int64_t v6 = v5 ^ a;
    
    // v7 uses v6 and v1, preventing v1's register from being reused yet
    int64_t v7 = v6 + v1;
    
    // v8 uses v7 and v2, keeping v2 alive
    int64_t v8 = v7 + v2;
    
    // v9 uses v8 and v3, keeping v3 alive
    int64_t v9 = v8 + v3;
    
    // v10 uses v9 and v4, keeping v4 alive
    int64_t v10 = v9 + v4;

    // Final result uses v10 and v5, ensuring v5 stayed alive until the end
    int64_t res = v10 + v5;
    
    return res;
}

}