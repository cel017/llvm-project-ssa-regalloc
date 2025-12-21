; RISC-V Non-Spilling Test
; Run with: llc -mtriple=riscv64 -regalloc=ssa -o -
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64"

define i64 @no_spill(i64 %a, i64 %b) {
entry:
  ; 10 active variables. RISC-V has ~30 GPRs, so this fits easily.
  %v1 = add i64 %a, %b
  %v2 = sub i64 %v1, 10
  %v3 = mul i64 %v2, 2
  %v4 = and i64 %v3, 255
  %v5 = or i64 %v4, 1024
  %v6 = xor i64 %v5, %a
  %v7 = add i64 %v6, %v1
  %v8 = add i64 %v7, %v2
  %v9 = add i64 %v8, %v3
  %v10 = add i64 %v9, %v4

  ; Final result uses a subset, allowing registers to be reclaimed
  %res = add i64 %v10, %v5
  ret i64 %res
}
