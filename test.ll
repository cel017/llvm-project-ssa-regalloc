; RISC-V 64-bit Spilling Test (No PHIs)
; Run with: llc -mtriple=riscv64 -regalloc=ssa -o -
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64"

define i64 @force_spill(i64 %a) {
entry:
  ; Define 40 independent values
  %v1 = add i64 %a, 1
  %v2 = add i64 %a, 2
  %v3 = add i64 %a, 3
  %v4 = add i64 %a, 4
  %v5 = add i64 %a, 5
  %v6 = add i64 %a, 6
  %v7 = add i64 %a, 7
  %v8 = add i64 %a, 8
  %v9 = add i64 %a, 9
  %v10 = add i64 %a, 10
  %v11 = add i64 %a, 11
  %v12 = add i64 %a, 12
  %v13 = add i64 %a, 13
  %v14 = add i64 %a, 14
  %v15 = add i64 %a, 15
  %v16 = add i64 %a, 16
  %v17 = add i64 %a, 17
  %v18 = add i64 %a, 18
  %v19 = add i64 %a, 19
  %v20 = add i64 %a, 20
  %v21 = add i64 %a, 21
  %v22 = add i64 %a, 22
  %v23 = add i64 %a, 23
  %v24 = add i64 %a, 24
  %v25 = add i64 %a, 25
  %v26 = add i64 %a, 26
  %v27 = add i64 %a, 27
  %v28 = add i64 %a, 28
  %v29 = add i64 %a, 29
  %v30 = add i64 %a, 30
  %v31 = add i64 %a, 31
  %v32 = add i64 %a, 32
  %v33 = add i64 %a, 33
  %v34 = add i64 %a, 34
  %v35 = add i64 %a, 35
  %v36 = add i64 %a, 36
  %v37 = add i64 %a, 37
  %v38 = add i64 %a, 38
  %v39 = add i64 %a, 39
  %v40 = add i64 %a, 40

  ; Use all of them at once to force high register pressure
  %sum1 = add i64 %v1, %v2
  %sum2 = add i64 %sum1, %v3
  %sum3 = add i64 %sum2, %v4
  %sum4 = add i64 %sum3, %v5
  %sum5 = add i64 %sum4, %v6
  %sum6 = add i64 %sum5, %v7
  %sum7 = add i64 %sum6, %v8
  %sum8 = add i64 %sum7, %v9
  %sum9 = add i64 %sum8, %v10
  %sum10 = add i64 %sum9, %v11
  %sum11 = add i64 %sum10, %v12
  %sum12 = add i64 %sum11, %v13
  %sum13 = add i64 %sum12, %v14
  %sum14 = add i64 %sum13, %v15
  %sum15 = add i64 %sum14, %v16
  %sum16 = add i64 %sum15, %v17
  %sum17 = add i64 %sum16, %v18
  %sum18 = add i64 %sum17, %v19
  %sum19 = add i64 %sum18, %v20
  %sum20 = add i64 %sum19, %v21
  %sum21 = add i64 %sum20, %v22
  %sum22 = add i64 %sum21, %v23
  %sum23 = add i64 %sum22, %v24
  %sum24 = add i64 %sum23, %v25
  %sum25 = add i64 %sum24, %v26
  %sum26 = add i64 %sum25, %v27
  %sum27 = add i64 %sum26, %v28
  %sum28 = add i64 %sum27, %v29
  %sum29 = add i64 %sum28, %v30
  %sum30 = add i64 %sum29, %v31
  %sum31 = add i64 %sum30, %v32
  %sum32 = add i64 %sum31, %v33
  %sum33 = add i64 %sum32, %v34
  %sum34 = add i64 %sum33, %v35
  %sum35 = add i64 %sum34, %v36
  %sum36 = add i64 %sum35, %v37
  %sum37 = add i64 %sum36, %v38
  %sum38 = add i64 %sum37, %v39
  %total = add i64 %sum38, %v40

  ret i64 %total
}
