//===-- llvm/CodeGen/PhiAnalysis.h -===//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PHIANALYSIS_H
#define LLVM_CODEGEN_PHIANALYSIS_H

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {

class PhiAnalysis : public MachineFunctionPass {
  // Disjoint Set Union to track equivalence classes
  IntEqClasses Classes;

public:
  static char ID;

  PhiAnalysis();

  StringRef getPassName() const override { return "Phi Congruence Analysis"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  /// Returns the representative ID for the set containing Reg.
  /// Two registers are phi-related if they have the same ID.
  unsigned getClass(Register Reg) const;
  
  /// Returns the underlying equivalence class data structure
  const IntEqClasses &getClasses() const { return Classes; }
};

} // namespace llvm

#endif // LLVM_CODEGEN_PHIANALYSIS_H