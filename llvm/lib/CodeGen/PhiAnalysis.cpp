//===-- PhiAnalysis.cpp - Phi Congruence Analysis -------------------------===//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/PhiAnalysis.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "phi-analysis"

char PhiAnalysis::ID = 0;

INITIALIZE_PASS(PhiAnalysis, "phi-analysis", "Phi Congruence Analysis", false, true)

PhiAnalysis::PhiAnalysis() : MachineFunctionPass(ID) {
  initializePhiAnalysisPass(*PassRegistry::getPassRegistry());
}

void PhiAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool PhiAnalysis::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  
  // 1. Initialize the Union-Find structure
  // Resize to accommodate all virtual registers. 
  // Physical registers are not tracked here (they are constants).
  unsigned NumVirtRegs = MRI.getNumVirtRegs();
  Classes.clear();
  
  // IntEqClasses maps indices 0..N-1. 
  // We will map VirtReg index (VirtReg - FirstVirtReg) to these indices.
  Classes.grow(NumVirtRegs);

  // 2. Iterate all blocks and PHIs
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB.phis()) {
      // PHI Format: %Def = PHI %Use1, %BB1, %Use2, %BB2 ...
      
      Register DefReg = MI.getOperand(0).getReg();
      if (!DefReg.isVirtual()) continue;

      unsigned DefIdx = (unsigned)DefReg & ~(1U << 31); //get id

      for (unsigned i = 1, e = MI.getNumOperands(); i < e; i += 2) {
        Register UseReg = MI.getOperand(i).getReg();
        
        // Only join Virtual Registers. 
        // (Physicals in PHIs are rare in SSA but can happen if pre-colored).
        if (UseReg.isVirtual()) {
          unsigned UseIdx = (unsigned)UseReg & ~(1U << 31);;
          Classes.join(DefIdx, UseIdx);
        }
      }
    }
  }
  
  // 3. Compress paths
  // This ensures getClass() is O(1)
  Classes.compress();
  
  return false; // We just analyzed, didn't modify IR
}

unsigned PhiAnalysis::getClass(Register Reg) const {
  if (!Reg.isVirtual()) return 0;
  // Map VirtReg -> Index -> Leader Index
  return Classes[(unsigned)Reg & ~(1U << 31)];
}

namespace llvm {
  FunctionPass *createPhiAnalysisPass() { return new PhiAnalysis(); }
}