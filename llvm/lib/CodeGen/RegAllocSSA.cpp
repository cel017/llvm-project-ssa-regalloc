//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Analyzer calculates:
// 1. Theoretical Spills: (Max Pressure - Available Regs)
// 2. Realistic Register Usage: (Max Pressure + ABI/Fixed Physical Registers)
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h" 
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <algorithm>
#include <map>
#include <set>

using namespace llvm;

FunctionPass *llvm::createSSARegisterAllocator();

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                      createSSARegisterAllocator);

namespace {
class RASSA : public MachineFunctionPass {
public:
  static char ID;
  RASSA() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SSA Register Allocator Analyzer"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  struct RegisterStats {
    unsigned MaxPressure = 0;
    unsigned SpillCount = 0;
    unsigned TotalPhysRegs = 0;
    unsigned FixedPhysRegsTouched = 0; // New metric for realism
  };

  bool runOnMachineFunction(MachineFunction &MF) override;
  void simulateChordalAllocation(MachineFunction &MF);

private:
  LiveIntervals *LIS = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
};
char RASSA::ID = 0;
} 

namespace llvm {
  void initializeRASSAPass(PassRegistry&);
  void initializeLiveIntervalsWrapperPassPass(PassRegistry&);
}

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

bool RASSA::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  simulateChordalAllocation(MF);
  return false;
}

void RASSA::simulateChordalAllocation(MachineFunction &MF) {
  std::map<const TargetRegisterClass *, RegisterStats> ClassStats;
  std::vector<const LiveInterval *> Intervals;

  // --- STEP 1: Count Fixed Physical Registers (The "ABI Reality" check) ---
  // This detects registers like a0-a7, ra, sp that are explicitly used 
  // by instructions, even if they aren't "virtual" registers.
  std::map<const TargetRegisterClass *, std::set<MCRegister>> FixedUsage;
  
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.getReg().isPhysical()) {
          Register Reg = MO.getReg();
          // Skip the zero register (x0) or PC as they don't count towards allocation
          if (Reg == 0) continue; 
          
          // Find which class this register belongs to
          for (const TargetRegisterClass *RC : TRI->regclasses()) {
            if (RC->contains(Reg)) {
              FixedUsage[RC].insert(Reg);
              // Break after finding the smallest/most specific class? 
              // Actually, simplified: we assign it to the class we are analyzing later.
            }
          }
        }
      }
    }
  }

  // --- STEP 2: Collect Virtual Intervals ---
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg)) continue; 
    Intervals.push_back(&LIS->getInterval(Reg));

    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if (ClassStats.find(RC) == ClassStats.end()) {
      BitVector Allocatable = TRI->getAllocatableSet(MF, RC);
      ClassStats[RC].TotalPhysRegs = Allocatable.count();
    }
  }

  // --- STEP 3: Sort (Linear Scan) ---
  std::sort(Intervals.begin(), Intervals.end(),
            [](const LiveInterval *A, const LiveInterval *B) {
              return A->beginIndex() < B->beginIndex();
            });

  // --- STEP 4: Simulate Pressure ---
  std::map<const TargetRegisterClass *, std::vector<const LiveInterval *>> ActiveIntervals;

  for (const LiveInterval *Current : Intervals) {
    const TargetRegisterClass *RC = MRI->getRegClass(Current->reg());
    RegisterStats &Stats = ClassStats[RC];
    auto &ActiveList = ActiveIntervals[RC];

    // Remove dead
    auto It = std::remove_if(ActiveList.begin(), ActiveList.end(),
                             [&](const LiveInterval *Active) {
                               return Active->endIndex() <= Current->beginIndex();
                             });
    ActiveList.erase(It, ActiveList.end());

    // Add new
    ActiveList.push_back(Current);

    // Update Max Pressure
    unsigned currentPressure = ActiveList.size();
    if (currentPressure > Stats.MaxPressure) {
      Stats.MaxPressure = currentPressure;
    }

    // Spill check
    if (currentPressure > Stats.TotalPhysRegs) {
      Stats.SpillCount++;
      ActiveList.pop_back(); 
    }
  }

  // --- STEP 5: Output Realistic Stats ---
  for (auto &Pair : ClassStats) {
    const TargetRegisterClass *RC = Pair.first;
    RegisterStats &Stats = Pair.second;

    // Estimate Fixed registers used for this class
    // We take a rough count of physical registers encountered in the code
    unsigned fixedCount = 0;
    // Iterate our FixedUsage map. If the fixed reg is allocatable in this class, count it.
    for (auto &Entry : FixedUsage) {
       const TargetRegisterClass *FixedRC = Entry.first;
       if (FixedRC == RC || RC->hasSuperClass(FixedRC) || FixedRC->hasSubClass(RC)) {
           // Simple heuristic: Take the set size if it matches class
           if (Entry.second.size() > fixedCount) fixedCount = Entry.second.size();
       }
    }

    // REALISM FORMULA: Max VReg Pressure + Fixed Physical Registers Used
    unsigned realisticRegs = Stats.MaxPressure + fixedCount;

    // Cap it at TotalPhysRegs so we don't report using 40 registers on a 32-register machine
    // (unless we spilled, but 'regs used' usually implies distinct hardware regs)
    if (realisticRegs > Stats.TotalPhysRegs && Stats.SpillCount == 0) {
        realisticRegs = Stats.TotalPhysRegs; 
    }

    errs() << "@SSA_REPORT "
           << "func=" << MF.getName() << " "
           << "spills=" << Stats.SpillCount << " "
           << "pressure=" << realisticRegs << "\n";
  }
  errs().flush(); 
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) { return new RASSA(); }