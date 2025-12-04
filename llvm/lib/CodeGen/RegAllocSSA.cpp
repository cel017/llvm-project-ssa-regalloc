//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REALISTIC ANALYZER:
// 1. Calculates Base Pressure (Chordal Graph).
// 2. Simulates ABI Constraints:
//    - Variables live across CALLs must fit in Callee-Saved Regs.
//    - If they don't fit, they SPILL.
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
    unsigned CalleeSavedLimit = 12; 
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

  // --- STEP 0: Configure Limits based on Architecture ---
  // If RV32E is detected (16 regs total), the saved pool is tiny.
  bool isRV32E = MF.getSubtarget().getFeatureString().contains("+e");
  unsigned savedRegLimit = isRV32E ? 2 : 12; // s0-s1 vs s0-s11

  // --- STEP 1: Find all Call Sites ---
  std::vector<SlotIndex> CallSites;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isCall()) {
        CallSites.push_back(LIS->getInstructionIndex(MI));
      }
    }
  }

  // --- STEP 2: Collect Virtual Intervals ---
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg)) continue; 
    
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);

    // FIXED: Removed !TRI->isAllocatableClass(RC)
    // Instead, we check the bitvector below.
    BitVector Allocatable = TRI->getAllocatableSet(MF, RC);
    if (Allocatable.none()) continue; // If 0 allocatable regs, ignore this class

    Intervals.push_back(&LIS->getInterval(Reg));

    if (ClassStats.find(RC) == ClassStats.end()) {
      ClassStats[RC].TotalPhysRegs = Allocatable.count();
      ClassStats[RC].CalleeSavedLimit = savedRegLimit;
    }
  }

  // --- STEP 3: Sort (Linear Scan) ---
  std::sort(Intervals.begin(), Intervals.end(),
            [](const LiveInterval *A, const LiveInterval *B) {
              return A->beginIndex() < B->beginIndex();
            });

  // --- STEP 4: Simulation ---
  std::map<const TargetRegisterClass *, std::vector<const LiveInterval *>> ActiveIntervals;
  std::map<const TargetRegisterClass *, std::vector<const LiveInterval *>> ActiveAcrossCalls;

  for (const LiveInterval *Current : Intervals) {
    const TargetRegisterClass *RC = MRI->getRegClass(Current->reg());
    RegisterStats &Stats = ClassStats[RC];
    auto &ActiveList = ActiveIntervals[RC];
    auto &CallList = ActiveAcrossCalls[RC]; 

    // 4a. Expire Old Intervals
    auto It = std::remove_if(ActiveList.begin(), ActiveList.end(),
                             [&](const LiveInterval *Active) {
                               return Active->endIndex() <= Current->beginIndex();
                             });
    ActiveList.erase(It, ActiveList.end());

    auto CallIt = std::remove_if(CallList.begin(), CallList.end(),
                             [&](const LiveInterval *Active) {
                               return Active->endIndex() <= Current->beginIndex();
                             });
    CallList.erase(CallIt, CallList.end());


    // 4b. Add New Interval
    ActiveList.push_back(Current);

    // 4c. Check if THIS interval crosses any Call Site
    bool crossesCall = false;
    for (SlotIndex CallIdx : CallSites) {
      // FIXED: Used liveAt() instead of covers()
      if (Current->liveAt(CallIdx)) {
        crossesCall = true;
        break;
      }
    }

    if (crossesCall) {
        CallList.push_back(Current);
    }

    // --- REALISTIC PRESSURE CHECK ---
    unsigned globalPressure = ActiveList.size();
    if (globalPressure > Stats.MaxPressure) {
      Stats.MaxPressure = globalPressure;
    }

    // ABI Bottleneck Check
    bool forcedSpill = false;
    if (CallList.size() > Stats.CalleeSavedLimit) {
        forcedSpill = true;
    }

    bool standardSpill = (globalPressure > Stats.TotalPhysRegs);

    if (forcedSpill || standardSpill) {
      Stats.SpillCount++;
      ActiveList.pop_back(); 
      if (forcedSpill && !CallList.empty()) CallList.pop_back(); 
    }
  }

  // --- STEP 5: Output ---
  for (auto &Pair : ClassStats) {
    const TargetRegisterClass *RC = Pair.first;
    RegisterStats &Stats = Pair.second;

    errs() << "@SSA_REPORT "
           << "func=" << MF.getName() << " "
           << "spills=" << Stats.SpillCount << " "
           << "pressure=" << Stats.MaxPressure << "\n";
  }
  errs().flush(); 
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) { return new RASSA(); }