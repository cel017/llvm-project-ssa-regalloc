//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a minimal implementation of an SSA-based Register Allocator analyzer.
// It simulates the "Chordal Graph Coloring" (Linear Scan) to determine
// register pressure and theoretical spill counts.
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <algorithm>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "regalloc-ssa"

// Forward declare the creator
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
    // We only need LiveIntervals to analyze pressure
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  struct RegisterStats {
    unsigned MaxPressure = 0;
    unsigned CurrentPressure = 0;
    unsigned SpillCount = 0;
    unsigned TotalPhysRegs = 0;
  };

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  LiveIntervals *LIS = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  void simulateChordalAllocation(MachineFunction &MF);
};

char RASSA::ID = 0;

} // end anonymous namespace

// === Forward Declarations ===
// Explicitly declare the initialization functions we need.
// This bypasses header lookup issues.
namespace llvm {
  void initializeRASSAPass(PassRegistry&);
  void initializeLiveIntervalsWrapperPassPass(PassRegistry&);
}

// === Initialization ===
INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

// Implementation
bool RASSA::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** SSA CHORDAL ANALYSIS **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  // Get the analyses
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  simulateChordalAllocation(MF);

  // Return false because we are only analyzing, not modifying code.
  return false;
}

void RASSA::simulateChordalAllocation(MachineFunction &MF) {
  // Store stats per Register Class
  std::map<const TargetRegisterClass *, RegisterStats> ClassStats;
  std::vector<const LiveInterval *> Intervals;

  // 1. Collect all Live Intervals for Virtual Registers
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    // Skip unused registers
    if (MRI->reg_nodbg_empty(Reg))
      continue; 

    const LiveInterval &LI = LIS->getInterval(Reg);
    Intervals.push_back(&LI);

    // Initialize stats for this class if not present
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if (ClassStats.find(RC) == ClassStats.end()) {
      BitVector Allocatable = TRI->getAllocatableSet(MF, RC);
      ClassStats[RC].TotalPhysRegs = Allocatable.count();
    }
  }

  // 2. Sort Intervals by Start Position (Linear Scan / Chordal ordering)
  std::sort(Intervals.begin(), Intervals.end(),
            [](const LiveInterval *A, const LiveInterval *B) {
              return A->beginIndex() < B->beginIndex();
            });

  // 3. The "Linear Scan" Simulation
  std::map<const TargetRegisterClass *, std::vector<const LiveInterval *>> ActiveIntervals;

  for (const LiveInterval *Current : Intervals) {
    Register Reg = Current->reg();
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    RegisterStats &Stats = ClassStats[RC];
    auto &ActiveList = ActiveIntervals[RC];

    // LIBERATE DEAD USES: Remove intervals that end before Current starts
    auto It = std::remove_if(ActiveList.begin(), ActiveList.end(),
                             [&](const LiveInterval *Active) {
                               return Active->endIndex() <= Current->beginIndex();
                             });
    ActiveList.erase(It, ActiveList.end());

    // ALLOCATE DEF: Add current interval to active
    ActiveList.push_back(Current);

    // Update Pressure Stats
    Stats.CurrentPressure = ActiveList.size();
    if (Stats.CurrentPressure > Stats.MaxPressure) {
      Stats.MaxPressure = Stats.CurrentPressure;
    }

    // CHECK FOR SPILL
    if (Stats.CurrentPressure > Stats.TotalPhysRegs) {
      Stats.SpillCount++;
      // For simulation, we assume the oldest (or newest) is spilled to keep
      // the "Active in Register" count accurate.
      ActiveList.pop_back(); 
    }
  }

  // 4. Print Results
  errs() << "-------------------------------------------------\n";
  errs() << "RegAllocSSA Analysis Report for " << MF.getName() << "\n";
  errs() << "-------------------------------------------------\n";
  
  for (auto &Pair : ClassStats) {
    const TargetRegisterClass *RC = Pair.first;
    RegisterStats &Stats = Pair.second;

    errs() << "RegisterClass: " << TRI->getRegClassName(RC) << "\n";
    errs() << "  Available PhysRegs: " << Stats.TotalPhysRegs << "\n";
    errs() << "  Max Regs Needed:    " << Stats.MaxPressure << "\n";
    errs() << "  Theoretical Spills: " << Stats.SpillCount << "\n";
    
    if (Stats.SpillCount > 0) {
        errs() << "  [!] Spill Required.\n";
    } else {
        errs() << "  [OK] No Spills.\n";
    }
    errs() << "\n";
  }
}

FunctionPass *llvm::createSSARegisterAllocator() {
  return new RASSA();
}

FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) {
  return new RASSA();
}