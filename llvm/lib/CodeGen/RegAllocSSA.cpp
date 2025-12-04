//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a minimal implementation of an SSA-based Register Allocator analyzer.
// It simulates the "Chordal Graph Coloring" approach (Linear Scan) to
// determine:
// 1. The maximum register pressure (How many registers are needed).
// 2. The theoretical spill count.
//
// It assumes the input is in SSA form.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
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
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMap>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Helper struct to track the simulation state
  struct RegisterStats {
    unsigned MaxPressure = 0;
    unsigned CurrentPressure = 0;
    unsigned SpillCount = 0;
    unsigned TotalPhysRegs = 0;
  };

  /// Main entry point
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  LiveIntervals *LIS = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  // Perform the "Linear Scan" simulation to count pressure and spills
  void simulateChordalAllocation(MachineFunction &MF);
};

char RASSA::ID = 0;

} // end anonymous namespace

char &llvm::RABasicID = RASSA::ID;

// Initialization
INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

// Implementation of the algorithm
bool RASSA::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** SSA CHORDAL ANALYSIS **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  simulateChordalAllocation(MF);

  // Return false because we are only analyzing, not modifying code yet.
  return false;
}

void RASSA::simulateChordalAllocation(MachineFunction &MF) {
  // Map to store stats per Register Class (e.g., General Purpose vs Floating Point)
  // We cannot mix them; pressure on Float regs doesn't cause spills on Int regs.
  std::map<const TargetRegisterClass *, RegisterStats> ClassStats;

  // 1. Collect all Live Intervals for Virtual Registers
  std::vector<const LiveInterval *> Intervals;
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue; // Ignore unused registers

    const LiveInterval &LI = LIS->getInterval(Reg);
    Intervals.push_back(&LI);

    // Initialize stats for this class if not present
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if (ClassStats.find(RC) == ClassStats.end()) {
      // Modern way to get number of allocatable registers in a class
      BitVector Allocatable = TRI->getAllocatableSet(MF, RC);
      ClassStats[RC].TotalPhysRegs = Allocatable.count();
    }
  }

  // 2. Sort Intervals by Start Position (This is the essence of Linear Scan / Chordal ordering)
  std::sort(Intervals.begin(), Intervals.end(),
            [](const LiveInterval *A, const LiveInterval *B) {
              return A->beginIndex() < B->beginIndex();
            });

  // 3. The "Linear Scan" Simulation
  // We maintain a list of "Active" intervals for each register class.
  std::map<const TargetRegisterClass *, std::vector<const LiveInterval *>> ActiveIntervals;

  for (const LiveInterval *Current : Intervals) {
    Register Reg = Current->reg();
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    RegisterStats &Stats = ClassStats[RC];
    auto &ActiveList = ActiveIntervals[RC];

    // LIBERATE DEAD USES (expireOldIntervals)
    // Remove intervals from the active list that end before the Current one starts.
    // In the 2006 code, this was 'liberate_dead_uses'
    auto It = std::remove_if(ActiveList.begin(), ActiveList.end(),
                             [&](const LiveInterval *Active) {
                               return Active->endIndex() <= Current->beginIndex();
                             });
    ActiveList.erase(It, ActiveList.end());

    // ALLOCATE DEF (addToActive)
    // Add the current interval to the active list.
    ActiveList.push_back(Current);

    // Update Stats
    Stats.CurrentPressure = ActiveList.size();
    if (Stats.CurrentPressure > Stats.MaxPressure) {
      Stats.MaxPressure = Stats.CurrentPressure;
    }

    // CHECK FOR SPILL
    // If active intervals > physical registers, we theoretically spill.
    // (In a real allocator, we would pick the one with lowest spill weight to evict)
    if (Stats.CurrentPressure > Stats.TotalPhysRegs) {
      Stats.SpillCount++;
      
      // For this simulation, to keep the "Active" count realistic to hardware,
      // we pretend we spilled one. In reality, one stays in a register, one goes to stack.
      // So active registers remains equal to TotalPhysRegs.
      // However, for pure pressure analysis, strictly accumulating pressure is interesting.
      // But let's assume we spill the newest one for simple counting:
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

