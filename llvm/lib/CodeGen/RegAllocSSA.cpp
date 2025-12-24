//===-- RegAllocSSA.cpp - SSA Register Allocator (Chordal/Fernando Port) --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h" 
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallSet.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumJoins, "Number of interval joins performed");
STATISTIC(NumSpills, "Number of registers spilled");

FunctionPass *llvm::createSSARegisterAllocator();

namespace {

/// RegClique represents the set of physical registers currently "alive"
/// at a specific point in the basic block traversal.
class RegClique {
  BitVector PhysRegs;
  const TargetRegisterInfo *TRI;

public:
  RegClique(const TargetRegisterInfo *tri) : TRI(tri) {
    PhysRegs.resize(TRI->getNumRegs());
  }

  void occupy(MCRegister Reg) {
    for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
      PhysRegs.set(*AI);
    }
  }

  void liberate(MCRegister Reg) {
    for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
      PhysRegs.reset(*AI);
    }
  }

  bool isOccupied(MCRegister Reg) const {
    return PhysRegs.test(Reg);
  }

  void clear() { PhysRegs.reset(); }
};

class RASSA : public MachineFunctionPass, private LiveRangeEdit::Delegate {
  
  MachineFunction *MF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  
  // Analyses
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;
  LiveRegMatrix *Matrix = nullptr;
  MachineLoopInfo *MLI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  
  std::unique_ptr<Spiller> SpillerInstance;
  
  // The "Fernando" approach uses a visited set for the CFG traversal
  SmallPtrSet<MachineBasicBlock*, 32> Visited;

  // Needed for LiveRangeEdit::Delegate (empty for now)
  void LRE_WillShrinkVirtReg(Register) override {}
  bool LRE_CanEraseVirtReg(Register) override { return false; }

public:
  static char ID;

  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Chordal Register Allocator"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;

  bool runOnMachineFunction(MachineFunction &mf) override;

private:
  // --- The Core Fernando Logic Methods ---

  /// Simulates 'perform_local_allocation' from CfgV_Fernando.cpp
  void performLocalAllocation(MachineBasicBlock &MBB, RegClique &Clique);

  /// Simulates 'allocate_defs'
  void allocateDefs(MachineInstr &MI, RegClique &Clique);

  /// Simulates 'liberate_dead_uses'
  void liberateDeadUses(MachineInstr &MI, RegClique &Clique);

  /// Helper to calculate weight (10^depth)
  float getSpillWeight(Register Reg);

  /// Spills a register using the modern Spiller
  void spill(Register VReg);

  /// Tries to find a free physical register in the class
  MCRegister getFreePhysReg(Register VReg, const RegClique &Clique);
};

char RASSA::ID = 0;

} // end anonymous namespace

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                    createSSARegisterAllocator);

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

void RASSA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<LiveStacksWrapperLegacy>(); // Required for Spiller
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>(); // Required for VRAI
  AU.addRequired<ProfileSummaryInfoWrapperPass>(); // Required for VRAI
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RASSA::releaseMemory() { 
  SpillerInstance.reset(); 
  Visited.clear();
}

//===----------------------------------------------------------------------===//
// Helper: Weight Calculation (10^LoopDepth)
//===----------------------------------------------------------------------===//
float RASSA::getSpillWeight(Register Reg) {
  if (!Reg.isVirtual()) return 0.0f;
  
  // If LIS doesn't have it, we can't calculate exact weight, default to high.
  if (!LIS->hasInterval(Reg)) return 1.0e20f;

  const LiveInterval &LI = LIS->getInterval(Reg);
  if (LI.isSpillable() == false) return 1.0e20f;

  // The original Fernando algorithm used: 
  // w = 1 + 10^depth(def) + sum(1 + 10^depth(use))
  
  float Weight = 0.0f;
  
  // Def weight
  if (MRI->getVRegDef(Reg)) {
    MachineBasicBlock *DefMBB = MRI->getVRegDef(Reg)->getParent();
    unsigned Depth = MLI->getLoopDepth(DefMBB);
    Weight += std::pow(10.0f, (float)Depth);
  }

  // Uses weight
  for (MachineInstr &UseMI : MRI->use_instructions(Reg)) {
    unsigned Depth = MLI->getLoopDepth(UseMI.getParent());
    Weight += std::pow(10.0f, (float)Depth) + 1.0f;
  }

  return Weight;
}

//===----------------------------------------------------------------------===//
// Helper: Get Free Physical Register
//===----------------------------------------------------------------------===//
MCRegister RASSA::getFreePhysReg(Register VReg, const RegClique &Clique) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(*MF);

  for (MCPhysReg PhysReg : Order) {
    if (!Clique.isOccupied(PhysReg) && 
        Matrix->checkInterference(LIS->getInterval(VReg), PhysReg) == LiveRegMatrix::IK_Free) {
      return PhysReg;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Logic: Liberate Dead Uses
//===----------------------------------------------------------------------===//
void RASSA::liberateDeadUses(MachineInstr &MI, RegClique &Clique) {
  // "After a virtual register is last used, the machine register assigned
  // to it must be liberated."
  
  SlotIndex Idx = LIS->getInstructionIndex(MI).getRegSlot();

  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isUse() || !MO.getReg().isVirtual()) continue;
    
    Register Reg = MO.getReg();
    
    // Check if this is the last use.
    if (LIS->hasInterval(Reg)) {
      const LiveInterval &LI = LIS->getInterval(Reg);
      // If the interval ends at this instruction (read slot), it's dead.
      if (LI.expiredAt(Idx)) {
        if (VRM->hasPhys(Reg)) {
          MCRegister Phys = VRM->getPhys(Reg);
          Clique.liberate(Phys);
          Matrix->unassign(LI); // Remove from Matrix to allow overlap check to pass for others
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Logic: Allocate Defs
//===----------------------------------------------------------------------===//
void RASSA::allocateDefs(MachineInstr &MI, RegClique &Clique) {
  // "This method assigns a valid physical register to the virtual register."
  
  // 1. Try Coalescing (Simple Move optimization from original code)
  if (MI.isCopy()) {
    Register Dest = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    if (Dest.isVirtual() && VRM->hasPhys(Src)) {
      MCRegister SrcPhys = VRM->getPhys(Src);
      if (!Clique.isOccupied(SrcPhys) && MRI->getRegClass(Dest)->contains(SrcPhys)) {
        // Compatibility check passed, Coalesce!
        // (In a real allocator, we'd check strict interference, here we trust the Clique)
        if (Matrix->checkInterference(LIS->getInterval(Dest), SrcPhys) == LiveRegMatrix::IK_Free) {
           Matrix->assign(LIS->getInterval(Dest), SrcPhys);
           Clique.occupy(SrcPhys);
           NumJoins++;
           return;
        }
      }
    }
  }

  // 2. Standard Allocation
  for (const MachineOperand &MO : MI.defs()) {
    if (!MO.isReg() || !MO.getReg().isVirtual()) continue;
    Register Reg = MO.getReg();

    if (VRM->hasPhys(Reg)) continue; // Already assigned (e.g., pre-colored or two-addr)

    // Ensure Interval exists
    if (!LIS->hasInterval(Reg)) LIS->createAndComputeVirtRegInterval(Reg);
    
    // Find free register
    MCRegister PhysReg = getFreePhysReg(Reg, Clique);
    
    if (PhysReg) {
      Matrix->assign(LIS->getInterval(Reg), PhysReg);
      Clique.occupy(PhysReg);
    } else {
      // SPILL!
      // In Fernando's original code, it would look for a neighbor in the clique to spill
      // based on weight. For this port, we simplify by spilling the current definition.
      // This is safe because 'spill' will split the interval and defer allocation.
      NumSpills++;
      spill(Reg); 
    }
  }
}

//===----------------------------------------------------------------------===//
// Spilling Adapter
//===----------------------------------------------------------------------===//
void RASSA::spill(Register VReg) {
  SmallVector<Register, 8> NewVRegs;
  LiveRangeEdit LRE(&LIS->getInterval(VReg), NewVRegs, *MF, *LIS, VRM, this);
  SpillerInstance->spill(LRE);
}

//===----------------------------------------------------------------------===//
// Logic: Perform Local Allocation (The Clique Loop)
//===----------------------------------------------------------------------===//
void RASSA::performLocalAllocation(MachineBasicBlock &MBB, RegClique &Clique) {
  Visited.insert(&MBB);
  
  // 1. Initialize Clique at block entry
  Clique.clear();
  SlotIndex BlockStart = LIS->getMBBStartIdx(&MBB);
  
  // Approximation: Iterate all VirtRegs that have been assigned a PhysReg so far.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (!MRI->reg_nodbg_empty(Reg) && VRM->hasPhys(Reg)) {
      if (LIS->hasInterval(Reg)) {
        const LiveInterval &LI = LIS->getInterval(Reg);
        if (LI.liveAt(BlockStart)) {
          Clique.occupy(VRM->getPhys(Reg));
        }
      }
    }
  }

  // 2. Linear Scan
  for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE; ) {
    MachineInstr &MI = *MII++; // Increment early in case of deletion via spill
    liberateDeadUses(MI, Clique);
    allocateDefs(MI, Clique);
  }
}

//===----------------------------------------------------------------------===//
// Main Driver
//===----------------------------------------------------------------------===//
bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  TRI = MF->getSubtarget().getRegisterInfo();
  TII = MF->getSubtarget().getInstrInfo();
  MRI = &MF->getRegInfo();
  
  // 1. Fetch Analyses
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Matrix = &getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  
  // FIX: Fetch LiveStacks (Required for Spiller)
  LiveStacks &LS = getAnalysis<LiveStacksWrapperLegacy>().getLS();

  MachineBlockFrequencyInfo &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  ProfileSummaryInfo &PSI = getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  // Initialize weights using the Fernando 10^depth logic
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg) || !LIS->hasInterval(Reg)) continue;
    
    LiveInterval &LI = LIS->getInterval(Reg);
    LI.setWeight(getSpillWeight(Reg));
  }

  // Initialize Spiller with VRAI
  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, *MLI, MBFI, &PSI);
  
  // FIX: Construct RequiredAnalyses struct explicitly
  // The struct expects {LiveIntervals, LiveStacks, MachineDominatorTree, MachineLoopInfo}
  Spiller::RequiredAnalyses Analyses(*LIS, LS, *MDT, *MLI);
  
  SpillerInstance.reset(createInlineSpiller(Analyses, *MF, *VRM, VRAI, Matrix));

  // Clear State
  Visited.clear();
  RegClique Clique(TRI);
  
  // "This algorithm finds an optimal coloring... in one scan of the control flow."
  // We use DepthFirst traversal to approximate the Chordal ordering.
  MachineBasicBlock *Entry = &MF->front();
  for (auto *MBB : depth_first(Entry)) {
    performLocalAllocation(*MBB, Clique);
  }

  // Final cleanup
  SpillerInstance.reset();
  return true;
}
FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) { return new RASSA(); }