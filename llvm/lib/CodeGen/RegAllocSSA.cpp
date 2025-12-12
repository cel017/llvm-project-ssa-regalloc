//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegAllocBase.h"
#include "AllocationOrder.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

// Forward declaration
FunctionPass *llvm::createSSARegisterAllocator();

/// Helper to calculate spill weights
class SpillWeightCalculator {
  const MachineRegisterInfo &MRI;
  const MachineLoopInfo &MLI;
  static constexpr unsigned Pow10[] = { 
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000 
  };

  unsigned getLoopWeight(const MachineBasicBlock *MBB) const {
    unsigned Depth = MLI.getLoopDepth(MBB);
    if (Depth >= std::size(Pow10)) Depth = std::size(Pow10) - 1;
    return Pow10[Depth];
  }

public:
  SpillWeightCalculator(const MachineRegisterInfo &mri, const MachineLoopInfo &mli) 
    : MRI(mri), MLI(mli) {}

  unsigned getWeight(Register Reg) const {
    if (!Reg.isVirtual()) return 0;
    unsigned W = 0;
    MachineInstr *DefMI = MRI.getVRegDef(Reg);
    if (DefMI) W += 1 + getLoopWeight(DefMI->getParent());
    for (MachineInstr &UseMI : MRI.reg_nodbg_instructions(Reg)) {
      if (&UseMI == DefMI) continue;          
        W += 1 + getLoopWeight(UseMI.getParent());
    }
    return W;
  }
};

class RASSA : public MachineFunctionPass,
              public RegAllocBase,
              private LiveRangeEdit::Delegate {
  
  MachineFunction *MF = nullptr;
  MachineDominatorTree *MDT = nullptr;
  std::unique_ptr<Spiller> SpillerInstance;
  std::unique_ptr<SpillWeightCalculator> WeightCalc;

  // LRE Delegate methods
  bool LRE_CanEraseVirtReg(Register) override;
  void LRE_WillShrinkVirtReg(Register) override;

public:
  static char ID;

  RASSA() : MachineFunctionPass(ID), RegAllocBase() {}

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  Spiller &spiller() override { return *SpillerInstance; }

  // Unused standard RegAllocBase queue methods
  void enqueueImpl(const LiveInterval *LI) override {} 
  const LiveInterval *dequeue() override { return nullptr; }

  // --- Core Allocation Logic (Mapped to Fernando's Implementation) ---

  /// 1. Main Driver: Iterates CFG in Dominator Tree Order
  bool runOnMachineFunction(MachineFunction &mf) override;

  /// 2. Perform Local Allocation: Matches `perform_local_allocation`
  ///    Iterates instructions in the block.
  void performLocalAllocation(MachineBasicBlock &MBB);

  /// 3. Allocate Defs: Matches `allocate_defs`
  ///    Assigns registers to definitions in the current instruction.
  void allocateDefs(MachineInstr &MI);

  /// 4. Recursive Allocation Helper
  ///    Handles the actual selection and immediate splitting/coloring.
  ///    (Combines parts of `allocate_spilled_uses` and `assign_colors`)
  void allocateRegister(const LiveInterval &VirtReg);

  // --- End Core Logic ---

  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(MachineFunctionProperties::Property::NoPHIs);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(MachineFunctionProperties::Property::IsSSA);
  }

private:
  bool spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                          SmallVectorImpl<Register> &SplitVRegs);
};

char RASSA::ID = 0;

// === Implementation ===

bool RASSA::LRE_CanEraseVirtReg(Register VirtReg) {
  LiveInterval &LI = LIS->getInterval(VirtReg);
  if (VRM->hasPhys(VirtReg)) {
    Matrix->unassign(LI);
    aboutToRemoveInterval(LI);
    return true;
  }
  LI.clear();
  return false;
}

void RASSA::LRE_WillShrinkVirtReg(Register VirtReg) {
  if (VRM->hasPhys(VirtReg))
    Matrix->unassign(LIS->getInterval(VirtReg));
}

void RASSA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>();
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addRequired<LiveDebugVariablesWrapperLegacy>();
  AU.addPreserved<LiveDebugVariablesWrapperLegacy>();
  AU.addRequired<LiveStacksWrapperLegacy>();
  AU.addPreserved<LiveStacksWrapperLegacy>();
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.addPreserved<LiveRegMatrixWrapperLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RASSA::releaseMemory() {
  SpillerInstance.reset();
  WeightCalc.reset();
}

bool RASSA::spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                               SmallVectorImpl<Register> &SplitVRegs) {
  SmallVector<const LiveInterval *, 8> Intfs;

  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    for (const auto *Intf : reverse(Q.interferingVRegs())) {
      unsigned IntfWeight = WeightCalc->getWeight(Intf->reg());
      unsigned VirtWeight = WeightCalc->getWeight(VirtReg.reg());

      if (!Intf->isSpillable() || IntfWeight > VirtWeight)
        return false;
      Intfs.push_back(Intf);
    }
  }

  for (const LiveInterval *Spill : Intfs) {
    if (!VRM->hasPhys(Spill->reg())) continue;
    Matrix->unassign(*Spill);
    LiveRangeEdit LRE(Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }
  return true;
}

MCRegister RASSA::selectOrSplit(const LiveInterval &VirtReg,
                                SmallVectorImpl<Register> &SplitVRegs) {
  SmallVector<MCRegister, 8> PhysRegSpillCands;
  auto Order = AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);

  // 1. Try free registers
  for (MCRegister PhysReg : Order) {
    if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_Free)
      return PhysReg;
    if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_VirtReg)
      PhysRegSpillCands.push_back(PhysReg);
  }

  // 2. Try evicting (spilling) interferences
  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (spillInterferences(VirtReg, PhysReg, SplitVRegs))
      return PhysReg;
  }

  // 3. Spill current
  if (!VirtReg.isSpillable()) return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);
  return 0;
}

// === Fernando Structure Breakdown ===

// Recursive allocation to handle immediate splitting/spilling.
// This replaces Fernando's `allocate_spilled_uses` logic (which manually found regs for reloads)
// because modern Spillers create new VRegs for reloads, which we simply recursively color here.
void RASSA::allocateRegister(const LiveInterval &VirtReg) {
    SmallVector<Register, 4> SplitVRegs;
    MCRegister PhysReg = selectOrSplit(VirtReg, SplitVRegs);

    if (PhysReg && PhysReg != ~0u) {
        Matrix->assign(VirtReg, PhysReg);
    } else {
        // Recurse on the new split products (loads/stores) immediately
        // to maintain the "Single Pass" invariant.
        for (Register Reg : SplitVRegs) {
             allocateRegister(LIS->getInterval(Reg));
        }
    }
}

// Matches `allocate_defs`
void RASSA::allocateDefs(MachineInstr &MI) {
    for (MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
           Register VirtReg = MO.getReg();
           if (LIS->hasInterval(VirtReg)) {
               allocateRegister(LIS->getInterval(VirtReg));
           }
        }
    }
}

// Matches `perform_local_allocation`
void RASSA::performLocalAllocation(MachineBasicBlock &MBB) {
    for (MachineInstr &MI : MBB) {
        if (MI.isDebugInstr()) continue;
        
        // Skip PHIs (handled by elimination pass, not implemented yet)
        if (MI.isPHI()) continue;

        // Note: `liberate_dead_uses` from Fernando's code is implicit here.
        // LiveRegMatrix checks interference against LiveIntervals. If a use
        // ended at the previous index, it automatically stops interfering.
        
        // Allocate definitions
        allocateDefs(MI);

        // Note: `liberate_dead_defs` is also implicit.
    }
}

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** SSA REGISTER ALLOCATION (Chordal) **********\n");

  MF = &mf;
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MRI = MF->getRegInfo();
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  RegAllocBase::init(getAnalysis<VirtRegMapWrapperLegacy>().getVRM(),
                     getAnalysis<LiveIntervalsWrapperPass>().getLIS(),
                     getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM());

  WeightCalc = std::make_unique<SpillWeightCalculator>(MRI, MLI);

  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, MLI,
                      getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI(),
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
  VRAI.calculateSpillWeightsAndHints();

  SpillerInstance.reset(createInlineSpiller(
      {*LIS, getAnalysis<LiveStacksWrapperLegacy>().getLS(), *MDT,
       getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI()},
      *MF, *VRM, VRAI));

  // Main Loop: Matches the graph traversal in Fernando's `runOnMachineFunction`
  // Instead of building a clique list, we traverse the Dominator Tree.
  for (auto *Node : depth_first(MDT)) {
    performLocalAllocation(*Node->getBlock());
  }

  postOptimization();
  releaseMemory();
  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                    createSSARegisterAllocator);