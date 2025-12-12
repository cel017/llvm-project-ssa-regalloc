//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegAllocBase.h"
#include "AllocationOrder.h"
#include "llvm/ADT/DepthFirstIterator.h" // Essential for DomTree walk
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

/// Helper to calculate spill weights (User provided logic)
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

/// RASSA: SSA-based Register Allocator
/// Implements Chordal Graph Coloring via Dominator Tree Traversal
class RASSA : public MachineFunctionPass,
              public RegAllocBase,
              private LiveRangeEdit::Delegate {
  
  MachineFunction *MF = nullptr;
  MachineDominatorTree *MDT = nullptr;
  std::unique_ptr<Spiller> SpillerInstance;
  
  // Custom weight calculator
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

  // We do NOT use the priority queue, so these are stubs/no-ops
  void enqueueImpl(const LiveInterval *LI) override {} 
  const LiveInterval *dequeue() override { return nullptr; }

  // Core logic methods
  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override;
  
  void allocateRegister(const LiveInterval &VirtReg);
  void processBlock(MachineBasicBlock *MBB, SmallVectorImpl<Register> &NewVRegs);
  
  bool runOnMachineFunction(MachineFunction &mf) override;

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

} 


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
  // Since we don't use a queue, we don't need to re-enqueue.
  // The interval effectively just gets shorter, which is fine for SSA.
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

// Logic to spill interferences (borrowed from user's skeleton)
bool RASSA::spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                               SmallVectorImpl<Register> &SplitVRegs) {
  SmallVector<const LiveInterval *, 8> Intfs;

  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    for (const auto *Intf : reverse(Q.interferingVRegs())) {
      // Check spill weights using our custom calculator
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

// Logic to select a register or spill the current one
MCRegister RASSA::selectOrSplit(const LiveInterval &VirtReg,
                                SmallVectorImpl<Register> &SplitVRegs) {
  SmallVector<MCRegister, 8> PhysRegSpillCands;
  auto Order = AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);

  // 1. Try to find a free register
  for (MCRegister PhysReg : Order) {
    if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_Free)
      return PhysReg;
    
    // If not free, but only blocked by Virtual Regs, it's a spill candidate
    if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_VirtReg)
      PhysRegSpillCands.push_back(PhysReg);
  }

  // 2. Try to spill existing interferences (Eviction)
  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (spillInterferences(VirtReg, PhysReg, SplitVRegs)) {
      // Logic: Fernando's algo liberated colors explicitly. 
      // Here we unassigned the interference, so checkInterference should now be free.
      return PhysReg;
    }
  }

  // 3. Spill the current register
  LLVM_DEBUG(dbgs() << "Spilling current: " << VirtReg << '\n');
  if (!VirtReg.isSpillable()) return ~0u;

  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);
  
  return 0; // 0 indicates we spilled the register we were trying to allocate
}

// The core specific to SSA/Chordal coloring:
// Recursively allocates a specific interval immediately.
// If it spills, it allocates the split products immediately.
void RASSA::allocateRegister(const LiveInterval &VirtReg) {
    SmallVector<Register, 4> SplitVRegs;
    MCRegister PhysReg = selectOrSplit(VirtReg, SplitVRegs);

    if (PhysReg && PhysReg != ~0u) {
        // Successful allocation
        Matrix->assign(VirtReg, PhysReg);
    } else {
        // Spilled. The SplitVRegs (new small intervals from loads/stores) 
        // need to be allocated immediately to maintain the "one pass" invariant.
        for (Register Reg : SplitVRegs) {
             allocateRegister(LIS->getInterval(Reg));
        }
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

  // Initialize helper classes
  WeightCalc = std::make_unique<SpillWeightCalculator>(MRI, MLI);

  // We need standard spill weights calculated for the Spiller to work,
  // even though we use custom logic for comparisons.
  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, MLI,
                      getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI(),
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
  VRAI.calculateSpillWeightsAndHints();

  SpillerInstance.reset(createInlineSpiller(
      {*LIS, getAnalysis<LiveStacksWrapperLegacy>().getLS(), *MDT,
       getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI()},
      *MF, *VRM, VRAI));

  // Traverse the Dominator Tree.
  // This guarantees that when we visit a Node, we have visited all its
  // Strict Dominators (predecessors in the interference graph context).

  for (auto *Node : depth_first(MDT)) {
    MachineBasicBlock *MBB = Node->getBlock();

    // Iterate over instructions in the block
    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr()) continue;
      
      // SKIP PHI NODES as per constraints. 
      // In a full implementation, we would handle PHI definition logic here
      // or rely on a pre-pass to break PHIs.
      if (MI.isPHI()) continue;

      // In SSA, a Virtual Register is defined exactly once.
      // We process the allocation at the definition site.
      for (MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
           Register VirtReg = MO.getReg();
           
           // Ensure we have an interval (sometimes dead code elim might leave weirdness)
           if (LIS->hasInterval(VirtReg)) {
               allocateRegister(LIS->getInterval(VirtReg));
           }
        }
      }
    }
  }

  // Common cleanup
  postOptimization();
  releaseMemory();
  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                    createSSARegisterAllocator);