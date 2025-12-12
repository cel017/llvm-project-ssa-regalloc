//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RASSA function pass, which provides a minimal
// implementation of the basic register allocator (adapted for SSA context).
//
//===----------------------------------------------------------------------===//

#include "RegAllocBase.h" // Essential base class
#include "AllocationOrder.h"
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
#include <queue>

#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

// Forward declaration for the factory
FunctionPass *llvm::createSSARegisterAllocator();
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F);

namespace {

//===----------------------------------------------------------------------===//
// Implementation of the spill weight calculation from the paper
//===----------------------------------------------------------------------===//
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

    // --- 1. Definition Site ---
    if (DefMI) {
      if (DefMI->isPHI()) {
        for (unsigned i = 1, e = DefMI->getNumOperands(); i < e; i += 2) {
          // i is Value, i+1 is MBB
          MachineBasicBlock *IncomingMBB = DefMI->getOperand(i + 1).getMBB();
          W += 1 + getLoopWeight(IncomingMBB);
        }
      } else {
        W += 1 + getLoopWeight(DefMI->getParent());
      }
    }

    // --- 2. Use Sites ---
    for (MachineInstr &UseMI : MRI.reg_nodbg_instructions(Reg)) {
      if (&UseMI == DefMI) continue;

      if (UseMI.isPHI()) {
        for (unsigned i = 1, e = UseMI.getNumOperands(); i < e; i += 2) {
          // If this operand (Val) matches our register...
          if (UseMI.getOperand(i).isReg() && UseMI.getOperand(i).getReg() == Reg) {
             // ... then the weight comes from the associated MBB (Val+1)
             MachineBasicBlock *IncomingMBB = UseMI.getOperand(i + 1).getMBB();
             W += 1 + getLoopWeight(IncomingMBB);
          }
        }
      } else {
        W += 1 + getLoopWeight(UseMI.getParent());
      }
    }
    return W;
  }
};

struct CompSpillWeight {
  bool operator()(const LiveInterval *A, const LiveInterval *B) const {
    return A->weight() < B->weight();
  }
};

/// RASSA provides a minimal implementation of the basic register allocation
/// algorithm. It prioritizes live virtual registers by spill weight and spills
/// whenever a register is unavailable.
class RASSA : public MachineFunctionPass,
              public RegAllocBase,
              private LiveRangeEdit::Delegate {
  // context
  MachineFunction *MF = nullptr;

  // state
  std::unique_ptr<Spiller> SpillerInstance;
  std::priority_queue<const LiveInterval *, std::vector<const LiveInterval *>,
                      CompSpillWeight>
      Queue;

  // Scratch space. Allocated here to avoid repeated malloc calls in
  // selectOrSplit().
  BitVector UsableRegs;

  bool LRE_CanEraseVirtReg(Register) override;
  void LRE_WillShrinkVirtReg(Register) override;

public:
  static char ID;

  RASSA(const RegAllocFilterFunc F = nullptr);

  /// Return the pass name.
  StringRef getPassName() const override { return "SSA Register Allocator"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }
  /// RASSA analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override;

  Spiller &spiller() override { return *SpillerInstance; }

  void enqueueImpl(const LiveInterval *LI) override { Queue.push(LI); }

  const LiveInterval *dequeue() override {
    if (Queue.empty())
      return nullptr;
    const LiveInterval *LI = Queue.top();
    Queue.pop();
    return LI;
  }

  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override;

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &mf) override;

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties();
  }

  // Helper for spilling all live virtual registers currently unified under preg
  // that interfere with the most recently queried lvr. Return true if spilling
  // was successful, and append any new spilled/split intervals to splitLVRs.
  bool spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                          SmallVectorImpl<Register> &SplitVRegs);
};

char RASSA::ID = 0;

} // end anonymous namespace

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                    createSSARegisterAllocator);

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariablesWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescerLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineSchedulerLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false,
                    false)

bool RASSA::LRE_CanEraseVirtReg(Register VirtReg) {
  LiveInterval &LI = LIS->getInterval(VirtReg);
  if (VRM->hasPhys(VirtReg)) {
    Matrix->unassign(LI);
    aboutToRemoveInterval(LI);
    return true;
  }
  // Unassigned virtreg is probably in the priority queue.
  // RegAllocBase will erase it after dequeueing.
  // Nonetheless, clear the live-range so that the debug
  // dump will show the right state for that VirtReg.
  LI.clear();
  return false;
}

void RASSA::LRE_WillShrinkVirtReg(Register VirtReg) {
  if (!VRM->hasPhys(VirtReg))
    return;

  // Register is assigned, put it back on the queue for reassignment.
  LiveInterval &LI = LIS->getInterval(VirtReg);
  Matrix->unassign(LI);
  enqueue(&LI);
}

RASSA::RASSA(RegAllocFilterFunc F)
    : MachineFunctionPass(ID), RegAllocBase(F) {}

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
  AU.addRequiredID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.addPreserved<LiveRegMatrixWrapperLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RASSA::releaseMemory() { SpillerInstance.reset(); }

// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RASSA::spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                               SmallVectorImpl<Register> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  SmallVector<const LiveInterval *, 8> Intfs;

  // Collect interferences assigned to any alias of the physical register.
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    for (const auto *Intf : reverse(Q.interferingVRegs())) {
      if (!Intf->isSpillable() || Intf->weight() > VirtReg.weight())
        return false;
      Intfs.push_back(Intf);
    }
  }
  LLVM_DEBUG(dbgs() << "spilling " << printReg(PhysReg, TRI)
                    << " interferences with " << VirtReg << "\n");
  assert(!Intfs.empty() && "expected interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  for (const LiveInterval *Spill : Intfs) {
    // Skip duplicates.
    if (!VRM->hasPhys(Spill->reg()))
      continue;

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(*Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }
  return true;
}

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
//
// This is a minimal implementation of register assignment and splitting that
// spills whenever we run out of registers.
MCRegister RASSA::selectOrSplit(const LiveInterval &VirtReg,
                                SmallVectorImpl<Register> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<MCRegister, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  auto Order =
      AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  for (MCRegister PhysReg : Order) {
    assert(PhysReg.isValid());
    // Check for interference in PhysReg
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      // PhysReg is available, allocate it.
      return PhysReg;

    case LiveRegMatrix::IK_VirtReg:
      // Only virtual registers in the way, we may be able to spill them.
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    default:
      // RegMask or RegUnit interference.
      continue;
    }
  }

  // Try to spill another interfering reg with less spill weight.
  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (!spillInterferences(VirtReg, PhysReg, SplitVRegs))
      continue;

    assert(!Matrix->checkInterference(VirtReg, PhysReg) &&
           "Interference after spill.");
    // Tell the caller to allocate to this newly freed physical register.
    return PhysReg;
  }

  // No other spill candidates were found, so spill the current VirtReg.
  LLVM_DEBUG(dbgs() << "spilling: " << VirtReg << '\n');
  if (!VirtReg.isSpillable())
    return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** SSA REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');

  MF = &mf;
  auto &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  auto &LiveStks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI(); 

  RegAllocBase::init(getAnalysis<VirtRegMapWrapperLegacy>().getVRM(),
                     getAnalysis<LiveIntervalsWrapperPass>().getLIS(),
                     getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM());

  // 1. Create VRAI (Needed by Spiller), but DO NOT RUN IT.
  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, MLI, MBFI,
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
  
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SpillWeightCalculator FSW(MRI, MLI);
  
  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    // Only process if the register is in use and has a LiveInterval
    if (MRI.reg_nodbg_empty(Reg) || !LIS->hasInterval(Reg))
      continue;

    LiveInterval &LI = LIS->getInterval(Reg);
    
    // Set the weight using the Fernando calculator
    LI.setWeight((float)FSW.getWeight(Reg));
  }

  // pass VRAI because the interface requires it.
  SpillerInstance.reset(
      createInlineSpiller({*LIS, LiveStks, MDT, MBFI}, *MF, *VRM, VRAI));

  allocatePhysRegs();
  postOptimization();

  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");

  releaseMemory();
  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }

FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) {
  return new RASSA(F);
}