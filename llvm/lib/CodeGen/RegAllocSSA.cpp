//===-- RegAllocSSA.cpp - SSA Register Allocator --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegAllocBase.h"
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
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
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
#include <queue>

#include "llvm/CodeGen/PhiAnalysis.h" 
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

FunctionPass *llvm::createSSARegisterAllocator();
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F);

namespace {

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
    if (DefMI) {
      if (DefMI->isPHI()) {
        for (unsigned i = 1, e = DefMI->getNumOperands(); i < e; i += 2) {
          MachineBasicBlock *IncomingMBB = DefMI->getOperand(i + 1).getMBB();
          W += 1 + getLoopWeight(IncomingMBB);
        }
      } else {
        W += 1 + getLoopWeight(DefMI->getParent());
      }
    }
    for (MachineInstr &UseMI : MRI.reg_nodbg_instructions(Reg)) {
      if (&UseMI == DefMI) continue;
      if (UseMI.isPHI()) {
        for (unsigned i = 1, e = UseMI.getNumOperands(); i < e; i += 2) {
          if (UseMI.getOperand(i).isReg() && UseMI.getOperand(i).getReg() == Reg) {
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

class RASSA : public MachineFunctionPass,
              public RegAllocBase,
              private LiveRangeEdit::Delegate {
  
  MachineFunction *MF = nullptr;
  std::unique_ptr<Spiller> SpillerInstance;
  std::priority_queue<const LiveInterval *, std::vector<const LiveInterval *>,
                      CompSpillWeight> Queue;

  bool LRE_CanEraseVirtReg(Register) override;
  void LRE_WillShrinkVirtReg(Register) override;

public:
  static char ID;

  RASSA(const RegAllocFilterFunc F = nullptr);

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }

  // NOTE: We do NOT implement getClearedProperties(). 
  // We let the VirtRegRewriter (which runs after us) handle the SSA->NoSSA transition.

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;

  Spiller &spiller() override { return *SpillerInstance; }

  void enqueueImpl(const LiveInterval *LI) override { Queue.push(LI); }

  const LiveInterval *dequeue() override {
    if (Queue.empty()) return nullptr;
    const LiveInterval *LI = Queue.top();
    Queue.pop();
    return LI;
  }

  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override;

  void allocatePhysRegs();

  bool runOnMachineFunction(MachineFunction &mf) override;

  bool spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                          SmallVectorImpl<Register> &SplitVRegs);
};

char RASSA::ID = 0;

} // end anonymous namespace

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                    createSSARegisterAllocator);

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
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
INITIALIZE_PASS_DEPENDENCY(PhiAnalysis) 
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

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
  if (!VRM->hasPhys(VirtReg)) return;
  LiveInterval &LI = LIS->getInterval(VirtReg);
  Matrix->unassign(LI);
  enqueue(&LI);
}

RASSA::RASSA(RegAllocFilterFunc F) : MachineFunctionPass(ID), RegAllocBase(F) {}

// >>> FIX: Analysis Usage Updates <<<
void RASSA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  
  // Required Analyses
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>(); 
  AU.addRequired<LiveDebugVariablesWrapperLegacy>();
  AU.addRequired<LiveStacksWrapperLegacy>();
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.addRequired<PhiAnalysis>(); 

  // Preserved Analyses
  // We ONLY preserve VirtRegMap (because we fill it for the Rewriter)
  // We do NOT preserve LiveIntervals or Matrix, because allocation invalidates them.
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addPreserved<SlotIndexesWrapperPass>(); 
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<LiveDebugVariablesWrapperLegacy>();
  AU.addPreserved<LiveStacksWrapperLegacy>();
  AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RASSA::releaseMemory() { SpillerInstance.reset(); }

bool RASSA::spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                               SmallVectorImpl<Register> &SplitVRegs) {
  SmallVector<const LiveInterval *, 8> Intfs;
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    for (const auto *Intf : reverse(Q.interferingVRegs())) {
      if (!Intf->isSpillable() || Intf->weight() > VirtReg.weight())
        return false;
      Intfs.push_back(Intf);
    }
  }
  assert(!Intfs.empty() && "expected interference");
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
  
  std::pair<Register, Register> Hint = MRI->getRegAllocationHint(VirtReg.reg());
  if (Hint.second.isPhysical()) {
      MCRegister PhysHint = Hint.second;
      if (Matrix->checkInterference(VirtReg, PhysHint) == LiveRegMatrix::IK_Free) {
          return PhysHint;
      }
  }

  SmallVector<MCRegister, 8> PhysRegSpillCands;
  auto Order = AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  
  for (MCRegister PhysReg : Order) {
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      return PhysReg;
    case LiveRegMatrix::IK_VirtReg:
      PhysRegSpillCands.push_back(PhysReg);
      continue;
    default:
      continue;
    }
  }

  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (!spillInterferences(VirtReg, PhysReg, SplitVRegs)) continue;
    return PhysReg;
  }

  if (!VirtReg.isSpillable()) {
     report_fatal_error("RegAllocSSA: Unable to allocate or spill register " + 
                       Twine(VirtReg.reg().id()));
  }

  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);
  return 0; 
}

void isolatePhis(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM, SlotIndexes &SI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr*, 8> Phis;
    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI()) break; 
      Phis.push_back(&MI);
    }

    if (Phis.empty()) continue;

    MachineBasicBlock::iterator InsertPos = MBB.getFirstNonPHI();

    for (MachineInstr *PhiMI : Phis) {
      Register PhiDef = PhiMI->getOperand(0).getReg();
      if (!PhiDef.isVirtual()) continue;

      const TargetRegisterClass *RC = MRI.getRegClass(PhiDef);
      Register NewReg = MRI.createVirtualRegister(RC);
      
      VRM.grow(); 

      MachineInstr *CopyMI = BuildMI(MBB, InsertPos, DebugLoc(), 
                                     TII->get(TargetOpcode::COPY), NewReg)
                                     .addReg(PhiDef);

      SI.insertMachineInstrInMaps(*CopyMI);

      for (MachineOperand &UseMO : llvm::make_early_inc_range(MRI.use_operands(PhiDef))) {
        MachineInstr *UseMI = UseMO.getParent();
        if (UseMI == CopyMI) continue;
        UseMO.setReg(NewReg);
      }

      LIS.createAndComputeVirtRegInterval(NewReg);
      LIS.removeInterval(PhiDef);
      LIS.createAndComputeVirtRegInterval(PhiDef);
    }
  }
}

// Unified Allocation Loop
void RASSA::allocatePhysRegs() {
  SmallVector<Register, 64> VRegsToAlloc;

  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr()) continue;
      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg()) continue;
        Register Reg = MO.getReg();
        if (Reg.isVirtual() && !VRM->hasPhys(Reg)) {
           VRegsToAlloc.push_back(Reg);
        }
      }
    }
  }

  for (size_t i = 0; i < VRegsToAlloc.size(); ++i) {
    Register Reg = VRegsToAlloc[i];

    if (VRM->hasPhys(Reg)) continue;

    if (!LIS->hasInterval(Reg)) {
        LIS->createAndComputeVirtRegInterval(Reg);
    }
    
    LiveInterval &LI = LIS->getInterval(Reg);
    if (LI.empty()) LI.setWeight(0.0f);
    
    if (i >= VRegsToAlloc.size()) { 
        LI.markNotSpillable();
    }

    SmallVector<Register, 4> SplitVRegs;
    MCRegister PhysReg = selectOrSplit(LI, SplitVRegs);

    if (PhysReg) {
      Matrix->assign(LI, PhysReg);
    } 
    else if (SplitVRegs.empty()) {
       report_fatal_error("RegAllocSSA: Failed to allocate or spill register " + 
                          Twine(Reg.id()));
    }

    if (!SplitVRegs.empty()) {
      VRM->grow(); 
      for (Register NewReg : SplitVRegs) {
        if (!LIS->hasInterval(NewReg)) {
           LIS->createAndComputeVirtRegInterval(NewReg);
        }
        LIS->getInterval(NewReg).markNotSpillable();
        VRegsToAlloc.push_back(NewReg);
      }
    }
  }
}

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** SSA REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');

  MF = &mf;

  RegClassInfo.runOnMachineFunction(*MF);

  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &SI = getAnalysis<SlotIndexesWrapperPass>().getSI();

  isolatePhis(*MF, LIS, VRM, SI);

  auto &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  auto &LiveStks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI(); 
  
  RegAllocBase::init(VRM, LIS, getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM());
  
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SpillWeightCalculator FSW(MRI, MLI);

  SmallVector<MachineBasicBlock*, 8> PhiBlocks;
  for (MachineBasicBlock &MBB : *MF) {
    if (!MBB.empty() && MBB.front().isPHI()) {
      PhiBlocks.push_back(&MBB);
    }
  }

  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI.reg_nodbg_empty(Reg) || !LIS.hasInterval(Reg))
      continue;
    
    LiveInterval &LI = LIS.getInterval(Reg);
    LI.setWeight((float)FSW.getWeight(Reg));

    MachineInstr *DefMI = MRI.getVRegDef(Reg);
    if (DefMI && DefMI->isPHI()) {
      LI.markNotSpillable(); 
      continue;
    }

    bool IsUnsafe = false;
    for (MachineBasicBlock *MBB : PhiBlocks) {
      if (LIS.isLiveInToMBB(LI, MBB)) {
        IsUnsafe = true;
        break;
      }
    }

    if (IsUnsafe) LI.markNotSpillable();
  }

  VirtRegAuxInfo VRAI(*MF, LIS, VRM, MLI, MBFI,
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
  
  SpillerInstance.reset(
      createInlineSpiller({LIS, LiveStks, MDT, MBFI}, *MF, VRM, VRAI));

  // 1. Run Allocation
  allocatePhysRegs();
  
  // 2. Post-Optimization (Standard LLVM hook)
  postOptimization();

  //  Cleanup Spilled Registers from Live-Ins 
  // The Spiller removes instructions, but stale Virtual Registers might remain 
  // in the MBB.liveins() list. We must remove them to prevent the Rewriter from crashing.
  for (MachineBasicBlock &MBB : *MF) {
    // We cannot modify the list while iterating, so we collect valid ones first.
    SmallVector<MachineBasicBlock::RegisterMaskPair, 8> ValidLiveIns;

    for (const auto &LI : MBB.liveins()) {
      if (LI.PhysReg.isPhysical()) {
        // Keep Physical Registers (added by us or previous passes)
        ValidLiveIns.push_back(LI);
      } else {
        // For Virtual Registers, check if they are allocated.
        // If VRM.getPhys() returns 0 (NoRegister), it means it was Spilled.
        // Spilled registers are dead (replaced by reloads), so we drop them.
        if (VRM.hasPhys(LI.PhysReg)) {
           ValidLiveIns.push_back(LI);
        }
      }
    }

    // Replace the dirty list with the clean one
    MBB.clearLiveIns();
    for (const auto &LI : ValidLiveIns) {
      MBB.addLiveIn(LI.PhysReg, LI.LaneMask);
    }
    MBB.sortUniqueLiveIns();
  }

  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << VRM << "\n");
  releaseMemory();
  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }

FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) {
  return new RASSA(F);
}