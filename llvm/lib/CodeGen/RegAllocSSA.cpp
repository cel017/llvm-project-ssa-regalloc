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
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/MachineInstrBuilder.h" // Needed for BuildMI
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
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

// Helper: Spill Weight Calculator
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

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties();
  }

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
  AU.addRequired<PhiAnalysis>(); 
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
  
  // Hint Logic
  auto &PA = getAnalysis<PhiAnalysis>();
  // (Optional: Use PA.getClass(VirtReg.reg()) to find preference)

  std::pair<Register, Register> Hint = MRI->getRegAllocationHint(VirtReg.reg());
  if (Hint.second.isPhysical()) {
      MCRegister PhysHint = Hint.second;
      if (Matrix->checkInterference(VirtReg, PhysHint) == LiveRegMatrix::IK_Free) {
          return PhysHint;
      }
  }

  // Standard Allocation
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

  // Spilling
  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (!spillInterferences(VirtReg, PhysReg, SplitVRegs)) continue;
    return PhysReg;
  }

  // SAFETY CHECK: Fatal error if we can't spill
  if (!VirtReg.isSpillable()) {
     report_fatal_error("RegAllocSSA: Unable to allocate or spill register " + 
                       Twine(VirtReg.reg().id()));
  }

  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);
  return 0; // Return 0 = Handled by Spilling
}

//===----------------------------------------------------------------------===//
// Helper: Isolate PHIs to prevent Spiller crashes
//===----------------------------------------------------------------------===//
void isolatePhis(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  for (MachineBasicBlock &MBB : MF) {
    // 1. Find the safe insertion point (after PHIs)
    MachineBasicBlock::iterator InsertPos = MBB.getFirstNonPHI();

    // 2. Iterate over PHIs
    for (auto I = MBB.begin(); I != InsertPos; ++I) {
      MachineInstr &PhiMI = *I;
      Register PhiDef = PhiMI.getOperand(0).getReg();

      if (!PhiDef.isVirtual()) continue;

      // 3. Create a COPY: %New = COPY %PhiDef
      const TargetRegisterClass *RC = MRI.getRegClass(PhiDef);
      Register NewReg = MRI.createVirtualRegister(RC);

      BuildMI(MBB, InsertPos, DebugLoc(), 
              MF.getSubtarget().getInstrInfo()->get(TargetOpcode::COPY), NewReg)
          .addReg(PhiDef);

      // 4. Replace all downstream uses with NewReg
      for (MachineOperand &UseMO : llvm::make_early_inc_range(MRI.use_operands(PhiDef))) {
        MachineInstr *UseMI = UseMO.getParent();
        // Skip the COPY we just made
        if (UseMI->getParent() == &MBB && UseMI->isCopy() && 
            UseMI->getOperand(0).getReg() == NewReg) continue;
        
        UseMO.setReg(NewReg);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pre-Order Traversal Allocation
//===----------------------------------------------------------------------===//
void RASSA::allocatePhysRegs() {
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  for (auto *Node : depth_first(MDT.getRootNode())) {
    MachineBasicBlock *MBB = Node->getBlock();
    if (!MBB) continue;

    // Collect VRegs first to avoid iterator invalidation
    SmallVector<Register, 16> VRegsToAlloc;

    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr()) continue;
      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg()) continue;
        Register Reg = MO.getReg();
        if (Reg.isVirtual() && !VRM->hasPhys(Reg)) {
           VRegsToAlloc.push_back(Reg);
        }
      }
    }

    // Allocate
    for (Register Reg : VRegsToAlloc) {
      if (VRM->hasPhys(Reg)) continue; 
      if (!LIS->hasInterval(Reg)) continue; 

      LiveInterval &LI = LIS->getInterval(Reg);
      
      SmallVector<Register, 4> SplitVRegs;
      MCRegister PhysReg = selectOrSplit(LI, SplitVRegs);
      
      if (PhysReg) {
        Matrix->assign(LI, PhysReg);
      }
      // Note: We do NOT need normalizeMBB here anymore because isolatePhis 
      // ensures any spills happen at the COPY, which is safely after the PHIs.
    }
  }
}

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** SSA REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');

  MF = &mf;

  // >>> CRITICAL FIX: Isolate PHIs before anything else <<<
  isolatePhis(*MF);

  auto &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  auto &LiveStks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI(); 
  
  RegAllocBase::init(getAnalysis<VirtRegMapWrapperLegacy>().getVRM(),
                     getAnalysis<LiveIntervalsWrapperPass>().getLIS(),
                     getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM());

  MachineRegisterInfo &MRI = MF->getRegInfo();
  SpillWeightCalculator FSW(MRI, MLI);
  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI.reg_nodbg_empty(Reg) || !LIS->hasInterval(Reg))
      continue;
    LiveInterval &LI = LIS->getInterval(Reg);
    LI.setWeight((float)FSW.getWeight(Reg));
  }

  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, MLI, MBFI,
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
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