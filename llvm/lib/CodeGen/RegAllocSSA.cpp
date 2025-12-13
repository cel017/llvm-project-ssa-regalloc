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

//===----------------------------------------------------------------------===//
// Spill Weight Calculator
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

void cleanupBlockLayout(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty()) continue;

    // We want to find the boundary where PHIs *should* end.
    // However, if the block is messed up, getFirstNonPHI might stop early.
    // Instead, we manually scan for the *last* PHI.
    
    MachineBasicBlock::iterator LastPHI = MBB.end();
    bool FoundPHI = false;
    
    // Find the last PHI in the block
    for (auto I = MBB.begin(); I != MBB.end(); ++I) {
      if (I->isPHI()) {
        LastPHI = I;
        FoundPHI = true;
      }
    }

    if (!FoundPHI) continue; // No PHIs, no problem.

    // The valid insertion point is immediately after the Last PHI.
    MachineBasicBlock::iterator InsertPos = std::next(LastPHI);

    // Now scan from the top of the block up to LastPHI.
    // Any instruction that is NOT a PHI is in the wrong place.
    auto I = MBB.begin();
    while (I != InsertPos) {
      MachineInstr &MI = *I;
      auto Next = std::next(I);

      if (!MI.isPHI()) {
        // Found a stray instruction (e.g., a Reload). 
        // Move it to InsertPos (after all PHIs).
        MBB.splice(InsertPos, &MBB, I);
      }
      
      I = Next;
    }
  }
}

struct CompSpillWeight {
  bool operator()(const LiveInterval *A, const LiveInterval *B) const {
    return A->weight() < B->weight();
  }
};

//===----------------------------------------------------------------------===//
// RASSA Class
//===----------------------------------------------------------------------===//
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
  
  AU.addRequired<SlotIndexesWrapperPass>(); 
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
  
  // Hint Logic (Optional)
  // auto &PA = getAnalysis<PhiAnalysis>();

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

  // SAFETY CHECK
  if (!VirtReg.isSpillable()) {
     report_fatal_error("RegAllocSSA: Unable to allocate or spill register " + 
                       Twine(VirtReg.reg().id()));
  }

  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);
  return 0; 
}

//===----------------------------------------------------------------------===//
// Helper: Isolate PHIs
//===----------------------------------------------------------------------===//
void isolatePhis(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM, SlotIndexes &SI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  for (MachineBasicBlock &MBB : MF) {
    // 1. Collect PHIs
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

      // 2. Create the new register
      const TargetRegisterClass *RC = MRI.getRegClass(PhiDef);
      Register NewReg = MRI.createVirtualRegister(RC);
      
      VRM.grow(); 

      // 3. Create the COPY instruction
      MachineInstr *CopyMI = BuildMI(MBB, InsertPos, DebugLoc(), 
                                     TII->get(TargetOpcode::COPY), NewReg)
                                     .addReg(PhiDef);

      SI.insertMachineInstrInMaps(*CopyMI);

      // 4. Replace downstream uses
      for (MachineOperand &UseMO : llvm::make_early_inc_range(MRI.use_operands(PhiDef))) {
        MachineInstr *UseMI = UseMO.getParent();
        if (UseMI == CopyMI) continue;
        UseMO.setReg(NewReg);
      }

      // 5. Update Liveness for BOTH registers
      LIS.createAndComputeVirtRegInterval(NewReg);

      // CRITICAL: The old PHI register's live interval is now stale (it used to reach the end of the block).
      // We must recompute it, otherwise it still "interferes" with everything, causing more spills.
      LIS.removeInterval(PhiDef);
      LIS.createAndComputeVirtRegInterval(PhiDef);
    }
  }
}

//===----------------------------------------------------------------------===//
// Pre-Order Traversal
//===----------------------------------------------------------------------===//
void RASSA::allocatePhysRegs() {
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  for (auto *Node : depth_first(MDT.getRootNode())) {
    MachineBasicBlock *MBB = Node->getBlock();
    if (!MBB) continue;

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

    for (Register Reg : VRegsToAlloc) {
      if (VRM->hasPhys(Reg)) continue; 
      if (!LIS->hasInterval(Reg)) continue; 

      LiveInterval &LI = LIS->getInterval(Reg);
      
      SmallVector<Register, 4> SplitVRegs;
      MCRegister PhysReg = selectOrSplit(LI, SplitVRegs);
      
      if (PhysReg) {
        Matrix->assign(LI, PhysReg);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Main Driver
//===----------------------------------------------------------------------===//
bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** SSA REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');

  MF = &mf;

  // 1. Initialize RegClassInfo (Crucial for AllocationOrder)
  RegClassInfo.runOnMachineFunction(*MF);

  // 2. Fetch Analyses
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &SI = getAnalysis<SlotIndexesWrapperPass>().getSI();

  // 3. Isolate PHIs (Fixes "Spilling PHI Def" crashes)
  isolatePhis(*MF, LIS, VRM, SI);

  auto &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  auto &LiveStks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI(); 
  
  // 4. Initialize Allocator Base
  RegAllocBase::init(VRM, LIS, getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM());
  
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SpillWeightCalculator FSW(MRI, MLI);

  // 5. Calculate Weights & Apply STRICT Safety Constraints
  // We must identify "Danger Blocks" (blocks with PHIs) and ensure no 
  // live-in registers are spilled there.
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
    
    // Default Weight
    LI.setWeight((float)FSW.getWeight(Reg));

    // CONSTRAINT A: Never spill a register defined by a PHI.
    // (We handled this with isolatePhis, but this is a double-check).
    MachineInstr *DefMI = MRI.getVRegDef(Reg);
    if (DefMI && DefMI->isPHI()) {
      LI.markNotSpillable(); 
      continue;
    }

    // CONSTRAINT B: Never spill a register that is Live-In to a PHI Block.
    // The Spiller would insert a reload at the top (before PHIs), which is illegal.
    bool IsUnsafe = false;
    for (MachineBasicBlock *MBB : PhiBlocks) {
      if (LIS.isLiveInToMBB(LI, MBB)) {
        IsUnsafe = true;
        break;
      }
    }

    if (IsUnsafe) {
       LI.markNotSpillable();
    }
  }

  VirtRegAuxInfo VRAI(*MF, LIS, VRM, MLI, MBFI,
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
  
  SpillerInstance.reset(
      createInlineSpiller({LIS, LiveStks, MDT, MBFI}, *MF, VRM, VRAI));

  allocatePhysRegs();
  
  postOptimization();
  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << VRM << "\n");
  releaseMemory();
  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }

FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) {
  return new RASSA(F);
}