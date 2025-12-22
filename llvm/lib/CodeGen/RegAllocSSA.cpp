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
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

FunctionPass *llvm::createSSARegisterAllocator();
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F);

namespace {

class RASSA : public MachineFunctionPass,
              public RegAllocBase,
              private LiveRangeEdit::Delegate {
  
  MachineFunction *MF = nullptr;
  std::unique_ptr<Spiller> SpillerInstance;

  // Needed for RegAllocBase
  bool LRE_CanEraseVirtReg(Register) override { return false; }
  void LRE_WillShrinkVirtReg(Register) override {}

  // Queue for Allocation
  std::vector<Register> VRegsToAlloc;

public:
  static char ID;

  RASSA(const RegAllocFilterFunc F = nullptr);

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }

  // Clear IsSSA so the verifier and rewriter know we are done with SSA
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;

  Spiller &spiller() override { return *SpillerInstance; }

  // We implement a custom enqueue, so these can be no-ops or simple pushes
  void enqueueImpl(const LiveInterval *LI) override {}
  const LiveInterval *dequeue() override { return nullptr; } // We don't use the RegAllocBase priority queue

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
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

RASSA::RASSA(RegAllocFilterFunc F) : MachineFunctionPass(ID), RegAllocBase(F) {}

void RASSA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
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
  
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>(); // Keep LIS valid for rewriting
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<LiveDebugVariablesWrapperLegacy>();
  AU.addPreserved<LiveStacksWrapperLegacy>();
  AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RASSA::releaseMemory() { 
    SpillerInstance.reset(); 
    VRegsToAlloc.clear();
}

bool RASSA::spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                               SmallVectorImpl<Register> &SplitVRegs) {
  // Simple spilling: if we interfere, we fail. 
  // We rely on the Spiller to handle the actual splitting of the current VirtReg.
  return false; 
}

MCRegister RASSA::selectOrSplit(const LiveInterval &VirtReg,
                                SmallVectorImpl<Register> &SplitVRegs) {
  
  // 1. Check Hints (Essential for Copy $x10)
  std::pair<Register, Register> Hint = MRI->getRegAllocationHint(VirtReg.reg());
  if (Hint.second.isPhysical()) {
      MCRegister PhysHint = Hint.second;
      if (Matrix->checkInterference(VirtReg, PhysHint) == LiveRegMatrix::IK_Free) {
          return PhysHint;
      }
  }

  // 2. Greedy Search
  auto Order = AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  for (MCRegister PhysReg : Order) {
    if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_Free)
      return PhysReg;
  }

  // 3. Fallback: Spill
  // If we can't find a register, we tell the spiller to handle it.
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);
  return 0; // 0 indicates assignment failure (spill)
}

//===----------------------------------------------------------------------===//
// UNIFIED ALLOCATION LOOP (The Fix)
//===----------------------------------------------------------------------===//
void RASSA::allocatePhysRegs() {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  VRegsToAlloc.clear();

  // STEP 1: COLLECT
  // Iterate the MRI directly. This cannot miss %0.
  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI.reg_nodbg_empty(Reg)) continue;
    if (!VRM->hasPhys(Reg)) {
        VRegsToAlloc.push_back(Reg);
    }
  }

  // STEP 2: SORT (Simple Weight)
  std::sort(VRegsToAlloc.begin(), VRegsToAlloc.end(), [&](Register A, Register B) {
     return LIS->getInterval(A).weight() > LIS->getInterval(B).weight();
  });

  // STEP 3: ALLOCATE
  for (size_t i = 0; i < VRegsToAlloc.size(); ++i) {
    Register Reg = VRegsToAlloc[i];
    if (VRM->hasPhys(Reg)) continue; // Already handled (maybe by hint or pre-alloc)

    // Verify Interval
    if (!LIS->hasInterval(Reg)) LIS->createAndComputeVirtRegInterval(Reg);
    LiveInterval &LI = LIS->getInterval(Reg);

    SmallVector<Register, 4> SplitVRegs;
    MCRegister PhysReg = selectOrSplit(LI, SplitVRegs);

    if (PhysReg) {
      Matrix->assign(LI, PhysReg);
    } 
    else if (SplitVRegs.empty()) {
       // Only crash if we didn't assign AND didn't spill.
       report_fatal_error("RegAllocSSA: Failed to allocate or spill register " + 
                          Twine(Reg.id()));
    }

    // Handle Spill Artifacts
    if (!SplitVRegs.empty()) {
      for (Register NewReg : SplitVRegs) {
         VRegsToAlloc.push_back(NewReg); // Process new registers in this same pass
      }
    }
  }
}

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  LLVM_DEBUG(dbgs() << "********** SSA REGISTER ALLOCATION **********\n"
                    << "********** Function: " << mf.getName() << '\n');
  MF = &mf;

  RegClassInfo.runOnMachineFunction(*MF);

  // Initialize Analyses
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  auto &LiveStks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  
  RegAllocBase::init(VRM, LIS, getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM());
  
  MachineRegisterInfo &MRI = MF->getRegInfo();
  
  // Calculate simple spill weights
  calculateSpillWeightsAndHints(LIS, *MF, &VRM, *MLI, MBFI);

  // Initialize Spiller
  VirtRegAuxInfo VRAI(*MF, LIS, VRM, *MLI, MBFI, 
                      &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI());
  SpillerInstance.reset(createInlineSpiller({LIS, LiveStks, MDT, MBFI}, *MF, VRM, VRAI));

  // Run the Robust Allocation
  allocatePhysRegs();
  
  // Cleanup
  postOptimization();
  
  // DO NOT MANIPULATE LIVE-INS MANUALLY.
  // The rewriter will handle it if the map is correct.

  releaseMemory();
  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) { return new RASSA(F); }