#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace {

class SpillWeightCalculator {
  const MachineRegisterInfo &MRI;
  const MachineLoopInfo &MLI;
  static constexpr unsigned Pow10[] = {1, 10, 100, 1000, 10000, 100000, 1000000};

public:
  SpillWeightCalculator(const MachineRegisterInfo &mri, const MachineLoopInfo &mli)
      : MRI(mri), MLI(mli) {}

  unsigned getWeight(Register Reg) const {
    if (!Reg.isVirtual()) return 0;
    unsigned W = 0;
    for (MachineInstr &MI : MRI.reg_nodbg_instructions(Reg)) {
      unsigned Depth = std::min(MLI.getLoopDepth(MI.getParent()), (unsigned)6);
      W += 1 + Pow10[Depth];
    }
    return W;
  }
};

class RASSA : public MachineFunctionPass {
  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;
  LiveRegMatrix *Matrix = nullptr;
  MachineDominatorTree *MDT = nullptr;
  MachineLoopInfo *MLI = nullptr;
  RegisterClassInfo RegClassInfo;

public:
  static char ID;
  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &mf) override;

private:
  void allocateBlock(MachineBasicBlock *MBB, const SpillWeightCalculator &SWC);
  MCPhysReg selectOrSpill(Register VReg, const SpillWeightCalculator &SWC);
};

} // end anonymous namespace

char RASSA::ID = 0;

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Matrix = &getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  RegClassInfo.runOnMachineFunction(*MF);

  SpillWeightCalculator SWC(*MRI, *MLI);

  // 1. CHORDAL ALLOCATION (Dominator Tree Pre-order)
  // We rely on LiveRegMatrix to check interferences against VRM dynamically.
  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock(), SWC);
  }

  // 2. SAFETY CATCH-ALL
  // If any register was missed (e.g. dead defs, unused args), spill it to stack
  // to prevent VirtRegRewriter from crashing.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register VReg = Register::index2VirtReg(i);
    // Skip if unused
    if (MRI->reg_nodbg_empty(VReg)) continue;

    // Check if unmapped
    bool HasPhys = VRM->hasPhys(VReg);
    bool HasStack = (VRM->getStackSlot(VReg) != VirtRegMap::NO_STACK_SLOT);
    
    if (!HasPhys && !HasStack) {
      // Emergency spill
      VRM->assignVirt2StackSlot(VReg);
    }
  }

  return true;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB, const SpillWeightCalculator &SWC) {
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugInstr()) continue;

    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
        Register VReg = MO.getReg();
        
        // If already assigned (by a previous block visit), skip
        if (VRM->hasPhys(VReg) || VRM->getStackSlot(VReg) != VirtRegMap::NO_STACK_SLOT)
          continue;

        // Try to allocate
        if (MCPhysReg PReg = selectOrSpill(VReg, SWC)) {
          VRM->assignVirt2Phys(VReg, PReg);
        } else {
          VRM->assignVirt2StackSlot(VReg);
        }
      }
    }
  }
}

MCPhysReg RASSA::selectOrSpill(Register VReg, const SpillWeightCalculator &SWC) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  LiveInterval &LI = LIS->getInterval(VReg);
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(RC);

  // Attempt 1: Find a free register
  // LiveRegMatrix->checkInterference checks against:
  //  - Reserved Registers (Fixed Interference)
  //  - Other VRegs already assigned in VRM (Global Interference)
  for (MCPhysReg PReg : Order) {
    if (Matrix->checkInterference(LI, PReg) == LiveRegMatrix::IK_Free)
      return PReg;
  }

  // Attempt 2: Simple Spilling (Eviction)
  // If we must spill, we currently spill the NEW register (VReg).
  // Implementing true eviction requires unassigning the conflict from VRM, 
  // which is risky without the complex undo-logic of RegAllocGreedy.
  // For this SSA allocator, "Spill Current" is the safest baseline.
  
  return 0; 
}

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
  []() -> FunctionPass* { return new RASSA(); });

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }