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
#include <map>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace {
// --- Spill Weight Calculator ---
class SpillWeightCalculator {
  const MachineRegisterInfo &MRI;
  const MachineLoopInfo &MLI;
  static constexpr unsigned Pow10[] = {1, 10, 100, 1000, 10000, 100000, 1000000};

  unsigned getLoopWeight(const MachineBasicBlock *MBB) const {
    unsigned Depth = MLI.getLoopDepth(MBB);
    return Pow10[std::min(Depth, (unsigned)6)];
  }

public:
  SpillWeightCalculator(const MachineRegisterInfo &mri, const MachineLoopInfo &mli)
      : MRI(mri), MLI(mli) {}

  unsigned getWeight(Register Reg) const {
    if (!Reg.isVirtual()) return 0;
    unsigned W = 0;
    for (MachineInstr &MI : MRI.reg_nodbg_instructions(Reg)) {
      W += 1 + getLoopWeight(MI.getParent());
    }
    return W;
  }
};

// --- SSA Register Allocator ---
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

  std::map<MCPhysReg, Register> PhysRegState;

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
  PhysRegState.clear();

  // Single-pass greedy allocation via Dominator Tree
  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock(), SWC);
  }

  // Safety: Ensure no VReg is left unmapped (prevents Rewriter crash)
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register VReg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(VReg)) continue;
    if (!VRM->hasPhys(VReg) && VRM->getStackSlot(VReg) == VirtRegMap::NO_STACK_SLOT) {
      VRM->assignVirt2StackSlot(VReg);
    }
  }

  return true;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB, const SpillWeightCalculator &SWC) {
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugInstr()) continue;

    SlotIndex CurrIdx = LIS->getInstructionIndex(MI).getRegSlot();

    // 1. Liberation: Free registers whose live intervals expire
    for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ) {
      Register VReg = it->second;
      if (LIS->hasInterval(VReg) && LIS->getInterval(VReg).expiredAt(CurrIdx))
        it = PhysRegState.erase(it);
      else
        ++it;
    }

    // 2. Allocation: Map defs
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
        Register VReg = MO.getReg();
        
        // Skip if already assigned (e.g. from Pass 1 or function start)
        if (VRM->hasPhys(VReg) || VRM->getStackSlot(VReg) != VirtRegMap::NO_STACK_SLOT) {
          if (VRM->hasPhys(VReg)) PhysRegState[VRM->getPhys(VReg)] = VReg;
          continue;
        }

        if (MCPhysReg PReg = selectOrSpill(VReg, SWC)) {
          VRM->assignVirt2Phys(VReg, PReg);
          PhysRegState[PReg] = VReg;
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

  // First Attempt: Find a free register
  for (MCPhysReg PReg : Order) {
    bool Busy = false;
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) { Busy = true; break; }
    }
    if (Busy) continue;

    if (Matrix->checkInterference(LI, PReg) == LiveRegMatrix::IK_Free)
      return PReg;
  }

  // Second Attempt: Spill the occupant with the lowest weight
  Register BestToSpill;
  unsigned MinWeight = ~0U;
  MCPhysReg BestPReg = 0;

  for (MCPhysReg PReg : Order) {
    auto it = PhysRegState.find(PReg);
    if (it != PhysRegState.end() && it->second.isVirtual()) {
      unsigned W = SWC.getWeight(it->second);
      if (W < MinWeight) {
        MinWeight = W;
        BestToSpill = it->second;
        BestPReg = PReg;
      }
    }
  }

  // If the occupant's weight is lower than the current VReg, evict it
  if (BestToSpill && MinWeight < SWC.getWeight(VReg)) {
    LLVM_DEBUG(dbgs() << "Evicting " << printReg(BestToSpill, TRI) << " for " << printReg(VReg, TRI) << "\n");
    VRM->assignVirt2StackSlot(BestToSpill);
    PhysRegState.erase(BestPReg);
    return BestPReg;
  }

  return 0; // Spill current VReg
}

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
  []() -> FunctionPass* { return new RASSA(); });

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }