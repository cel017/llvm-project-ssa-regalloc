#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h" // Required for interference checking
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
#include <algorithm>

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
  LiveRegMatrix *Matrix = nullptr; // Added
  MachineLoopInfo *MLI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  RegisterClassInfo RegClassInfo;

  std::map<MCPhysReg, Register> PhysRegState;

public:
  static char ID;
  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>(); // Added
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
  void allocateBlock(MachineBasicBlock *MBB);
  MCPhysReg selectPhysReg(Register VReg);
  void spill(Register VReg);
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
  Matrix = &getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM(); // Added
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  RegClassInfo.runOnMachineFunction(*MF);

  SpillWeightCalculator SWC(*MRI, *MLI);
  PhysRegState.clear();

  // PASS 1: Sorted Global Assignment
  std::vector<Register> VRegs;
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register VReg = Register::index2VirtReg(i);
    if (!MRI->reg_nodbg_empty(VReg)) VRegs.push_back(VReg);
  }

  // Use your SpillWeightCalculator to prioritize high-use registers
  std::sort(VRegs.begin(), VRegs.end(), [&](Register A, Register B) {
    return SWC.getWeight(A) > SWC.getWeight(B);
  });

  for (Register VReg : VRegs) {
    if (!VRM->hasPhys(VReg) && VRM->getStackSlot(VReg) == VirtRegMap::NO_STACK_SLOT) {
      if (MCPhysReg PReg = selectPhysReg(VReg)) {
        VRM->assignVirt2Phys(VReg, PReg);
      } else {
        VRM->assignVirt2StackSlot(VReg);
      }
    }
  }

  // PASS 2: Local Refinement (Dominator Tree Pre-order)
  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock());
  }

  return true;
}

MCPhysReg RASSA::selectPhysReg(Register VReg) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  LiveInterval &LI = LIS->getInterval(VReg);

  for (MCPhysReg PReg : RegClassInfo.getOrder(RC)) {
    // Check interference with other virtual registers currently in PhysRegState
    bool Busy = false;
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) {
        Busy = true;
        break;
      }
    }
    if (Busy) continue;

    // Check interference with Physical Registers/Pre-colors (IG_Builder_Fer logic)
    // LiveRegMatrix handles checking all regunits of PReg against the interval LI.
    if (Matrix->checkInterference(LI, PReg) != LiveRegMatrix::IK_Free)
      continue;

    return PReg;
  }
  return 0;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB) {
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugInstr()) continue;
    SlotIndex CurrIdx = LIS->getInstructionIndex(MI).getRegSlot();

    for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ) {
      Register VReg = it->second;
      if (LIS->hasInterval(VReg) && LIS->getInterval(VReg).expiredAt(CurrIdx))
        it = PhysRegState.erase(it);
      else
        ++it;
    }

    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
        Register VReg = MO.getReg();
        if (VRM->hasPhys(VReg))
          PhysRegState[VRM->getPhys(VReg)] = VReg;
      }
    }
  }
}

void RASSA::spill(Register VReg) {
  if (VRM->getStackSlot(VReg) == VirtRegMap::NO_STACK_SLOT)
    VRM->assignVirt2StackSlot(VReg);
}

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
  []() -> FunctionPass* { return new RASSA(); });

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }