#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace {
  // SpillWeightCalculator implementation
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
    MachineFunction *MF;
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    VirtRegMap *VRM;
    LiveIntervals *LIS;
    MachineLoopInfo *MLI;
    MachineDominatorTree *MDT;
    RegisterClassInfo RegClassInfo;
    std::map<MCPhysReg, Register> PhysRegState;

  public:
    static char ID;
    RASSA() : MachineFunctionPass(ID) {}

    StringRef getPassName() const override { return "SSA Register Allocator"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<LiveIntervals>();
      AU.addRequired<SlotIndexes>();
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addRequired<VirtRegMap>();
      AU.addPreserved<LiveIntervals>();
      AU.addPreserved<SlotIndexes>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &mf) override;

  private:
    void allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC);
    MCPhysReg selectPhysReg(Register VReg, SpillWeightCalculator &SWC);
    void spill(Register VReg);
  };
} // end anonymous namespace

char RASSA::ID = 0;

// IMPORTANT: This block must be outside the anonymous namespace and 
// inside the llvm namespace for the macros to work in some build configs.
namespace llvm {
  void initializeRASSAPass(PassRegistry &);
}

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  VRM = &getAnalysis<VirtRegMap>();
  LIS = &getAnalysis<LiveIntervals>();
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  RegClassInfo.runOnMachineFunction(*MF);

  SpillWeightCalculator SWC(*MRI, *MLI);
  PhysRegState.clear();

  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock(), SWC);
  }
  return true;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC) {
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

    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
        Register VReg = MO.getReg();
        if (MCPhysReg PReg = selectPhysReg(VReg, SWC)) {
          VRM->assignVirt2Phys(VReg, PReg);
          PhysRegState[PReg] = VReg;
        } else {
          spill(VReg);
        }
      }
    }
  }
}

MCPhysReg RASSA::selectPhysReg(Register VReg, SpillWeightCalculator &SWC) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  for (MCPhysReg PReg : RegClassInfo.getOrder(RC)) {
    bool Busy = false;
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) { Busy = true; break; }
    }
    if (!Busy) return PReg;
  }

  Register BestToSpill;
  unsigned MinWeight = ~0U;
  MCPhysReg BestPReg = 0;

  for (MCPhysReg PReg : RegClassInfo.getOrder(RC)) {
    if (PhysRegState.count(PReg) && PhysRegState[PReg].isVirtual()) {
      unsigned W = SWC.getWeight(PhysRegState[PReg]);
      if (W < MinWeight) {
        MinWeight = W;
        BestToSpill = PhysRegState[PReg];
        BestPReg = PReg;
      }
    }
  }

  if (BestToSpill) {
    spill(BestToSpill);
    return BestPReg;
  }
  return 0;
}

void RASSA::spill(Register VReg) {
  if (VRM->getStackSlot(VReg) == VirtRegMap::NO_STACK_SLOT)
    VRM->assignVirt2StackSlot(VReg);
  
  for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ++it) {
    if (it->second == VReg) {
      PhysRegState.erase(it);
      break;
    }
  }
}

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
                                    []() -> FunctionPass* { return new RASSA(); });