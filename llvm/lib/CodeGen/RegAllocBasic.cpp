#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervals.h"
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
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace {
// Your SpillWeightCalculator integrated as a helper
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

class RASSA : public MachineFunctionPass {
  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;
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
    // Use WrapperPasses to match modern LLVM infrastructure
    AU.addRequired<LiveIntervalsWrapperPass>();
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
  void allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC);
  MCPhysReg selectPhysReg(Register VReg, SpillWeightCalculator &SWC);
  void spill(Register VReg);
};
} // end anonymous namespace

char RASSA::ID = 0;

// Initialize using the WrapperPass naming convention
INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  
  // Extract analyses from WrapperPasses
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  
  RegClassInfo.runOnMachineFunction(*MF);
  SpillWeightCalculator SWC(*MRI, *MLI);
  PhysRegState.clear();

  // Perform allocation in Dominator Tree Pre-order
  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock(), SWC);
  }

  return true;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC) {
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugInstr()) continue;

    SlotIndex CurrIdx = LIS->getInstructionIndex(MI).getRegSlot();

    // Check expiration of physical register assignments
    for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ) {
      Register VReg = it->second;
      if (LIS->hasInterval(VReg) && LIS->getInterval(VReg).expiredAt(CurrIdx))
        it = PhysRegState.erase(it);
      else
        ++it;
    }

    // Assign registers to definitions
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
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(RC);

  for (MCPhysReg PReg : Order) {
    bool Busy = false;
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) { Busy = true; break; }
    }
    if (!Busy) return PReg;
  }

  // Heuristic: Spill lowest weight current occupant
  Register BestToSpill;
  unsigned MinWeight = ~0U;
  MCPhysReg BestPReg = 0;

  for (MCPhysReg PReg : Order) {
    if (PhysRegState.count(PReg)) {
      Register Occupant = PhysRegState[PReg];
      if (Occupant.isVirtual()) {
        unsigned W = SWC.getWeight(Occupant);
        if (W < MinWeight) {
          MinWeight = W;
          BestToSpill = Occupant;
          BestPReg = PReg;
        }
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

// Allow invocation via -regalloc=ssa
static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
  []() -> FunctionPass* { return new RASSA(); });