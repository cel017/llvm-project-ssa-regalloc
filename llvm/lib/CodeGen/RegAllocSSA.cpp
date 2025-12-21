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

// Move the class out of the anonymous namespace if the error persists.
// Some LLVM build configurations struggle with template instantiation 
// of classes inside anonymous namespaces when using INITIALIZE_PASS macros.
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
  RASSA() : MachineFunctionPass(ID) {
    // Explicitly call the initialization function
    initializeRASSAPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // Use the WrapperPass types explicitly
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
  void allocateBlock(MachineBasicBlock *MBB);
  MCPhysReg selectPhysReg(Register VReg);
  void spill(Register VReg);
};

// Define ID in the global/llvm scope
char RASSA::ID = 0;

// This macro MUST be in the llvm namespace or global scope
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
  
  // Extract the actual analysis classes from the Wrapper Passes
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  
  RegClassInfo.runOnMachineFunction(*MF);
  PhysRegState.clear();

  // The Chordal property: process blocks in Dominator Tree Pre-order
  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock());
  }

  return true;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB) {
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugInstr()) continue;

    SlotIndex CurrIdx = LIS->getInstructionIndex(MI).getRegSlot();

    // 1. Expire live ranges
    for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ) {
      Register VReg = it->second;
      if (LIS->hasInterval(VReg) && LIS->getInterval(VReg).expiredAt(CurrIdx))
        it = PhysRegState.erase(it);
      else
        ++it;
    }

    // 2. Assign defs
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
        Register VReg = MO.getReg();
        if (MCPhysReg PReg = selectPhysReg(VReg)) {
          VRM->assignVirt2Phys(VReg, PReg);
          PhysRegState[PReg] = VReg;
        } else {
          spill(VReg);
        }
      }
    }
  }
}

MCPhysReg RASSA::selectPhysReg(Register VReg) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(RC);

  for (MCPhysReg PReg : Order) {
    bool Busy = false;
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) { Busy = true; break; }
    }
    if (!Busy) return PReg;
  }
  return 0; // Simplified for initial compile - will need your weight logic back
}

void RASSA::spill(Register VReg) {
  if (VRM->getStackSlot(VReg) == VirtRegMap::NO_STACK_SLOT)
    VRM->assignVirt2StackSlot(VReg);
}

// Global factory function
FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }

// LLC Command line registration
static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
                                    []() -> FunctionPass* { return new RASSA(); });