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
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  RegClassInfo.runOnMachineFunction(*MF);

  PhysRegState.clear();

  // --- PASS 1: GLOBAL ASSIGNMENT (Silences the Rewriter) ---
  // Every VReg must be mapped to either a PhysReg or a Stack Slot.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register VReg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(VReg)) continue;

    // If it's not already mapped, try to give it a reg or force a spill.
    if (!VRM->hasPhys(VReg) && VRM->getStackSlot(VReg) == VirtRegMap::NO_STACK_SLOT) {
      if (MCPhysReg PReg = selectPhysReg(VReg)) {
        VRM->assignVirt2Phys(VReg, PReg);
      } else {
        VRM->assignVirt2StackSlot(VReg);
      }
    }
  }

  // --- PASS 2: CHORDAL LOCAL ALLOCATION ---
  // Now refine PhysRegState (your active pool) for each block.
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

    // Process ALL operands, not just the first def
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef()) continue;
      Register Reg = MO.getReg();
      if (!Reg.isVirtual()) continue;
      
      // If already assigned (e.g., from a previous block/phi), skip
      if (VRM->hasPhys(Reg) || VRM->getStackSlot(Reg) != VirtRegMap::NO_STACK_SLOT)
        continue;

      if (MCPhysReg PReg = selectPhysReg(Reg)) {
        VRM->assignVirt2Phys(Reg, PReg);
        PhysRegState[PReg] = Reg;
      } else {
        spill(Reg);
      }
    }
  }
}

MCPhysReg RASSA::selectPhysReg(Register VReg) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(RC);

  for (MCPhysReg PReg : Order) {
    bool IsAliasBusy = false;
    
    // A physical register is busy if IT or ANY of its aliases are in use
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) {
        IsAliasBusy = true;
        break;
      }
    }
    
    if (!IsAliasBusy) return PReg;
  }
  return 0; // Out of registers
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