#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace {
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

class RASSA : public MachineFunctionPass {
  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  RegisterClassInfo RegClassInfo;

  // Tracking physical register availability
  // Maps PhysReg -> VirtualReg currently occupying it
  std::map<MCPhysReg, Register> PhysRegState;

public:
  static char ID;
  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Register Allocator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<LiveIntervals>();
    AU.addRequired<SlotIndexes>();
    AU.addPreserved<SlotIndexes>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<VirtRegMap>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &mf) override;

private:
  void allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC);
  MCPhysReg selectPhysReg(Register VReg, SpillWeightCalculator &SWC, MachineInstr &MI);
  void spill(Register VReg);
};
} // end anonymous namespace

char RASSA::ID = 0;

bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  TII = MF->getSubtarget().getInstrInfo();
  VRM = &getAnalysis<VirtRegMap>();
  LIS = &getAnalysis<LiveIntervals>();
  MLI = &getAnalysis<MachineLoopInfo>();
  MDT = &getAnalysis<MachineDominatorTree>();
  RegClassInfo.runOnMachineFunction(*MF);

  SpillWeightCalculator SWC(*MRI, *MLI);

  // Traverse the Dominator Tree in Pre-order.
  // This satisfies the Chordal Graph coloring property for SSA.
  for (auto *Node : depth_first(MDT->getRootNode())) {
    allocateBlock(Node->getBlock(), SWC);
  }

  return true;
}

void RASSA::allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC) {
  for (MachineInstr &MI : *MBB) {
    if (MI.isDebugInstr()) continue;

    // 1. Liberate registers that are no longer live after this instruction
    // In SSA, we check if the current slot is the end of the live range.
    SlotIndex CurrIdx = LIS->getInstructionIndex(MI).getRegSlot();
    
    for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ) {
      Register VReg = it->second;
      if (VReg.isVirtual() && LIS->hasInterval(VReg)) {
        if (LIS->getInterval(VReg).expiredAt(CurrIdx)) {
          it = PhysRegState.erase(it);
          continue;
        }
      }
      ++it;
    }

    // 2. Allocate Definitions
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
        Register VReg = MO.getReg();
        MCPhysReg PReg = selectPhysReg(VReg, SWC, MI);
        
        if (PReg) {
          VRM->assignVirt2Phys(VReg, PReg);
          PhysRegState[PReg] = VReg;
        } else {
          spill(VReg);
        }
      }
    }
  }
}

MCPhysReg RASSA::selectPhysReg(Register VReg, SpillWeightCalculator &SWC, MachineInstr &MI) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  ArrayRef<MCPhysReg> AllocationOrder = RegClassInfo.getOrder(RC);

  // Strategy: Find a free physical register in the allocation order
  for (MCPhysReg PReg : AllocationOrder) {
    bool Busy = false;
    // Check if PReg or any aliases are occupied
    for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
      if (PhysRegState.count(*Alias)) {
        Busy = true;
        break;
      }
    }
    if (!Busy) return PReg;
  }

  // No free registers: Implement simple "Spill Lowest Weight" heuristic
  Register BestToSpill;
  unsigned MinWeight = ~0U;
  MCPhysReg BestPReg = 0;

  for (MCPhysReg PReg : AllocationOrder) {
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

  if (BestToSpill) {
    spill(BestToSpill);
    return BestPReg;
  }

  return 0; // Should trigger catastrophic spill
}

void RASSA::spill(Register VReg) {
  if (VRM->hasStackSlot(VReg)) return;
  
  // Assign a stack slot via VirtRegMap
  int Slot = VRM->assignVirt2StackSlot(VReg);
  
  // Clear from physical tracking
  for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ++it) {
    if (it->second == VReg) {
      PhysRegState.erase(it);
      break;
    }
  }
  
  LLVM_DEBUG(dbgs() << "Spilling " << printReg(VReg, TRI) << " to slot " << Slot << "\n");
}

// Pass Registration
static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
                                    []() -> FunctionPass* { return new RASSA(); });

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)