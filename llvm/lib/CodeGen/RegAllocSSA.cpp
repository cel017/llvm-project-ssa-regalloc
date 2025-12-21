#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/MachineDominators.h"  // Definition of MachineDominatorTree
#include "llvm/CodeGen/MachineLoopInfo.h"       // Definition of MachineLoopInfo
#include "llvm/CodeGen/LiveIntervals.h"         // Definition of LiveIntervals
#include "llvm/CodeGen/SlotIndexes.h"           // Definition of SlotIndexes
#include "llvm/CodeGen/VirtRegMap.h"            // Definition of VirtRegMap
#include "llvm/CodeGen/Passes.h"                // Declarations for initialize...Pass
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DepthFirstIterator.h"        // Definition of depth_first

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

  bool runOnMachineFunction(MachineFunction &mf) override {
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

    // Dominator Tree Pre-order traversal
    for (auto *Node : depth_first(MDT->getRootNode())) {
      allocateBlock(Node->getBlock(), SWC);
    }
    return true;
  }

private:
  void allocateBlock(MachineBasicBlock *MBB, SpillWeightCalculator &SWC) {
    for (MachineInstr &MI : *MBB) {
      if (MI.isDebugInstr()) continue;

      SlotIndex CurrIdx = LIS->getInstructionIndex(MI).getRegSlot();
      
      // Cleanup expired registers
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

  MCPhysReg selectPhysReg(Register VReg, SpillWeightCalculator &SWC, MachineInstr &MI) {
    const TargetRegisterClass *RC = MRI->getRegClass(VReg);
    ArrayRef<MCPhysReg> AllocationOrder = RegClassInfo.getOrder(RC);

    for (MCPhysReg PReg : AllocationOrder) {
      bool Busy = false;
      for (MCRegAliasIterator Alias(PReg, TRI, true); Alias.isValid(); ++Alias) {
        if (PhysRegState.count(*Alias)) {
          Busy = true;
          break;
        }
      }
      if (!Busy) return PReg;
    }

    Register BestToSpill;
    unsigned MinWeight = ~0U;
    MCPhysReg BestPReg = 0;

    for (MCPhysReg PReg : AllocationOrder) {
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

    if (BestToSpill) {
      spill(BestToSpill);
      return BestPReg;
    }
    return 0;
  }

  void spill(Register VReg) {
    if (VRM->getStackSlot(VReg) != VirtRegMap::NO_STACK_SLOT) return;
    VRM->assignVirt2StackSlot(VReg);
    for (auto it = PhysRegState.begin(); it != PhysRegState.end(); ++it) {
      if (it->second == VReg) {
        PhysRegState.erase(it);
        break;
      }
    }
  }
};

} // end anonymous namespace

char RASSA::ID = 0;

// Register the allocator so it can be called via llc -regalloc=ssa
static RegisterRegAlloc ssaRegAlloc("ssa", "SSA Register Allocator", 
                                    []() -> FunctionPass* { return new RASSA(); });

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)