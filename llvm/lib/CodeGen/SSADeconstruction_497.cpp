//===-- SSADeconstruction.cpp - Phi Elimination for SSA Allocator ---------===//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervals.h" // <--- Added
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/CodeGen/Passes.h"

using namespace llvm;

#define DEBUG_TYPE "ssa-deconstruction"

namespace {

struct CopyOp {
  MCRegister Dest; 
  MCRegister Src;
};

class SSADeconstruction : public MachineFunctionPass {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr; // <--- Added

public:
  static char ID;

  SSADeconstruction() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { 
    return "SSA Deconstruction (Fernando)"; 
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<VirtRegMapWrapperLegacy>(); 
    AU.addRequired<LiveIntervalsWrapperPass>(); // <--- Added
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  void deconstructBlock(MachineBasicBlock &MBB);
  void insertParallelCopies(MachineBasicBlock &PredMBB, 
                            MachineBasicBlock &SuccMBB,
                            SmallVectorImpl<CopyOp> &Copies);
  
  void emitSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator I, 
                MCRegister R1, MCRegister R2) const;
};

} // end anonymous namespace

char SSADeconstruction::ID = 0;

static RegisterPass<SSADeconstruction> 
X("ssa-deconstruction", "SSA Deconstruction", false, false);

namespace llvm {
  FunctionPass *createSSADeconstructionPass() {
    return new SSADeconstruction();
  }
}

bool SSADeconstruction::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS(); // <--- Fetch LIS

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty()) continue;
    
    if (MBB.front().isPHI()) {
      deconstructBlock(MBB);
      Changed = true;
    }
  }

  return Changed;
}

void SSADeconstruction::deconstructBlock(MachineBasicBlock &MBB) {
  DenseMap<MachineBasicBlock*, SmallVector<CopyOp, 4>> ParallelCopies;

  MachineBasicBlock::iterator I = MBB.begin();
  while (I != MBB.end() && I->isPHI()) {
    MachineInstr &Phi = *I;
    Register DefVReg = Phi.getOperand(0).getReg();
    MCRegister PhysDef = VRM->getPhys(DefVReg);

    if (!PhysDef) {
       ++I; continue; 
    }
    
    // >>> FIX: Propagate Live-Ins Globally <<<
    // Since we are replacing DefVReg with PhysDef everywhere, 
    // any block that had DefVReg as Live-In must now have PhysDef as Live-In.
    if (LIS->hasInterval(DefVReg)) {
        const LiveInterval &LI = LIS->getInterval(DefVReg);
        for (MachineBasicBlock &AnyMBB : *MBB.getParent()) {
            if (LIS->isLiveInToMBB(LI, &AnyMBB)) {
                if (!AnyMBB.isLiveIn(PhysDef)) {
                    AnyMBB.addLiveIn(PhysDef);
                }
            }
        }
    }
    
    // Collect Copies for Predecessors
    for (unsigned i = 1, e = Phi.getNumOperands(); i < e; i += 2) {
      Register SrcVReg = Phi.getOperand(i).getReg();
      MachineBasicBlock *Pred = Phi.getOperand(i + 1).getMBB();
      MCRegister PhysSrc = VRM->getPhys(SrcVReg);

      if (!PhysSrc) continue; 
      if (PhysDef != PhysSrc) {
        ParallelCopies[Pred].push_back({PhysDef, PhysSrc});
      }
    }
    ++I;
  }

  // Insert Parallel Copies
  for (auto &Item : ParallelCopies) {
    MachineBasicBlock *Pred = Item.first;
    insertParallelCopies(*Pred, MBB, Item.second);
  }

  // Remove PHIs and Rewrite Uses
  I = MBB.begin();
  while (I != MBB.end() && I->isPHI()) {
    MachineInstr &Phi = *I;
    Register DefVReg = Phi.getOperand(0).getReg();
    MCRegister PhysDef = VRM->getPhys(DefVReg);

    // Rewrite all uses to Physical Register
    if (PhysDef) {
      MRI->replaceRegWith(DefVReg, PhysDef);
    }

    ++I;
    Phi.eraseFromParent();
  }
  
  // Cleanup LiveIns for the current block just in case
  MBB.sortUniqueLiveIns();
}

void SSADeconstruction::emitSwap(MachineBasicBlock &MBB, 
                                 MachineBasicBlock::iterator I, 
                                 MCRegister R1, MCRegister R2) const {
  LLVM_DEBUG(dbgs() << "Emitting Cycle Swap: " << printReg(R1, TRI) 
                    << " <-> " << printReg(R2, TRI) << "\n");
  BuildMI(MBB, I, DebugLoc(), TII->get(TargetOpcode::COPY), R1).addReg(R2);
}

void SSADeconstruction::insertParallelCopies(
    MachineBasicBlock &PredMBB, 
    MachineBasicBlock &SuccMBB,
    SmallVectorImpl<CopyOp> &Copies) {

  SmallVector<CopyOp, 4> WorkList(Copies.begin(), Copies.end());
  MachineBasicBlock::iterator InsertPos = PredMBB.getFirstTerminator();

  while (!WorkList.empty()) {
    bool Progress = false;
    auto It = WorkList.begin();

    while (It != WorkList.end()) {
      MCRegister Dst = It->Dest;
      bool DstIsSource = false;

      for (const auto &Other : WorkList) {
        if (&Other == &*It) continue;
        if (Other.Src == Dst) {
          DstIsSource = true;
          break;
        }
      }

      if (!DstIsSource) {
        TII->copyPhysReg(PredMBB, InsertPos, DebugLoc(), 
                         It->Dest, It->Src, It->Dest != It->Src); 
        It = WorkList.erase(It);
        Progress = true;
      } else {
        ++It;
      }
    }

    if (!Progress && !WorkList.empty()) {
      CopyOp &Current = WorkList.back();
      MCRegister D = Current.Dest;
      MCRegister S = Current.Src;
      
      emitSwap(PredMBB, InsertPos, D, S);
      
      WorkList.pop_back();
      
      for (auto &Other : WorkList) {
        if (Other.Src == D) {
            Other.Src = S;
        }
      }
    }
  }
}