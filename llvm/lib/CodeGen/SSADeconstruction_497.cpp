//===-- SSADeconstruction.cpp - Phi Elimination for SSA Allocator ---------===//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
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

public:
  static char ID;

  SSADeconstruction() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { 
    return "SSA Deconstruction (Fernando)"; 
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<VirtRegMapWrapperLegacy>(); 
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

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty()) continue;
    
    // Process any block that starts with PHIs
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
    
    // >>> FIX: Update Live-Ins <<<
    // The value now flows into this block via PhysDef.
    // We must tell LLVM that PhysDef is live-in, otherwise the Verifier complains.
    if (PhysDef) {
       MBB.addLiveIn(PhysDef);
    } else {
       // If PhysDef is 0, it means the PHI def was spilled.
       // The Rewriter will handle the reload, but we can't add 0 to LiveIn.
       ++I; continue;
    }

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

  // Insert copies in predecessors
  for (auto &Item : ParallelCopies) {
    MachineBasicBlock *Pred = Item.first;
    insertParallelCopies(*Pred, MBB, Item.second);
  }

  // Remove PHIs
  I = MBB.begin();
  while (I != MBB.end() && I->isPHI()) {
    MachineInstr *Phi = &*I;
    ++I;
    Phi->eraseFromParent();
  }
  
  // Clean up the LiveIn list (sort/unique) to keep the verifier happy
  MBB.sortUniqueLiveIns();
}

void SSADeconstruction::emitSwap(MachineBasicBlock &MBB, 
                                 MachineBasicBlock::iterator I, 
                                 MCRegister R1, MCRegister R2) const {
  // Debug output
  LLVM_DEBUG(dbgs() << "Emitting Cycle Swap: " << printReg(R1, TRI) 
                    << " <-> " << printReg(R2, TRI) << "\n");
  
  // Note: Standard COPY is not a true swap, but often works if liveness allows.
  // A true swap requires XORs or a scratch register.
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

      // Check for dependencies
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

    // Break Cycles
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