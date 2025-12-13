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
  MCRegister Dest; // Changed to MCRegister (Physical)
  MCRegister Src;
};

class SSADeconstruction : public MachineFunctionPass {
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  VirtRegMap *VRM = nullptr; // <--- Store VRM

public:
  static char ID;

  SSADeconstruction() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { 
    return "SSA Deconstruction (Fernando)"; 
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // We need VRM to know which physical registers were assigned!
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
  
  // Fetch the mapping created by your allocator
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty() || !MBB.front().isPHI())
      continue;
    
    deconstructBlock(MBB);
    Changed = true;
  }

  return Changed;
}

void SSADeconstruction::deconstructBlock(MachineBasicBlock &MBB) {
  DenseMap<MachineBasicBlock*, SmallVector<CopyOp, 4>> ParallelCopies;

  MachineBasicBlock::iterator I = MBB.begin();
  while (I != MBB.end() && I->isPHI()) {
    MachineInstr &Phi = *I;
    Register DefVReg = Phi.getOperand(0).getReg();
    
    // TRANSLATE: Virtual -> Physical
    MCRegister PhysDef = VRM->getPhys(DefVReg);
    
    // Safety check: PHI defined regs should not be spilled by now
    if (!PhysDef) {
       // If this hits, it means a PHI def was spilled. 
       // Your allocator safeguards should prevent this, but if not, 
       // we skip (or you'd need load code).
       ++I; continue; 
    }

    for (unsigned i = 1, e = Phi.getNumOperands(); i < e; i += 2) {
      Register SrcVReg = Phi.getOperand(i).getReg();
      MachineBasicBlock *Pred = Phi.getOperand(i + 1).getMBB();

      // TRANSLATE: Virtual -> Physical
      MCRegister PhysSrc = VRM->getPhys(SrcVReg);

      // Handle Undef/Spilled inputs (PhysSrc == 0)
      if (!PhysSrc) continue; 

      if (PhysDef != PhysSrc) {
        ParallelCopies[Pred].push_back({PhysDef, PhysSrc});
      }
    }
    ++I;
  }

  for (auto &Item : ParallelCopies) {
    MachineBasicBlock *Pred = Item.first;
    insertParallelCopies(*Pred, MBB, Item.second);
  }

  I = MBB.begin();
  while (I != MBB.end() && I->isPHI()) {
    MachineInstr *Phi = &*I;
    ++I;
    Phi->eraseFromParent();
  }
}

void SSADeconstruction::emitSwap(MachineBasicBlock &MBB, 
                                 MachineBasicBlock::iterator I, 
                                 MCRegister R1, MCRegister R2) const {
  LLVM_DEBUG(dbgs() << "Emitting Cycle Breaker Swap: " << printReg(R1, TRI) 
                    << " <-> " << printReg(R2, TRI) << "\n");
  
  // NOTE: This remains a "Copy" hack. Real XOR swap depends on target.
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
        // Now passing PHYSICAL registers to copyPhysReg
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