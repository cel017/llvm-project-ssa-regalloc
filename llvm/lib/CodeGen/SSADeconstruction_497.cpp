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
#include "llvm/ADT/SmallPtrSet.h"
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
    // We explicitly DO NOT require LiveIntervals, as we invalidate it.
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  void insertParallelCopies(MachineBasicBlock &PredMBB, 
                            MachineBasicBlock &SuccMBB,
                            SmallVectorImpl<CopyOp> &Copies);
  
  void emitSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator I, 
                MCRegister R1, MCRegister R2) const;
                
  void propagateLiveIn(MCRegister PhysReg, Register VReg, MachineBasicBlock *DefMBB);
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

// Trace liveness from Uses up to the DefMBB
void SSADeconstruction::propagateLiveIn(MCRegister PhysReg, Register VReg, MachineBasicBlock *DefMBB) {
    SmallVector<MachineBasicBlock*, 8> Worklist;
    SmallPtrSet<MachineBasicBlock*, 16> Visited;

    // 1. The Definition Block itself needs the register Live-In 
    // (because the value actually comes from the Predecessors via copies)
    if (!DefMBB->isLiveIn(PhysReg))
        DefMBB->addLiveIn(PhysReg);
        
    Visited.insert(DefMBB); 

    // 2. Find all User Blocks and trace up
    for (MachineInstr &UseMI : MRI->use_nodbg_instructions(VReg)) {
        if (UseMI.isPHI()) continue; // PHI uses are handled by the split copies
        
        MachineBasicBlock *UseMBB = UseMI.getParent();
        if (UseMBB != DefMBB && Visited.insert(UseMBB).second) {
            Worklist.push_back(UseMBB);
        }
    }

    // 3. BFS up the CFG
    while (!Worklist.empty()) {
        MachineBasicBlock *Curr = Worklist.pop_back_val();

        if (!Curr->isLiveIn(PhysReg)) {
            Curr->addLiveIn(PhysReg);
        }

        for (MachineBasicBlock *Pred : Curr->predecessors()) {
            if (Visited.insert(Pred).second) {
                Worklist.push_back(Pred);
            }
        }
    }
}

bool SSADeconstruction::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();

  SmallVector<MachineInstr*, 16> PhisToRemove;
  DenseMap<MachineBasicBlock*, SmallVector<CopyOp, 4>> ParallelCopies;

  // --- PHASE 1: Analysis & Copy Insertion ---
  // We identify all copies needed and update Liveness, but do not touch the PHIs yet.
  
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty() || !MBB.front().isPHI()) continue;

    for (MachineInstr &Phi : MBB) {
      if (!Phi.isPHI()) break;

      Register DefVReg = Phi.getOperand(0).getReg();
      MCRegister PhysDef = VRM->getPhys(DefVReg);

      if (!PhysDef) continue; // Spilled?

      // 1. Update Liveness for the Definition
      // We do this BEFORE rewriting uses so MRI->use_instructions works.
      propagateLiveIn(PhysDef, DefVReg, &MBB);

      // 2. Collect Copies needed in Predecessors
      for (unsigned i = 1, e = Phi.getNumOperands(); i < e; i += 2) {
        Register SrcVReg = Phi.getOperand(i).getReg();
        MachineBasicBlock *Pred = Phi.getOperand(i + 1).getMBB();
        
        // Handle case where Src is already physical (rare but possible in mixed code)
        MCRegister PhysSrc;
        if (SrcVReg.isPhysical()) {
             PhysSrc = SrcVReg.asMCReg();
        } else {
             PhysSrc = VRM->getPhys(SrcVReg);
        }

        if (PhysSrc && PhysDef != PhysSrc) {
          ParallelCopies[Pred].push_back({PhysDef, PhysSrc});
        }
      }

      PhisToRemove.push_back(&Phi);
    }
  }

  // Insert all Parallel Copies now
  for (auto &Item : ParallelCopies) {
    MachineBasicBlock *Pred = Item.first;
    // We assume Pred flows to the PHI block. 
    // Note: insertParallelCopies handles the sequentialization.
    // We pass *Pred as the second arg just as a placeholder for context if needed,
    // but the function actually inserts at the end of Pred.
    insertParallelCopies(*Pred, *Pred, Item.second);
  }

  // --- PHASE 2: Destruction ---
  // Now it is safe to remove PHIs and rewrite uses.
  
  for (MachineInstr *Phi : PhisToRemove) {
    Register DefVReg = Phi->getOperand(0).getReg();
    MCRegister PhysDef = VRM->getPhys(DefVReg);

    if (PhysDef) {
       // Rewrite all non-PHI uses to the physical register
       // Note: replaceRegWith replaces *all* uses. 
       // Uses in other PHI nodes effectively become "PhysReg inputs", 
       // which is why we waited until Phase 2 to do this (so we don't process them again).
       MRI->replaceRegWith(DefVReg, PhysDef);
    }
    
    Phi->eraseFromParent();
  }

  // Cleanup LiveIns lists
  for (MachineBasicBlock &MBB : MF) {
    MBB.sortUniqueLiveIns();
  }

  return !PhisToRemove.empty();
}

void SSADeconstruction::emitSwap(MachineBasicBlock &MBB, 
                                 MachineBasicBlock::iterator I, 
                                 MCRegister R1, MCRegister R2) const {
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