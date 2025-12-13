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
#include "llvm/ADT/SmallPtrSet.h" // Needed for Visited set
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
                
  // Helper to trace liveness without relying on stale LiveIntervals
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

bool SSADeconstruction::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();

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

// Trace liveness from Uses up to the DefMBB
void SSADeconstruction::propagateLiveIn(MCRegister PhysReg, Register VReg, MachineBasicBlock *DefMBB) {
    SmallVector<MachineBasicBlock*, 8> Worklist;
    SmallPtrSet<MachineBasicBlock*, 16> Visited;

    // 1. Definition Block ALWAYS needs LiveIn (value flows from Predecessors via copies)
    if (!DefMBB->isLiveIn(PhysReg))
        DefMBB->addLiveIn(PhysReg);
        
    Visited.insert(DefMBB); // Stop traversal here

    // 2. Find all User Blocks
    for (MachineInstr &UseMI : MRI->use_nodbg_instructions(VReg)) {
        MachineBasicBlock *UseMBB = UseMI.getParent();
        // If use is in a different block, start searching up
        if (UseMBB != DefMBB && Visited.insert(UseMBB).second) {
            Worklist.push_back(UseMBB);
        }
    }

    // 3. BFS up the CFG
    while (!Worklist.empty()) {
        MachineBasicBlock *Curr = Worklist.pop_back_val();

        // Mark Live-In
        if (!Curr->isLiveIn(PhysReg)) {
            Curr->addLiveIn(PhysReg);
        }

        // Push Predecessors
        for (MachineBasicBlock *Pred : Curr->predecessors()) {
            if (Visited.insert(Pred).second) {
                Worklist.push_back(Pred);
            }
        }
    }
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

    // >>> ROBUST FIX: Update Live-Ins via Use-Def BFS <<<
    // This marks MBB and all blocks on the path to uses as LiveIn(PhysDef).
    propagateLiveIn(PhysDef, DefVReg, &MBB);
    
    // Collect Copies
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

    if (PhysDef) {
      MRI->replaceRegWith(DefVReg, PhysDef);
    }

    ++I;
    Phi.eraseFromParent();
  }
  
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