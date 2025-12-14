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
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  void insertParallelCopies(MachineBasicBlock &PredMBB, 
                            MachineBasicBlock &SuccMBB,
                            SmallVectorImpl<CopyOp> &Copies);
  
  void emitSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator I, 
                MCRegister R1, MCRegister R2) const;
                
  void propagateLiveIn(MCRegister PhysReg, MachineBasicBlock *StartMBB);
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

// Robust Liveness Propagation:
// Ensures PhysReg is marked LiveIn from StartMBB up to its reaching definition.
void SSADeconstruction::propagateLiveIn(MCRegister PhysReg, MachineBasicBlock *StartMBB) {
    SmallVector<MachineBasicBlock*, 8> Worklist;
    SmallPtrSet<MachineBasicBlock*, 16> Visited;

    Worklist.push_back(StartMBB);
    Visited.insert(StartMBB);

    while (!Worklist.empty()) {
        MachineBasicBlock *MBB = Worklist.pop_back_val();

        // If PhysReg is defined locally in this block, we stop traversing up.
        // We scan backwards from the end (or where we conceptually added the use).
        // However, a simple "is defined" check is checking for Defs.
        bool IsDefinedLocally = false;
        for (const MachineInstr &MI : *MBB) {
            if (MI.definesRegister(PhysReg, TRI)) {
                IsDefinedLocally = true;
                break;
            }
        }

        // If not defined locally, it must be Live-In.
        if (!IsDefinedLocally) {
            if (!MBB->isLiveIn(PhysReg)) {
                MBB->addLiveIn(PhysReg);
            }

            // Continue searching up predecessors
            for (MachineBasicBlock *Pred : MBB->predecessors()) {
                if (Visited.insert(Pred).second) {
                    Worklist.push_back(Pred);
                }
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

  // --- PHASE 1: Analyze & Prepare ---
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty() || !MBB.front().isPHI()) continue;

    for (MachineInstr &Phi : MBB) {
      if (!Phi.isPHI()) break;

      Register DefVReg = Phi.getOperand(0).getReg();
      MCRegister PhysDef = VRM->getPhys(DefVReg);

      if (!PhysDef) continue;

      // 1. Handle Definition Liveness (Downstream)
      // The PHI definition becomes a Live-In to the current block (MBB)
      // because it is now defined in the Predecessors.
      if (!MBB.isLiveIn(PhysDef)) {
          MBB.addLiveIn(PhysDef);
      }
      
      // We also need to ensure downstream blocks that use this value know it's live-in.
      // (This uses the BFS approach for VReg uses)
      {
          SmallVector<MachineBasicBlock*, 8> UseWorklist;
          SmallPtrSet<MachineBasicBlock*, 16> UseVisited;
          UseVisited.insert(&MBB); // Start here

          for (MachineInstr &UseMI : MRI->use_nodbg_instructions(DefVReg)) {
              if (UseMI.isPHI()) continue;
              MachineBasicBlock *UseMBB = UseMI.getParent();
              if (UseMBB != &MBB && UseVisited.insert(UseMBB).second) {
                  UseWorklist.push_back(UseMBB);
              }
          }
          
          while (!UseWorklist.empty()) {
              MachineBasicBlock *Curr = UseWorklist.pop_back_val();
              if (!Curr->isLiveIn(PhysDef)) Curr->addLiveIn(PhysDef);
              
              for (MachineBasicBlock *Pred : Curr->predecessors()) {
                  if (UseVisited.insert(Pred).second) UseWorklist.push_back(Pred);
              }
          }
      }

      // 2. Handle Source Liveness (Upstream)
      for (unsigned i = 1, e = Phi.getNumOperands(); i < e; i += 2) {
        Register SrcVReg = Phi.getOperand(i).getReg();
        MachineBasicBlock *Pred = Phi.getOperand(i + 1).getMBB();
        
        MCRegister PhysSrc;
        if (SrcVReg.isPhysical()) PhysSrc = SrcVReg.asMCReg();
        else PhysSrc = VRM->getPhys(SrcVReg);

        if (PhysSrc) {
            // If we insert a read of PhysSrc in Pred, we must ensure 
            // PhysSrc is live-in to Pred (unless defined there).
            propagateLiveIn(PhysSrc, Pred);

            if (PhysDef != PhysSrc) {
                ParallelCopies[Pred].push_back({PhysDef, PhysSrc});
            }
        }
      }

      PhisToRemove.push_back(&Phi);
    }
  }

  // --- PHASE 2: Execute ---
  // Insert Copies
  for (auto &Item : ParallelCopies) {
    MachineBasicBlock *Pred = Item.first;
    insertParallelCopies(*Pred, *Pred, Item.second);
  }

  // Rewrite Uses & Remove PHIs
  for (MachineInstr *Phi : PhisToRemove) {
    Register DefVReg = Phi->getOperand(0).getReg();
    MCRegister PhysDef = VRM->getPhys(DefVReg);

    if (PhysDef) {
       MRI->replaceRegWith(DefVReg, PhysDef);
    }
    Phi->eraseFromParent();
  }

  // Final Cleanup
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
        if (Other.Src == D) Other.Src = S;
      }
    }
  }
}