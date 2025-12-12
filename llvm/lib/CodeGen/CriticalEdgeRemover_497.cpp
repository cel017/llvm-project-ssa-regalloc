#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "crit-edge-removal"

namespace {

class CriticalEdgeRemoval : public MachineFunctionPass {
public:
  static char ID;

  CriticalEdgeRemoval() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { 
    return "Critical Edge Removal (Fernando)"; 
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<LiveVariablesWrapperPass>(); // runs before this in TargetPassConfig
    
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Changed = false;
    
    // collect edges first to avoid iterator invalidation
    SmallVector<std::pair<MachineBasicBlock*, MachineBasicBlock*>, 32> CriticalEdges;

    for (MachineBasicBlock &MBB : MF) {
      if (MBB.succ_size() <= 1) continue;

      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (Succ->pred_size() > 1) {
          CriticalEdges.push_back({&MBB, Succ});
        }
      }
    }

    if (CriticalEdges.empty()) return false;

    for (auto &Edge : CriticalEdges) {
      MachineBasicBlock *Src = Edge.first;
      MachineBasicBlock *Dst = Edge.second;

      // preserve (DomTree, LiveVars) and update them automatically.
      MachineBasicBlock *NewBB = Src->SplitCriticalEdge(Dst, *this);
      
      if (NewBB) {
        Changed = true;
        LLVM_DEBUG(dbgs() << "Split critical edge: BB#" << Src->getNumber() 
                          << " -> BB#" << Dst->getNumber() 
                          << " (New BB#" << NewBB->getNumber() << ")\n");
      }
    }

    return Changed;
  }
};

} // end anonymous namespace

char CriticalEdgeRemoval::ID = 0;

static RegisterPass<CriticalEdgeRemoval> 
X("crit-edge-removal", "Critical Edge Removal", false, false);

namespace llvm {
  FunctionPass *createCriticalEdgeRemovalPass() {
    return new CriticalEdgeRemoval();
  }
}