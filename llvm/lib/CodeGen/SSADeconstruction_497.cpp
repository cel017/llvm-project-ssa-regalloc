//===-- SSADeconstruction.cpp - Phi Elimination for SSA Allocator ---------===//
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
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/InitializePasses.h"

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
    return "SSA Deconstruction"; 
  }


void getAnalysisUsage(AnalysisUsage &AU) const override {
  AU.setPreservesCFG();
  
  // We need VRM to know where the VRegs were assigned
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>(); 
  
  // We need LIS if we are doing any liveness propagation
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>(); 

  MachineFunctionPass::getAnalysisUsage(AU);
}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  void insertParallelCopies(MachineBasicBlock &PredMBB, 
                            MachineBasicBlock &SuccMBB,
                            SmallVectorImpl<CopyOp> &Copies);
  
  // FIX: Helper to emit a safe swap without a scratch register
  void emitXorSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator I, 
                   MCRegister R1, MCRegister R2) const;
                
  void propagateLiveIn(MCRegister PhysReg, MachineBasicBlock *StartMBB);
};

} // end anonymous namespace

char SSADeconstruction::ID = 0;

static RegisterPass<SSADeconstruction> 
X("ssa-deconstruction", "SSA Deconstruction", false, false);

INITIALIZE_PASS_BEGIN(SSADeconstruction, "ssa-deconstruction",
                      "SSA Deconstruction", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(SSADeconstruction, "ssa-deconstruction",
                    "SSA Deconstruction", false, false)

namespace llvm {
  FunctionPass *createSSADeconstructionPass() {
    return new SSADeconstruction();
  }
}

// Helper to find the target-specific XOR opcode by name.
// This works for RISC-V because the instruction is literally named "XOR".
static unsigned getXorOpcode(const TargetInstrInfo *TII) {
    int Num = TII->getNumOpcodes();
    for (int i = 0; i < Num; ++i) {
        // Check for RISC-V "XOR"
        if (StringRef(TII->getName(i)) == "XOR") return i;
    }
    llvm_unreachable("XOR instruction not found! (Is this the right target?)");
}

void SSADeconstruction::emitXorSwap(MachineBasicBlock &MBB, 
                                    MachineBasicBlock::iterator I, 
                                    MCRegister R1, MCRegister R2) const {
    // 1. Find the target-specific opcode dynamically
    unsigned XorOpcode = getXorOpcode(TII);

    // 2. Emit the 3-XOR Swap Sequence
    // R1 = R1 ^ R2
    BuildMI(MBB, I, DebugLoc(), TII->get(XorOpcode), R1)
        .addReg(R1).addReg(R2);
        
    // R2 = R2 ^ R1
    BuildMI(MBB, I, DebugLoc(), TII->get(XorOpcode), R2)
        .addReg(R2).addReg(R1);
        
    // R1 = R1 ^ R2
    BuildMI(MBB, I, DebugLoc(), TII->get(XorOpcode), R1)
        .addReg(R1).addReg(R2);
}

void SSADeconstruction::propagateLiveIn(MCRegister PhysReg, MachineBasicBlock *StartMBB) {
    SmallVector<MachineBasicBlock*, 8> Worklist;
    SmallPtrSet<MachineBasicBlock*, 16> Visited;

    Worklist.push_back(StartMBB);
    Visited.insert(StartMBB);

    while (!Worklist.empty()) {
        MachineBasicBlock *MBB = Worklist.pop_back_val();

        bool IsDefinedLocally = false;
        for (const MachineInstr &MI : *MBB) {
            if (MI.definesRegister(PhysReg, TRI)) {
                IsDefinedLocally = true;
                break;
            }
        }

        if (!IsDefinedLocally) {
            if (!MBB->isLiveIn(PhysReg)) {
                MBB->addLiveIn(PhysReg);
            }
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

  // --- PHASE 1: Analyze PHIs ---
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty()) continue;

    for (MachineInstr &Phi : MBB) {
      if (!Phi.isPHI()) break; // PHIs are always at the start

      Register DefVReg = Phi.getOperand(0).getReg();
      
      // CRITICAL: The allocator MUST have assigned a register to this PHI def.
      if (!VRM->hasPhys(DefVReg)) {
          // If this triggers, your Allocator skipped the PHI register!
          llvm_unreachable("PHI Definition has no Physical Register assigned!");
      }
      MCRegister PhysDef = VRM->getPhys(DefVReg);

      // 1. Mark Def as Live-In to this block (it comes from preds)
      if (!MBB.isLiveIn(PhysDef)) MBB.addLiveIn(PhysDef);

      // 2. Analyze Sources
      for (unsigned i = 1, e = Phi.getNumOperands(); i < e; i += 2) {
        Register SrcVReg = Phi.getOperand(i).getReg();
        MachineBasicBlock *Pred = Phi.getOperand(i + 1).getMBB();
        
        MCRegister PhysSrc;
        if (SrcVReg.isPhysical()) {
            PhysSrc = SrcVReg.asMCReg();
        } else {
            // Source might be undefined or spilled? 
            // We assume allocator handled it.
            if (VRM->hasPhys(SrcVReg))
                PhysSrc = VRM->getPhys(SrcVReg);
            else
                continue; // Should imply implicit undef
        }

        // Add to Parallel Copy List for that Predecessor
        if (PhysDef != PhysSrc) {
            ParallelCopies[Pred].push_back({PhysDef, PhysSrc});
            // Ensure the source is available in the predecessor
            propagateLiveIn(PhysSrc, Pred);
        }
      }
      PhisToRemove.push_back(&Phi);
    }
  }

  // --- PHASE 2: Insert Copies ---
  for (auto &Item : ParallelCopies) {
    MachineBasicBlock *Pred = Item.first;
    insertParallelCopies(*Pred, *Pred, Item.second);
  }

  // --- PHASE 3: Rewrite & Remove ---
for (MachineInstr *Phi : PhisToRemove) {
    Phi->eraseFromParent(); // Just remove the PHI, leave the VRegs alone!
  }

  return !PhisToRemove.empty();
}

void SSADeconstruction::insertParallelCopies(
    MachineBasicBlock &PredMBB, 
    MachineBasicBlock &SuccMBB, // Unused in this logic, but good for context
    SmallVectorImpl<CopyOp> &Copies) {

  SmallVector<CopyOp, 4> WorkList(Copies.begin(), Copies.end());
  
  // Insert at the end of the predecessor (before terminators like JMP/RET)
  MachineBasicBlock::iterator InsertPos = PredMBB.getFirstTerminator();

  while (!WorkList.empty()) {
    bool Progress = false;
    auto It = WorkList.begin();

    while (It != WorkList.end()) {
      MCRegister Dst = It->Dest;
      bool DstIsSource = false;

      // Check if writing to Dst overwrites a value needed by another pending copy
      for (const auto &Other : WorkList) {
        if (&Other == &*It) continue;
        if (Other.Src == Dst) {
          DstIsSource = true;
          break;
        }
      }

      if (!DstIsSource) {
        // Safe to copy: Dst is not needed as a source anymore
        TII->copyPhysReg(PredMBB, InsertPos, DebugLoc(), 
                         It->Dest, It->Src, It->Dest != It->Src); 
        It = WorkList.erase(It);
        Progress = true;
      } else {
        ++It;
      }
    }

    // Cycle detected! (e.g. A->B, B->A)
    if (!Progress && !WorkList.empty()) {
      CopyOp &Current = WorkList.back();
      MCRegister D = Current.Dest;
      MCRegister S = Current.Src;
      
      // Perform Swap to break the cycle
      emitXorSwap(PredMBB, InsertPos, D, S);
      
      WorkList.pop_back();
      
      // Update remaining dependencies
      for (auto &Other : WorkList) {
        if (Other.Src == D) Other.Src = S;
      }
    }
  }
}