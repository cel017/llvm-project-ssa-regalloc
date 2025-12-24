#include "RegAllocBase.h"
#include "AllocationOrder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <algorithm>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

namespace {

class RASSA : public MachineFunctionPass, public RegAllocBase {
  MachineFunction *MF = nullptr;
  MachineLoopInfo *MLI = nullptr;

  // We define a Manual Spiller that does nothing because we handle 
  // stack assignment manually in selectOrSplit to bypass the 
  // crashing 'isRematerializable' logic in standard LLVM spillers.
  class ManualSSA_Spiller : public Spiller {
  public:
    void spill(LiveRangeEdit &) override {}
    void postOptimization() override {}
  };
  ManualSSA_Spiller MSSASpiller;

public:
  static char ID;
  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Chordal-Style Allocator"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    
    // Preserve analyses for the Deconstruction pass and the Rewriter
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  /// Fernando's Weighting Algorithm:
  /// w(v) = (1 + 10^loop_depth(def)) + sum_{uses}(1 + 10^loop_depth(use))
  float calculateFernandoWeight(Register Reg) {
    float Weight = 0.0f;
    MachineRegisterInfo &MRI = MF->getRegInfo();

    auto AddWeight = [&](const MachineInstr *MI) {
      if (!MI) return;
      unsigned Depth = MLI->getLoopDepth(MI->getParent());
      Weight += (1.0f + std::pow(10.0f, (float)Depth));
    };

    AddWeight(MRI.getVRegDef(Reg));
    for (MachineInstr &UseMI : MRI.use_instructions(Reg))
      AddWeight(&UseMI);

    return Weight;
  }

  Spiller &spiller() override { return MSSASpiller; }

  void enqueueImpl(const LiveInterval *LI) override {}
  const LiveInterval *dequeue() override { return nullptr; }

  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override {
    
    // 1. Try to assign a physical register
    auto Order = AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
    for (MCRegister PhysReg : Order) {
      if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_Free)
        return PhysReg;
    }

    // 2. If no PhysReg is free, we "Spill" by assigning a stack slot manually.
    // PHIs and ImplicitDefs are protected (marked unspillable) in the main loop.
    if (!VirtReg.isSpillable())
      return 0;

    if (VRM->getStackSlot(VirtReg.reg()) == VirtRegMap::NO_STACK_SLOT)
      VRM->assignVirt2StackSlot(VirtReg.reg());

    return 0; // Return 0 to indicate stack residency
  }

  bool runOnMachineFunction(MachineFunction &mf) override {
    MF = &mf;
    MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    auto &VRM_Wrapper = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
    MachineRegisterInfo &MRI = MF->getRegInfo();

    RegAllocBase::init(VRM_Wrapper, LIS, LRM);
    RegClassInfo.runOnMachineFunction(*MF);

    std::vector<Register> VRegs;

    // --- Phase 1: Pre-process and Weighting ---
    for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
      Register Reg = Register::index2VirtReg(i);
      if (MRI.reg_nodbg_empty(Reg)) continue;

      LiveInterval &LI = LIS.getInterval(Reg);
      
      // Prioritize registers in deep loops using Fernando's algorithm
      LI.setWeight(calculateFernandoWeight(Reg));

      // Shield PHIs and ImplicitDefs from spilling logic to prevent Segfaults
      MachineInstr *DefMI = MRI.getVRegDef(Reg);
      if (DefMI && (DefMI->isPHI() || DefMI->isImplicitDef())) {
        LI.markNotSpillable();
        LI.setWeight(HUGE_VALF); 
      }
      VRegs.push_back(Reg);
    }

    // --- Phase 2: Priority Sorting ---
    std::sort(VRegs.begin(), VRegs.end(), [&](Register A, Register B) {
      LiveInterval &LIA = LIS.getInterval(A);
      LiveInterval &LIB = LIS.getInterval(B);
      if (LIA.isSpillable() != LIB.isSpillable())
        return !LIA.isSpillable(); 
      return LIA.weight() > LIB.weight();
    });

    // --- Phase 3: Assignment ---
    for (Register Reg : VRegs) {
      if (VRM->hasPhys(Reg)) continue;
      LiveInterval &LI = LIS.getInterval(Reg);
      
      SmallVector<Register, 4> SplitVRegs;
      MCRegister PhysReg = selectOrSplit(LI, SplitVRegs);
      
      if (PhysReg) {
        Matrix->assign(LI, PhysReg);
      } else if (!LI.isSpillable()) {
        report_fatal_error("Infeasible: SSA PHI node could not be allocated!");
      }
    }

    return true;
  }
};

char RASSA::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(RASSA, "ssa", "SSA Chordal-Style Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "ssa", "SSA Chordal-Style Allocator", false, false)

namespace llvm {
  FunctionPass *createSSARegisterAllocator() { return new RASSA(); }
}