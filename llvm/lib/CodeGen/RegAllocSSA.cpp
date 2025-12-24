#include "RegAllocBase.h"
#include "AllocationOrder.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
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

  // Needed to satisfy RegAllocBase interface, but we use manual spilling
  class DummySpiller : public Spiller {
  public:
    void spill(LiveRangeEdit &) override {}
    void postOptimization() override {}
  };
  DummySpiller DS;

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
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // --- Fernando's Weight Method ---
  // w = (1 + 10^depth) [for def] + sum(1 + 10^depth) [for each use]
  float calculateFernandoWeight(Register Reg) {
    float Weight = 0.0f;
    MachineRegisterInfo &MRI = MF->getRegInfo();

    auto AddWeight = [&](const MachineInstr *MI) {
      unsigned Depth = MLI->getLoopDepth(MI->getParent());
      Weight += (1.0f + std::pow(10.0f, (float)Depth));
    };

    if (MachineInstr *DefMI = MRI.getVRegDef(Reg))
      AddWeight(DefMI);

    for (MachineInstr &UseMI : MRI.use_instructions(Reg))
      AddWeight(&UseMI);

    return Weight;
  }

  Spiller &spiller() override { return DS; }

  void enqueueImpl(const LiveInterval *LI) override {}
  const LiveInterval *dequeue() override { return nullptr; }

  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override {
    // 1. Try to find free physical register
    auto Order = AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
    for (MCRegister PhysReg : Order) {
      if (Matrix->checkInterference(VirtReg, PhysReg) == LiveRegMatrix::IK_Free)
        return PhysReg;
    }

    // 2. Manual Spill (Fernando Method)
    // If it's a PHI or critical, we must not spill. 
    // Otherwise, we assign a stack slot and the rewriter handles it.
    if (!VirtReg.isSpillable())
      return 0;

    if (!VRM->hasStackSlot(VirtReg.reg()))
      VRM->assignVirt2StackSlot(VirtReg.reg());

    return 0; 
  }

  bool runOnMachineFunction(MachineFunction &mf) override {
    MF = &mf;
    MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    auto &Matrix = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
    MachineRegisterInfo &MRI = MF->getRegInfo();

    RegAllocBase::init(VRM, LIS, Matrix);
    RegClassInfo.runOnMachineFunction(*MF);

    // --- Prepare Weights & Protect PHIs ---
    std::vector<Register> VRegs;
    for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
      Register Reg = Register::index2VirtReg(i);
      if (MRI.reg_nodbg_empty(Reg)) continue;

      LiveInterval &LI = LIS.getInterval(Reg);
      LI.setWeight(calculateFernandoWeight(Reg));

      // PHI Protection: Mark unspillable to prevent Spiller crashes
      if (MRI.getVRegDef(Reg) && MRI.getVRegDef(Reg)->isPHI()) {
        LI.markNotSpillable();
        LI.setWeight(HUGE_VALF); 
      }
      VRegs.push_back(Reg);
    }

    // --- Sort by Fernando's Weights ---
    std::sort(VRegs.begin(), VRegs.end(), [&](Register A, Register B) {
      LiveInterval &LIA = LIS.getInterval(A);
      LiveInterval &LIB = LIS.getInterval(B);
      if (LIA.isSpillable() != LIB.isSpillable())
        return !LIA.isSpillable();
      return LIA.weight() > LIB.weight();
    });

    // --- Main Allocation Loop ---
    for (Register Reg : VRegs) {
      if (VRM.hasPhys(Reg)) continue;
      LiveInterval &LI = LIS.getInterval(Reg);
      
      SmallVector<Register, 4> SplitVRegs;
      MCRegister PhysReg = selectOrSplit(LI, SplitVRegs);
      
      if (PhysReg) {
        Matrix.assign(LI, PhysReg);
      } else if (!LI.isSpillable()) {
        report_fatal_error("Could not allocate register for PHI node in SSA!");
      }
    }

    mf.getProperties().set(MachineFunctionProperties::Property::NoPHIs);
    return true;
  }
};

char RASSA::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(RASSA, "ssa", "SSA Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "ssa", "SSA Allocator", false, false)

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }