// ... (Includes and Headers same as before)
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h" 
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <algorithm>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "regalloc-ssa"

FunctionPass *llvm::createSSARegisterAllocator();

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                      createSSARegisterAllocator);

namespace {

class RASSA : public MachineFunctionPass {
public:
  static char ID;

  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Register Allocator Analyzer"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  struct RegisterStats {
    unsigned MaxPressure = 0;
    unsigned CurrentPressure = 0;
    unsigned SpillCount = 0;
    unsigned TotalPhysRegs = 0;
  };

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  LiveIntervals *LIS = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  void simulateChordalAllocation(MachineFunction &MF);
};

char RASSA::ID = 0;

} // end anonymous namespace

namespace llvm {
  void initializeRASSAPass(PassRegistry&);
  void initializeLiveIntervalsWrapperPassPass(PassRegistry&);
}

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

bool RASSA::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  simulateChordalAllocation(MF);
  return false;
}

void RASSA::simulateChordalAllocation(MachineFunction &MF) {
  std::map<const TargetRegisterClass *, RegisterStats> ClassStats;
  std::vector<const LiveInterval *> Intervals;

  // 1. Collect Intervals
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg)) continue; 

    const LiveInterval &LI = LIS->getInterval(Reg);
    Intervals.push_back(&LI);

    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if (ClassStats.find(RC) == ClassStats.end()) {
      BitVector Allocatable = TRI->getAllocatableSet(MF, RC);
      ClassStats[RC].TotalPhysRegs = Allocatable.count();
    }
  }

  // 2. Linear Scan Sort
  std::sort(Intervals.begin(), Intervals.end(),
            [](const LiveInterval *A, const LiveInterval *B) {
              return A->beginIndex() < B->beginIndex();
            });

  // 3. Simulate
  std::map<const TargetRegisterClass *, std::vector<const LiveInterval *>> ActiveIntervals;

  for (const LiveInterval *Current : Intervals) {
    Register Reg = Current->reg();
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    RegisterStats &Stats = ClassStats[RC];
    auto &ActiveList = ActiveIntervals[RC];

    // Remove expired
    auto It = std::remove_if(ActiveList.begin(), ActiveList.end(),
                             [&](const LiveInterval *Active) {
                               return Active->endIndex() <= Current->beginIndex();
                             });
    ActiveList.erase(It, ActiveList.end());

    // Add new
    ActiveList.push_back(Current);

    // Update Pressure
    Stats.CurrentPressure = ActiveList.size();
    if (Stats.CurrentPressure > Stats.MaxPressure) {
      Stats.MaxPressure = Stats.CurrentPressure;
    }

    // Spill Check
    if (Stats.CurrentPressure > Stats.TotalPhysRegs) {
      Stats.SpillCount++;
      ActiveList.pop_back(); 
    }
  }

  // 4. Print Machine-Readable Output
  // Format: @SSA_REPORT func=<name> class=<name> spills=<count> pressure=<count>
  for (auto &Pair : ClassStats) {
    const TargetRegisterClass *RC = Pair.first;
    RegisterStats &Stats = Pair.second;

    errs() << "@SSA_REPORT "
           << "func=" << MF.getName() << " "
           << "class=" << TRI->getRegClassName(RC) << " "
           << "spills=" << Stats.SpillCount << " "
           << "pressure=" << Stats.MaxPressure << "\n";
  }
  
  // Ensure the stream is flushed so Python sees it immediately
  errs().flush();
}

FunctionPass *llvm::createSSARegisterAllocator() {
  return new RASSA();
}

FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) {
  return new RASSA();
}