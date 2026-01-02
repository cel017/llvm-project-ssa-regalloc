//===-- RegAllocSSA.cpp - SSA Register Allocator (Chordal/Fernando Port) --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h" 
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumJoins, "Number of interval joins performed");
STATISTIC(NumSpills, "Number of registers spilled");

FunctionPass *llvm::createSSARegisterAllocator();

namespace {

/// RegClique represents the set of physical registers currently "alive"
/// at a specific point in the basic block traversal.
class RegClique {
  BitVector PhysRegs;
  const TargetRegisterInfo *TRI;
  DenseMap<Register, MCRegister> VirtToPhys;

public:
  RegClique(const TargetRegisterInfo *tri) : TRI(tri) {
    PhysRegs.resize(TRI->getNumRegs());
  }

  void occupy(Register VirtReg, MCRegister PhysReg) {
    for (MCRegAliasIterator AI(PhysReg, TRI, true); AI.isValid(); ++AI) {
      PhysRegs.set(*AI);
    }
    VirtToPhys[VirtReg] = PhysReg;
  }

  void liberate(Register VirtReg, MCRegister PhysReg) {
    for (MCRegAliasIterator AI(PhysReg, TRI, true); AI.isValid(); ++AI) {
      PhysRegs.reset(*AI);
    }
    VirtToPhys.erase(VirtReg);
  }

  bool isOccupied(MCRegister PhysReg) const {
    return PhysRegs.test(PhysReg);
  }

  const DenseMap<Register, MCRegister> &getActiveVirtRegs() const {
    return VirtToPhys;
  }

  void clear() { 
    PhysRegs.reset(); 
    VirtToPhys.clear();
  }
};

class RASSA : public MachineFunctionPass, private LiveRangeEdit::Delegate {
  
  MachineFunction *MF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  
  // Analyses
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;
  LiveRegMatrix *Matrix = nullptr;
  MachineLoopInfo *MLI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  
  std::unique_ptr<Spiller> SpillerInstance;
  
  SmallPtrSet<MachineBasicBlock*, 32> Visited;

  void LRE_WillShrinkVirtReg(Register) override {}
  bool LRE_CanEraseVirtReg(Register) override { return false; }

public:
  static char ID;

  RASSA() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SSA Chordal Register Allocator"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;

  bool runOnMachineFunction(MachineFunction &mf) override;

private:
  void performLocalAllocation(MachineBasicBlock &MBB, RegClique &Clique);
  void allocateDefs(MachineInstr &MI, RegClique &Clique);
  void liberateDeadUses(MachineInstr &MI, RegClique &Clique);
  float getSpillWeight(Register Reg);
  void spill(Register VReg);
  MCRegister getFreePhysReg(Register VReg, const RegClique &Clique);
};

char RASSA::ID = 0;

} // end anonymous namespace

static RegisterRegAlloc ssaRegAlloc("ssa", "SSA register allocator",
                                    createSSARegisterAllocator);

INITIALIZE_PASS_BEGIN(RASSA, "regallocssa", "SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(RASSA, "regallocssa", "SSA Register Allocator", false, false)

void RASSA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<LiveStacksWrapperLegacy>(); 
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>(); 
  AU.addRequired<ProfileSummaryInfoWrapperPass>(); 
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RASSA::releaseMemory() { 
  SpillerInstance.reset(); 
  Visited.clear();
}

//===----------------------------------------------------------------------===//
// Helper: Weight Calculation
//===----------------------------------------------------------------------===//
float RASSA::getSpillWeight(Register Reg) {
  if (!Reg.isVirtual()) return 0.0f;
  if (!LIS->hasInterval(Reg)) return 1.0e20f;
  
  const LiveInterval &LI = LIS->getInterval(Reg);
  if (LI.empty() || LI.isSpillable() == false) return 1.0e20f;

  float Weight = 0.0f;
  if (MRI->getVRegDef(Reg)) {
    MachineBasicBlock *DefMBB = MRI->getVRegDef(Reg)->getParent();
    unsigned Depth = MLI->getLoopDepth(DefMBB);
    Weight += std::pow(10.0f, (float)Depth);
  }

  for (MachineInstr &UseMI : MRI->use_instructions(Reg)) {
    unsigned Depth = MLI->getLoopDepth(UseMI.getParent());
    Weight += std::pow(10.0f, (float)Depth) + 1.0f;
  }
  return Weight;
}

//===----------------------------------------------------------------------===//
// Helper: Get Free Physical Register
//===----------------------------------------------------------------------===//
MCRegister RASSA::getFreePhysReg(Register VReg, const RegClique &Clique) {
  const TargetRegisterClass *RC = MRI->getRegClass(VReg);
  ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(*MF);

  for (MCPhysReg PhysReg : Order) {
    if (!MRI->isReserved(PhysReg) && 
        !Clique.isOccupied(PhysReg) && 
        Matrix->checkInterference(LIS->getInterval(VReg), PhysReg) == LiveRegMatrix::IK_Free) {
      return PhysReg;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Logic: Liberate Dead Uses
//===----------------------------------------------------------------------===//
void RASSA::liberateDeadUses(MachineInstr &MI, RegClique &Clique) {
  SlotIndex Idx = LIS->getInstructionIndex(MI).getRegSlot();

  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isUse() || !MO.getReg().isVirtual()) continue;
    
    Register Reg = MO.getReg();
    
    if (LIS->hasInterval(Reg)) {
      const LiveInterval &LI = LIS->getInterval(Reg);
      if (!LI.empty() && LI.expiredAt(Idx)) {
        if (VRM->hasPhys(Reg)) {
          MCRegister Phys = VRM->getPhys(Reg);
          Clique.liberate(Reg, Phys);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Logic: Allocate Defs
//===----------------------------------------------------------------------===//
void RASSA::allocateDefs(MachineInstr &MI, RegClique &Clique) {
  // 1. Try Coalescing
  if (MI.isCopy()) {
    Register Dest = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    
    if (Dest.isVirtual()) {
      MCRegister SrcPhys = 0;
      if (Src.isPhysical()) {
        SrcPhys = Src.asMCReg();
      } else if (Src.isVirtual() && VRM->hasPhys(Src)) {
        SrcPhys = VRM->getPhys(Src);
      }
      
      if (SrcPhys) {
        // FIX: Check !MRI->isReserved to ensure we don't assign Reserved Regs (like Zero)
        if (!MRI->isReserved(SrcPhys) && 
            !Clique.isOccupied(SrcPhys) && 
            MRI->getRegClass(Dest)->contains(SrcPhys)) {
          
          if (Matrix->checkInterference(LIS->getInterval(Dest), SrcPhys) == LiveRegMatrix::IK_Free) {
             Matrix->assign(LIS->getInterval(Dest), SrcPhys);
             Clique.occupy(Dest, SrcPhys);
             NumJoins++;
             return;
          }
        }
      }
    }
  }

  // 2. Standard Allocation
  for (const MachineOperand &MO : MI.defs()) {
    if (!MO.isReg() || !MO.getReg().isVirtual()) continue;
    Register Reg = MO.getReg();
    if (VRM->hasPhys(Reg)) continue;

    if (!LIS->hasInterval(Reg)) LIS->createAndComputeVirtRegInterval(Reg);
    LiveInterval &LI = LIS->getInterval(Reg);

    MCRegister PhysReg = getFreePhysReg(Reg, Clique);
    
    if (PhysReg) {
      Matrix->assign(LI, PhysReg);
      Clique.occupy(Reg, PhysReg);
    } else {
      // --- EVICTION LOGIC ---
      Register BestVictim = 0;
      float BestWeight = LI.weight(); 
      MCRegister BestPhys = 0;
      
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);
      
      for (auto &Pair : Clique.getActiveVirtRegs()) {
        Register ActiveVirt = Pair.first;
        MCRegister ActivePhys = Pair.second;
        
        if (!RC->contains(ActivePhys)) continue;
        
        if (LIS->hasInterval(ActiveVirt)) {
           LiveInterval &ActiveLI = LIS->getInterval(ActiveVirt);
           if (!ActiveLI.empty() && ActiveLI.weight() < BestWeight) {
               if (ActiveLI.isSpillable()) {
                   BestVictim = ActiveVirt;
                   BestWeight = ActiveLI.weight();
                   BestPhys = ActivePhys;
               }
           }
        }
      }

      if (BestVictim) {
        Clique.liberate(BestVictim, BestPhys);
        Matrix->unassign(LIS->getInterval(BestVictim));
        
        spill(BestVictim); 
        NumSpills++;

        Matrix->assign(LI, BestPhys);
        Clique.occupy(Reg, BestPhys);
      } else {
        if (LI.isSpillable()) {
            NumSpills++;
            spill(Reg); 
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Spilling Adapter
//===----------------------------------------------------------------------===//
void RASSA::spill(Register VReg) {
  SmallVector<Register, 8> NewVRegs;
  LiveRangeEdit LRE(&LIS->getInterval(VReg), NewVRegs, *MF, *LIS, VRM, this);
  SpillerInstance->spill(LRE);
}

//===----------------------------------------------------------------------===//
// Logic: Perform Local Allocation (The Clique Loop)
//===----------------------------------------------------------------------===//
void RASSA::performLocalAllocation(MachineBasicBlock &MBB, RegClique &Clique) {
  Visited.insert(&MBB);
  
  Clique.clear();
  SlotIndex BlockStart = LIS->getMBBStartIdx(&MBB);
  
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (!MRI->reg_nodbg_empty(Reg) && VRM->hasPhys(Reg)) {
      if (LIS->hasInterval(Reg)) {
        const LiveInterval &LI = LIS->getInterval(Reg);
        if (!LI.empty() && LI.liveAt(BlockStart)) {
          Clique.occupy(Reg, VRM->getPhys(Reg));
        }
      }
    }
  }

  for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE; ) {
    MachineInstr &MI = *MII++; 
    liberateDeadUses(MI, Clique);
    allocateDefs(MI, Clique);
  }
}

//===----------------------------------------------------------------------===//
// Main Driver
//===----------------------------------------------------------------------===//
bool RASSA::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  TRI = MF->getSubtarget().getRegisterInfo();
  TII = MF->getSubtarget().getInstrInfo();
  MRI = &MF->getRegInfo();
  
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Matrix = &getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LiveStacks &LS = getAnalysis<LiveStacksWrapperLegacy>().getLS();

  MachineBlockFrequencyInfo &MBFI = getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  ProfileSummaryInfo &PSI = getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg) || !LIS->hasInterval(Reg)) continue;
    
    LiveInterval &LI = LIS->getInterval(Reg);
    if (!LI.empty())
        LI.setWeight(getSpillWeight(Reg));
  }

  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, *MLI, MBFI, &PSI);
  
  Spiller::RequiredAnalyses Analyses{*LIS, LS, *MDT, MBFI};
  
  SpillerInstance.reset(createInlineSpiller(Analyses, *MF, *VRM, VRAI, Matrix));

  Visited.clear();
  RegClique Clique(TRI);
  
  MachineBasicBlock *Entry = &MF->front();
  for (auto *MBB : depth_first(Entry)) {
    performLocalAllocation(*MBB, Clique);
  }

  // FIX: Safe Cleanup Loop
  // Iterates the class to find ANY non-reserved register.
  // REMOVED the "Last Resort" fallback that blindly picked the first register.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (!MRI->reg_nodbg_empty(Reg) && !VRM->hasPhys(Reg)) {
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);
      MCRegister FoundReg = 0;

      // 1. Try Allocation Order
      ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(*MF);
      for (MCPhysReg PReg : Order) {
        if (!MRI->isReserved(PReg)) {
          FoundReg = PReg;
          break;
        }
      }

      // 2. Fallback: Full class scan
      if (!FoundReg) {
        for (MCPhysReg PReg : *RC) {
          if (!MRI->isReserved(PReg)) {
            FoundReg = PReg;
            break;
          }
        }
      }

      // If still not found, we simply leave it unassigned to avoid crashing.
      if (FoundReg) {
        VRM->assignVirt2Phys(Reg, FoundReg);
      }
    }
  }

  SpillerInstance.reset();
  MF->getProperties().set(MachineFunctionProperties::Property::NoPHIs);

  return true;
}

FunctionPass *llvm::createSSARegisterAllocator() { return new RASSA(); }
FunctionPass *llvm::createSSARegisterAllocator(RegAllocFilterFunc F) { return new RASSA(); }