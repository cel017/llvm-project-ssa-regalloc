#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc-ssa"

STATISTIC(NumSpills, "Number of spills inserted");

namespace {

// Helper Class for spill weight
class SpillWeightCalculator {
    const MachineRegisterInfo &MRI;
    const MachineLoopInfo &MLI;

    // lookup table + prevent overflows
    static constexpr unsigned Pow10[] = { 
        1, 10, 100, 1000, 10000, 100000, 1000000, 10000000 
    };

    unsigned getLoopWeight(const MachineBasicBlock *MBB) const {
        unsigned Depth = MLI.getLoopDepth(MBB);
        if (Depth >= std::size(Pow10)) Depth = std::size(Pow10) - 1;
        return Pow10[Depth];
    }

public:
    SpillWeightCalculator(const MachineRegisterInfo &mri, const MachineLoopInfo &mli) 
        : MRI(mri), MLI(mli) {}

    unsigned getWeight(Register Reg) const {
        if (!Reg.isVirtual()) return 0;
        unsigned W = 0;
        MachineInstr *DefMI = MRI.getVRegDef(Reg);
        if (DefMI) W += 1 + getLoopWeight(DefMI->getParent());

        for (MachineInstr &UseMI : MRI.reg_nodbg_instructions(Reg)) {
            if (&UseMI == DefMI) continue;
            unsigned AddedW = 1 + getLoopWeight(UseMI.getParent());
            
            // prevent overflows
            if (W + AddedW < W) W = UINT32_MAX; else W += AddedW;
        }
        return W;
    }
};

class RegAllocSSA : public MachineFunctionPass {
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    LiveIntervals *LIS;
    VirtRegMap *VRM;
    MachineLoopInfo *MLI;
    
    // The "Coloring" Map: Virtual -> Physical
    std::map<Register, MCRegister> VRegToPhys;
    
    // Reverse Map for Eviction: Physical -> Virtual
    // We need this to know WHO is in Register X so we can spill them.
    std::map<MCRegister, Register> PhysToVReg;
    
    // Track current state of Physical Registers
    BitVector PhysRegsUsed;

public:
    static char ID;
    RegAllocSSA() : MachineFunctionPass(ID) {}

    StringRef getPassName() const override { return "SSA Chordal Register Allocator"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<LiveIntervalsWrapperPass>();
        AU.addRequired<VirtRegMapWrapperLegacy>();
        AU.addRequired<MachineLoopInfoWrapperPass>();
        AU.addRequired<MachineDominatorTreeWrapperPass>();
        AU.addPreserved<MachineDominatorTreeWrapperPass>();
        AU.addPreserved<MachineLoopInfoWrapperPass>();
        MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

private:
    void performLocalAllocation(MachineBasicBlock &MBB, SpillWeightCalculator &Weigher);
    
    // Phase 1: Reload spilled inputs
    void reloadSpilledUses(MachineInstr &MI, SpillWeightCalculator &Weigher);
    
    // Phase 2: Free dead registers
    void liberateDeadUses(MachineInstr &MI);
    
    // Phase 3: Assign registers to definitions
    void allocateDefs(MachineInstr &MI, SpillWeightCalculator &Weigher);
    
    MCRegister pickPhysReg(Register VReg, const TargetRegisterClass *RC);
    
    // The "Evict" function
    MCRegister evict(Register CurrentVReg, const TargetRegisterClass *RC, 
                     MachineInstr &MI, SpillWeightCalculator &Weigher);
};

char RegAllocSSA::ID = 0;

bool RegAllocSSA::runOnMachineFunction(MachineFunction &MF) {
    LLVM_DEBUG(dbgs() << "--- SSA Chordal Allocator: " << MF.getName() << " ---\n");

    MRI = &MF.getRegInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    TII = MF.getSubtarget().getInstrInfo();
    LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    
    PhysRegsUsed.resize(TRI->getNumRegs());
    VRegToPhys.clear();
    PhysToVReg.clear();

    SpillWeightCalculator Weigher(*MRI, *MLI);

    MachineDominatorTree &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
    
    for (auto *Node : depth_first(MDT.getRootNode())) {
        MachineBasicBlock *MBB = Node->getBlock();
        if (!MBB) continue;

        // Rebuild Clique at Block Entry
        PhysRegsUsed.reset();
        PhysToVReg.clear();
        
        SlotIndex BlockStart = LIS->getMBBStartIdx(MBB);
        
        for (auto It = VRegToPhys.begin(); It != VRegToPhys.end();) {
            Register VReg = It->first;
            MCRegister PReg = It->second;
            
            // Check if VReg is still live entering this block
            bool IsLive = false;
            if (LIS->hasInterval(VReg)) {
                if (LIS->getInterval(VReg).liveAt(BlockStart)) {
                    IsLive = true;
                }
            }

            if (IsLive) {
                // Mark occupied
                for (MCRegUnit Unit : TRI->regunits(PReg)) PhysRegsUsed.set(Unit);
                PhysToVReg[PReg] = VReg;
                ++It;
            } else {
                // dead mappings
                It = VRegToPhys.erase(It);
            }
        }

        performLocalAllocation(*MBB, Weigher);
    }

    // Rewrite Instructions
    // Note: We scan again because reloads created new VRegs that are in the map
    for (MachineBasicBlock &MBB : MF) {
        for (MachineInstr &MI : MBB) {
            for (MachineOperand &MO : MI.operands()) {
                if (MO.isReg() && MO.getReg().isVirtual()) {
                    Register VReg = MO.getReg();
                    if (VRegToPhys.count(VReg)) {
                        MO.setReg(VRegToPhys[VReg]);
                    }
                }
            }
        }
    }

    return true;
}

void RegAllocSSA::performLocalAllocation(MachineBasicBlock &MBB, SpillWeightCalculator &Weigher) {
    // We iterate via index because we might insert Reloads (instructions) during iteration
    for (auto MII = MBB.begin(); MII != MBB.end(); ) {
        MachineInstr &MI = *MII++; // Increment iterator before invalidating it
        if (MI.isPHI() || MI.isDebugInstr()) continue;

        // 1. Reload any inputs that are currently on the stack
        reloadSpilledUses(MI, Weigher);

        // 2. Mark registers that die here as free
        liberateDeadUses(MI);

        // 3. Assign output registers (Spilling/Evicting if necessary)
        allocateDefs(MI, Weigher);
    }
}

void RegAllocSSA::reloadSpilledUses(MachineInstr &MI, SpillWeightCalculator &Weigher) {
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual()) {
            Register VReg = MO.getReg();
            
            // If it's spilled (has stack slot and NO phys reg)
            if (VRM->hasStackSlot(VReg) && !VRegToPhys.count(VReg)) {
                int Slot = VRM->getStackSlot(VReg);
                const TargetRegisterClass *RC = MRI->getRegClass(VReg);
                
                // Create a temporary register for this reload
                Register ReloadReg = MRI->createVirtualRegister(RC);
                
                // Insert LOAD before the instruction
                TII->loadRegFromStackSlot(*MI.getParent(), MI, ReloadReg, Slot, RC, TRI);
                NumSpills++; // Count reloads as spills for stats
                
                // Update the operand to use the new temp register
                MO.setReg(ReloadReg);
                
                // Immediately allocate a physical register for this reload.
                // Since it's a Use, we treat it like a Def occurring "just before" the instr.
                // Note: We assume we can evict something because this reload range is tiny.
                MCRegister PReg = pickPhysReg(ReloadReg, RC);
                if (!PReg) {
                    // Forced Eviction: Reload is critical, kick out the lightest neighbor
                    PReg = evict(ReloadReg, RC, MI, Weigher);
                }
                
                VRegToPhys[ReloadReg] = PReg;
                PhysToVReg[PReg] = ReloadReg;
                for (MCRegUnit Unit : TRI->regunits(PReg)) PhysRegsUsed.set(Unit);
                
                LLVM_DEBUG(dbgs() << "  RELOADED " << VReg << " into " << TRI->getName(PReg) << "\n");
            }
        }
    }
}

void RegAllocSSA::liberateDeadUses(MachineInstr &MI) {
    SlotIndex Idx = LIS->getInstructionIndex(MI).getRegSlot();

    for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual()) {
            Register VReg = MO.getReg();
            
            // If mapped and dying
            if (VRegToPhys.count(VReg)) {
                if (LIS->getInterval(VReg).expiredAt(Idx)) {
                    MCRegister PReg = VRegToPhys[VReg];
                    
                    // Simple RefCount check (are we the only one using this PReg?)
                    // For prototype, we assume strict 1-to-1 mapping
                    for (MCRegUnit Unit : TRI->regunits(PReg)) PhysRegsUsed.reset(Unit);
                    PhysToVReg.erase(PReg);
                    
                    LLVM_DEBUG(dbgs() << "  Freed " << TRI->getName(PReg) << "\n");
                }
            }
        }
    }
}

void RegAllocSSA::allocateDefs(MachineInstr &MI, SpillWeightCalculator &Weigher) {
    for (MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
            Register VReg = MO.getReg();
            const TargetRegisterClass *RC = MRI->getRegClass(VReg);

            // Try to find free register
            MCRegister PReg = pickPhysReg(VReg, RC);

            // If none, Evict someone
            if (!PReg) {
                PReg = evict(VReg, RC, MI, Weigher);
            }

            // Assign
            VRegToPhys[VReg] = PReg;
            PhysToVReg[PReg] = VReg;
            for (MCRegUnit Unit : TRI->regunits(PReg)) PhysRegsUsed.set(Unit);
            
            LLVM_DEBUG(dbgs() << "  Def " << VReg << " -> " << TRI->getName(PReg) << "\n");
        }
    }
}

MCRegister RegAllocSSA::pickPhysReg(Register VReg, const TargetRegisterClass *RC) {
    ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(*MRI->getMF());
    for (MCPhysReg PReg : Order) {
        bool IsFree = true;
        for (MCRegUnit Unit : TRI->regunits(PReg)) {
            if (PhysRegsUsed.test(Unit)) { IsFree = false; break; }
        }
        if (IsFree && !MRI->isReserved(PReg)) return PReg;
    }
    return 0;
}

MCRegister RegAllocSSA::evict(Register CurrentVReg, const TargetRegisterClass *RC, 
                              MachineInstr &MI, SpillWeightCalculator &Weigher) {
    
    // 1. Find the best candidate to evict (Lowest Weight)
    // We only look at registers currently assigned to the target class
    ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(*MRI->getMF());
    
    MCRegister BestVictimPReg = 0;
    unsigned MinWeight = UINT32_MAX;
    
    // Simple heuristic: CurrentVReg weight
    unsigned CurrentWeight = Weigher.getWeight(CurrentVReg);

    for (MCPhysReg PReg : Order) {
        // Skip reserved regs (stack ptr)
        if (MRI->isReserved(PReg)) continue;
        
        // Who is here?
        if (PhysToVReg.count(PReg)) {
            Register VictimVReg = PhysToVReg[PReg];
            unsigned VictimWeight = Weigher.getWeight(VictimVReg);
            
            if (VictimWeight < MinWeight) {
                MinWeight = VictimWeight;
                BestVictimPReg = PReg;
            }
        }
    }
    
    // 2. Perform Eviction
    // Note: If MinWeight > CurrentWeight, we strictly shouldn't evict, 
    // but we HAVE to execution, so we evict anyway (local heuristic).
    if (BestVictimPReg != 0) {
        Register VictimVReg = PhysToVReg[BestVictimPReg];
        LLVM_DEBUG(dbgs() << "  EVICTING " << VictimVReg << " from " << TRI->getName(BestVictimPReg) 
                          << " for " << CurrentVReg << "\n");

        // A. Assign Stack Slot
        if (!VRM->hasStackSlot(VictimVReg)) VRM->assignVirt2StackSlot(VictimVReg);
        int Slot = VRM->getStackSlot(VictimVReg);
        
        // B. Insert Store Instruction
        // We store the physical register *after* the current instruction? 
        // NO. If we are evicting to make room for a DEF, we must store the OLD value 
        // *before* the DEF overwrites it.
        // If we are evicting for a Reload, we store before the Reload.
        // Basically: Save the old data NOW.
        TII->storeRegToStackSlot(*MI.getParent(), MI, BestVictimPReg, true, Slot, RC, TRI);
        NumSpills++;

        // C. Update Maps
        VRegToPhys.erase(VictimVReg);
        PhysToVReg.erase(BestVictimPReg);
        
        // D. Return the freed register
        return BestVictimPReg;
    }
    
    // Panic: No registers found (should happen only if RC is empty)
    report_fatal_error("RegAllocSSA: Run out of registers!");
    return 0;
}