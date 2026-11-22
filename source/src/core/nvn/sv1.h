// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// sv1 (lane A): log-only guest thread dumps when the NVN path stops presenting (docs/lane_a/apply_sv1.py)
#pragma once

#include <string_view>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Core::Sv1 {

// nso.cpp: a module loaded into the process (name, load base, image size)
void NoteModule(std::string_view name, u64 base, u64 size);
// nvn_output.cpp Push: the game presented
void NotePresent();
// nvn_layer.cpp SlowGateCall: starts the watchdog once (NXN1X_SV1=0: off)
void StartWatchdog(Core::System& system);
// combined1120: stops and joins the watchdog threads (System shutdown: they hold a Core::System*)
void StopWatchdog();
// One dump of every guest thread of the application process, to the log
void DumpGuestThreads(Core::System& system, const char* why);

// wakediag (NXN1X_HWA_POOLDIAG=1, log only): k_condition_variable.cpp records condition variable traffic into a ring that pooldiag
// summarizes after a slot-48 pool burst. Recording takes no lock and logs nothing (it runs under the scheduler lock).
enum class CvEvent : u32 {
    Signal = 1,      // a = requested count (<= 0: all), b = threads taken off the key
    Woken = 2,       // tid = the thread made runnable by a signal (the mutex was free)
    ToMutex = 3,     // tid = the thread moved to the mutex owner's waiters (it runs when the owner unlocks)
    WaitBegin = 4,   // tid = the thread starts waiting on the key
    WaitEnd = 5,     // tid = the thread is back from Wait (a = its result's raw value)
};
bool WakeTraceOn();
void NoteCv(CvEvent kind, u64 key, u64 tid, s32 a, s32 b);

// cpudiag (NXN1X_THREADCPU=1, log only): KScheduler::SwitchThread passes each switched-out thread's time on its core (the ticks it also adds with
// KThread::AddCpuTime) into a table without locks; the host layer's "threads up to frame" line (hle_nvn.c dm_thread_cpu) prints it per window
bool ThreadCpuOn();
void NoteThreadTime(u64 tid, s32 prio, s32 core, s64 ticks, bool idle);
// thrname (NPARK 11-21: which thread t171 is): svc CreateThread hands each new thread's entry point; the line names its top threads (the nnsdk name in the
// thread's TLS ThreadType, its entry as module+offset and where it is now)
void NoteThreadEntry(u64 tid, u64 entry);

} // namespace Core::Sv1
