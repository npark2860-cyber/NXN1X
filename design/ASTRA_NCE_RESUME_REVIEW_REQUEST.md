# Review request: NCE on Windows ARM64, exception / redirect / resume design

Date: 2026-11-20. Requested by NPARK. Scope: how a guest thread under NCE (native code execution) on Windows ARM64 enters guest code, takes exceptions, is redirected to the host, and is resumed. The focus is on the TEB stack bounds and on what runs inside the vectored exception handler (VEH).

The ask: review the design as a whole, not the latest patch. Five fixes in one day (tebfix2 → tebfix5) each covered one more resume path. tebfix5 replaced them with one rule. Is that rule complete? Which states still reach a resume or a dispatch with an SP that Windows will refuse?

## 1. Windows facts the design depends on

- **W1.** `RtlRestoreContext`, used when the VEH returns `EXCEPTION_CONTINUE_EXECUTION`, fast-fails with 0xd (ntdll+0x34a10) unless the SP is accepted. The rule (ntdll+0x34720):
  - accepted when `StackLimit <= SP <= StackBase`;
  - also accepted when `StackLimit - 0x1000 <= SP` and `StackLimit != DeallocationStack` (TEB+0x1478).
- **W2.** An exception raised while a handler runs (for example `OutputDebugStringW` → 0x4001000A, a C++ throw, `RaiseException`) is dispatched by walking frames. With SP outside the TEB bounds, no frame handler is found, and the exception is unhandled.
- **W3.** Windows dispatches a user exception on the faulting thread's current stack. Under NCE that is often the guest stack, so the VEH runs on the guest stack.
- **W4.** `NtContinue` (used by `ContinueContext`) does not apply W1's check the same way. The bridges publish their own bounds after it.
- **W5.** `SetThreadContext` on a suspended thread changes PC/SP/registers. It does not set physical x18 on all paths. If the target is inside kernel exception delivery, its own pending exception is delivered with the new context.

## 2. Threads, stacks and bounds

- Host thread (one per emulated core): host stack `[DeallocationStack, host top)`. At guest entry, `InstallGuestTebStackBounds` (`arm_nce_windows.cpp:143`) saves the host bounds into `GuestContext.windows_host_stack_base/limit` and computes the guest stack's allocation.
- Guest SP: a section-backed (MEM_MAPPED) view in the guest address space. Guest VA equals host VA. The guest may switch SP to other allocations inside one epoch (fibers, coroutines, game-allocated stacks).
- Physical x18 holds the TEB on host code. In guest code it may hold a guest value: x18 substitution windows in `patcher.cpp`, with trampolines that use TEB TLS slots.

## 3. State machine: where SP is and who publishes the TEB bounds

| State | SP | TEB bounds published by | Code |
|---|---|---|---|
| H: RunThread host C++ | host | host (saved at entry, restored at exit by `RestoreHostTebStackBounds`) | `arm_nce_windows.cpp:2124, 2158` |
| E1: asm entry | host → guest | guest pair stored right before `mov sp, x30` | `windows_nce_entry.asm:111-116` |
| E2: NtContinue entry | host → guest | `WindowsNceRestoreGuestContext` builds the CONTEXT; `WindowsNceGuestStackBridge` stores x1/x2 after NtContinue | `windows_nce_transition.cpp:~80-134`, `.asm:161-166` |
| G: guest code | guest (any allocation) | whatever was published last; stale after a fiber switch | — |
| X: patcher x18 direct path | guest | stores both bounds itself | `patcher.cpp` |
| R: redirect to host | guest → host | `RedirectToHost` sets the CONTEXT: PC=`WindowsNceHostStackBridge`, x1/x2=host bounds, x3=host SP, x4=target TEB, x16=host return PC. The bridge does `mov x18,x4`, stores the bounds, then `mov sp,x3` | `windows_nce_transition.cpp:139-195`, `.asm:168-175` |

## 4. Every resume / dispatch path

| # | Path | How it resumes | Bounds at resume |
|---|---|---|---|
| P1 | VEH, watched-page write fault (`NvnLayer::OnWatchedWrite`) | `CONTINUE_EXECUTION` → RtlRestoreContext (W1) | tebfix5 fit on the resumed SP |
| P2 | VEH, host-code read fault (`HandleNceReadFault`) and host `InvalidateNCE` | `CONTINUE_EXECUTION` | tebfix5 fit |
| P3 | VEH, inline guest fault (opt-in `EDEN_NCE_INLINE_FAULT`) | `CONTINUE_EXECUTION` | tebfix5 fit |
| P4 | VEH, guest access fault → `RedirectToHost` → `ContinueContext` | NtContinue to the bridge on the guest SP | the bridge publishes host bounds (x4 TEB) |
| P5 | VEH, BRK (x18 fallback trap / unknown breakpoint) → `RedirectToHost` → `ContinueContext` | NtContinue to the bridge | the bridge |
| P6 | VEH, cache-op emulation (`IC IVAU` / `DC *`) | `ContinueContext` straight to guest PC+4 | unchanged (guest bounds as they stood) |
| P7 | Cross-thread break: Suspend → GetThreadContext → transform (`RedirectToHost`, `arm_nce_windows.cpp:750, 851`, TEB = `m_windows_break->Teb()`) → SetThreadContext → Resume | the target resumes at the bridge | the bridge |
| P8 | P7 landing while the target is inside its own exception delivery (W5) | the target's VEH sees its fault with PC = bridge entry and a guest SP | tebfix4, now tebfix5's fit on SP |
| P9 | Any exception raised inside the VEH or its callees (logger, CRT) | frame dispatch (W2) | tebfix5 fits the bounds to the handler's own SP at entry |

## 5. Invariants claimed by tebfix5 (please check)

- **I1.** Before any `EXCEPTION_CONTINUE_EXECUTION`, the TEB bounds accept the resumed SP.
  - Already accepted: left as is.
  - On the thread's own host stack (`DeallocationStack < SP <= top of that reservation`): host bounds.
  - In guest memory (`IsValidVirtualAddressRange`): the allocation holding SP.
  - Otherwise: left as is and counted (never seen so far).
- **I2.** From the first instruction of the VEH wrapper, the bounds accept the handler's own SP. So W2 holds for anything raised inside it.
- **I3.** Nothing on the VEH path logs or raises. Diagnostics go to counters printed later by `LogTebFitCounters` from `PrepareRunThread` (host stack, outside the epoch).
- **I4.** Every path that leaves the guest stack for host code goes through `WindowsNceHostStackBridge`. It publishes host bounds from the target thread's own `GuestContext` and TEB (x4), never from physical x18 or another thread's binding.

Open questions for the reviewer:
- **Q1.** P6: `ContinueContext` from the VEH straight to guest code. Is NtContinue safe here when the bounds are stale? tebfix5 does not fit on this path; it fits only at entry, for the handler's SP.
- **Q2.** P7: the transform passes `m_windows_break->Teb()`. Is the binding always the suspended target thread? tebfix3's log showed the binding can be another thread's (TEB 0xb860b8a000 vs 0xb860b90000).
- **Q3.** P8: when the redirect lands during the target's own exception delivery, the guest store that faulted is not retired. The host continuation re-enters the guest at that PC. Is any guest state lost, for example a partial pair store, or an exclusive monitor lost?
- **Q4.** Is the fit on the handler's own SP at entry (I2) correct when that SP is deep in a guest stack whose allocation is a fiber carved from a heap block, so that the allocation is larger than the stack?
- **Q5.** Are there resume paths missing from section 4? For example: APCs, `SetThreadContext` from a debugger, the patcher's other fallbacks.
- **Q6.** `t_crash_depth` reaches 1400-1500. The NtContinue paths (P4-P6) never return through the wrapper, so its decrement is skipped. That is harmless as a counter, but please confirm no logic depends on it.

## 6. Crashes today, with evidence

| Time | Exe / mode / game | Code | Evidence | Cause | Fix |
|---|---|---|---|---|---|
| 15:42 | poolcopy self-test, copy | 0xd | crashring: 2 invalid-SP resumes; SP 0x20bec8eb10 vs TEB 0x208eaf4000..0x208eb04000 | P1-P3 on another guest stack (fiber) | tebfix2 |
| 16:17, 16:33, 16:42 | copy, 0400E88025B1A000 | 0xd | the fault was in the bridge at x18+8 on a protected pool page | the bridge stored host bounds through physical x18, which held a guest value | x18fix (wrong: used the binding's TEB; crashed again at 16:49) → tebfix3 |
| 16:54 | tebfix3, copy, StarFox | 0xd | AV 0x21769a001c reported at the bridge's first instruction; SP on another guest stack | P8 | tebfix4 |
| 17:06 | tebfix4, copy, StarFox | 0x4001000A (KERNELBASE+0x11f144) | crashring: enter at the bridge (exe+0x1203e64), no return; TEB 0x210e65d000..0x210e66d000, SP 0x2171d1ef60; 0 tebfix4 lines in the log | tebfix4 logged before refitting (P9, W2) | tebfix5 |
| 17:18 | tebfix5, copy, StarFox | abort (ucrtbase+0x798ec, c0000409/7) | `FATAL NVN command memory` at 117 s: 65536 host-lent segments, 65489 without a guest range | not NCE (NVN command memory, `hle_nvn.c` s55); tebfix5 counters: 16384 entry fits, 0 left | pending |
| 16:21 (and 04:15, 09:24) | various | abort (ucrtbase+0x798ec) | no log kept | probably the same `die()` | — |

## 7. Code locations (current)

- `source/src/core/arm/nce/arm_nce_windows.cpp`:
  - `InstallGuestTebStackBounds` (143)
  - crash ring (~1325-1450)
  - tebfix5 block (`FitTebToSp`, `LogTebFitCounters`, the VEH wrapper, ~1452-1600)
  - `WindowsNceVectoredExceptionHandlerImpl` (~1600-1835)
  - cross-thread transforms (750, 851)
  - RunThread entry/exit (2124, 2158)
- `source/src/core/arm/nce/windows_nce_transition.cpp`: `ContinueContext`, `WindowsNceRestoreGuestContext`, `RedirectToHost` (139-195)
- `source/src/core/arm/nce/windows_nce_entry.asm`: the entry, `GuestStackBridge` and `HostStackBridge`
- `source/src/core/arm/nce/windows_cross_thread_break.h`: Suspend / Get / SetThreadContext / Resume (640-700)
- `source/src/core/arm/nce/windows_x18_fallback_trap.cpp`, `patcher.cpp`: x18 paths
- Backups of each step: `_backup/tebfix2`, `tebfix3`, `tebfix4`, `tebfix5`. BUILD_LOG.txt has one line per build.

## 8. What I would like back

1. Any resume or dispatch state not covered by I1-I4, with the sequence that reaches it.
2. Answers to Q1-Q6.
3. Whether the SP-only rule should replace the bridges' own publishing (one owner of the bounds), or whether the bridges should stay as they are.

## 9. 검수 반영 (2026-11-20 17:45)

이 절이 반영한 검수는 두 개다. 둘 다 "브리지는 유지한다. tebfix5는 개선이지만 재개 설계 전체의 완성은 아니다"로 결론이 같다.
- `ASTRA_NCE_RESUME_REVIEW.md` (17:41): 문서와 공식 자료를 대조한 검수
- `NCE_RESUME_REVIEW.md` (17:29): 소스와 로그를 대조한 검수

아래는 항목별 판정이다.

| # | 지적 | 판정 | 근거와 조치 |
|---|---|---|---|
| R1 | I1: 맞출 수 없는 SP(left)도 그대로 CONTINUE_EXECUTION | **수용** | `arm_nce_windows.cpp:1581-1600` wrapper. left는 실패로 다룬다. crash ring 기록 뒤 NCE 전용 코드로 fail-fast한다. 0xd보다 증거가 남는 정의된 종료다. |
| R2 | 브리지 중간 PC 반례: 경계를 게시한 뒤, SP를 바꾸기 전에 재개 | **수용** (조건부 상태) | `windows_nce_entry.asm:168-175` HostStackBridge, `:111-116` EnterGuest. 재개 PC가 "게시 ~ mov sp" 구간이면 구간 시작으로 되돌린다. HostStack은 `mov x18,x4`부터, EnterGuest는 `add x14`부터다. 인수 x1-x4, x16, x17은 이 구간에서 바뀌지 않으므로 재실행해도 안전하다. |
| R3 | P6: 캐시 명령 뒤 ContinueContext가 재개 SP 검사 없이 직행 | **수용** | `arm_nce_windows.cpp:1737` 부근. VEH의 모든 직접 재개(P4-P6)와 wrapper 재개는 공통 `ResumeChecked(context)` 하나로 보낸다. 이 함수는 R1의 판정과 R2의 정규화를 함께 한다. |
| R4 | P7: 바인딩이 새 handle을 먼저 공개하고 TEB는 나중에 저장 | **수용** | `windows_cross_thread_break.h:586-606`. 587에서 handle을 store하고 602에서 TEB를 store하므로 그 사이가 race 구간이다. handle, TEB, host 경계, epoch를 한 레코드로 묶어 포인터 하나로 교체한다. transform은 그 레코드를 한 번만 읽는다. `RedirectToHost`의 바인딩 fallback(`windows_nce_transition.cpp:186-187`)은 레코드가 일치할 때만 쓴다. |
| R5 | VEH 하위 경로에 남은 로그와 첫 사용 초기화 | **수용** | `nvn_dirty.c:59-60` busy_wait의 RED LOG(fault 경로). `arm_nce_windows.cpp:1791-1795`의 getenv와 함수 내부 static. 앞의 것은 고정 레코드에 남기고 호스트 스택에서 출력한다. 뒤의 것은 VEH 설치 때 미리 읽어 둔다. `InvalidateNCE`의 unmapped LOG(`memory.cpp:1305`)도 host-fault 경로에서 쓰는 변형으로 분리한다. |
| R6 | `&here`는 실제 SP가 아니다 | **부분 수용** | `arm_nce_windows.cpp:1583`. 같은 프레임의 지역 변수라 같은 allocation 안에 있으므로, allocation 단위 판정 결과는 같다. 그래도 정확성을 위해 `mov x, sp`로 읽은 실제 SP를 쓴다. |
| R7 | allocation ≠ 논리 fiber 스택. 맞춘 StackLimit으로 남은 공간(headroom)을 과대평가 | **수용** | `arm_nce_windows.cpp:1622`(16 KB), 1790대(64 KB). 이 headroom 검사는 맞춘 StackLimit을 기준으로 삼지 않는다. 알려진 guest 스택 상태(`FindGuestStackState`)로 판정하고, 모르면 최소 경로를 쓴다. |
| R8 | SVC 트램펄린과 직접 x18 helper가 저장해 둔 `windows_guest_stack_*`를 다시 게시한다(fiber 전환 뒤 낡은 범위) | **수용** | `patcher.cpp`(검수 인용 923, 999, 1732). 다시 게시하기 전에 현재 guest SP가 저장된 범위 안인지 확인한다. 범위 밖이면 검증된 복귀 경로로 보낸다. |
| R9 | 다른 VEH(S36 `host_memory.cpp:203`, hwwatch, posix compat)가 NCE wrapper 전후에 직접 CONTINUE_EXECUTION | **수용** | NCE 스레드에서 재개하는 VEH는 R3의 공통 판정을 함께 쓴다. 활성 빌드의 등록 순서를 표로 남긴다. |
| R10 | W5/ContextFlags: `CONTEXT_FULL`에는 x18이 없다 | **부분 수용** | 브리지 경로는 CONTEXT의 x18에 의존하지 않는다(`windows_nce_entry.asm:169` `mov x18,x4`). "NtContinue 경로가 x18을 복원한다"는 `windows_nce_transition.cpp:191`의 주석은 틀렸으므로 고친다. P6에서 `context.X[18]=TEB`를 대입하는 부분은 flags에 따라 효과가 없을 수 있다. R3에서 flags를 명시한다. |
| R11 | W1 두 번째 조건에 `SP <= StackBase` 누락 | **반박** (문서 표기 오류) | 코드는 두 조건 모두 상한을 둔다(`arm_nce_windows.cpp:1477-1479` `TebAcceptsSp`). 요청서 1절의 문구만 고친다. |
| R12 | SP == StackBase이면 다음 allocation을 고를 수 있다 | **반박** | guest는 `AllocationOf(sp - 1)`(`:1523`)로 sp-1을 조회한다. host는 `sp > dealloc && sp <= host_top`로 판정한다. 둘 다 상단 경계를 포함하므로 다음 allocation을 고르지 않는다. |
| R13 | I2: 자체 VEH보다 먼저 실행되는 구간(OS dispatch, 앞선 handler, prologue) | **수용** (범위 명시) | 이 구간은 우리 코드로 보호할 수 없다. 불변식 문구를 "wrapper 첫 문장부터"로 한정한다. R2와 R8이 낡은 경계 상태 자체를 줄인다. |
| R14 | Q3/P8: 원래 PC로 다시 들어가는 것만으로 상태 보존이 입증되지 않는다(다중 접근 STP, exclusive) | **수용** (테스트 요구) | 코드 변경은 없다. 합성 테스트를 추가한다: STP 페이지 경계 fault 뒤 재실행 결과, LDXR/STXR 중단 뒤 STXR 실패와 guest 재시도. |
| R15 | Q6: `t_crash_depth`는 활성 깊이가 아니다 | **수용** | 제어 흐름에는 쓰지 않는다(`:1338`, `:1423`, `:1588`에서만 사용, 두 검수 모두 확인). 누적 진입 수, 활성 깊이, 직접 재개 수로 나눈다. |
| R16 | 크래시 표의 원인 단정 | **수용** | 17:18은 "직접 종료 지점은 NVN command memory fatal, NCE와의 인과는 미확인"으로 고친다. 16:21, 04:15, 09:24는 "원인 미확인"으로 고친다. |
| R17 | APC, debugger, 다른 handler 경로 | **부분 수용** (범위 제외 명시) | debugger를 붙인 실행은 지원 범위 밖으로 둔다. special user APC를 NXN1X가 쓰는지는 확인 대상이다. 드라이버나 Qt가 쓸 가능성이 있으면 R2의 정규화가 이 경우도 덮는다. |

### 코드 변경 순서 (승인 후)

1. `ResumeChecked`: R1, R2, R3, R6, R9, R10을 묶은 공통 재개 판정
2. 바인딩 레코드 일원화 (R4)
3. VEH 하위 경로의 로그와 초기화 정리 (R5)
4. headroom을 알려진 스택 기준으로 판정 (R7)
5. 직접 x18/SVC 경로의 guest 경계 재검증 (R8)
6. 카운터 분리 (R15)
7. 합성 테스트: P7을 단계별로 중단, 브리지 중간 주입, STP/exclusive (R14, 검수 7.2)

### NPARK가 결정할 것
- 순서: 이 NCE 정리를 먼저 할지, GPU 메모리 캐시 3단계(직렬화 해제)를 먼저 할지
- R1: 맞출 수 없는 재개를 명시적 fail-fast로 바꾸는 데 동의하는지. 지금은 0xd로 죽을 수도, 운 좋게 넘어갈 수도 있다.
