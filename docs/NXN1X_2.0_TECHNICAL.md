# NXN1X 2.0 기술 문서

작성일: 2026-11-22. 대상: NXN1X 첫 배포(2026-11-22 빌드, `build\release\bin\NXN1X.exe`).

이 문서는 저장소 안의 문서·설계서·소스 주석과 프로젝트 메모리 노트에서 확인한 내용만 정리한다. 각 항목 끝의 "출처"는 그 내용을 확인한 파일이다. 경로는 저장소 루트(`nxn1x\`) 기준이다. 문서끼리 맞지 않거나 근거가 없는 부분은 본문에 "불일치" 또는 "미확인"으로 적었다. 전체 목록은 8.3절에 있다.

용어:

| 용어 | 뜻 |
|---|---|
| NCE | Native Code Execution. 게스트 ARM64 코드를 호스트 CPU에서 직접 실행하는 Eden의 CPU 백엔드 |
| NVN / NVN2 | Switch 1 / Switch 2의 그래픽 API |
| NVN 호스트 레이어 | 게임의 NVN 호출을 HLE로 받아 Vulkan으로 재생하는 `host\dm` 코드. nxnative 계열에서 가져옴 |
| Emulation 경로 | Eden 원래의 Maxwell GPU 에뮬레이션 |
| import / copy / hybrid | 게임 메모리 풀을 Vulkan에 보이는 방식. 4.4절 |
| replay | 기록된 NVN 명령을 Vulkan 명령 버퍼로 다시 기록하는 단계 |
| HWAoI | Hyrule Warriors: Age of Imprisonment |
| OoT | Ocarina of Time (Switch 2판, 0400486024DEC000) |
| DK | Donkey Kong Bananza (040074A01BF12000) |
| XCDE / XC2 | Xenoblade Chronicles DE / Xenoblade Chronicles 2 (Switch 2 Edition) |

---

## 1. 개요

### 1.1 NXN1X란

NXN1X 2.0은 Eden(yuzu 계열)을 기반으로 한 Nintendo Switch / Switch 2 에뮬레이터다. 대상은 Windows ARM64이고, 현재 기준 하드웨어는 NVIDIA RTX Spark N1X 노트북이다.

- 버전: 2.0 (창 제목 "NXN1X 2.0", `source\src\yuzu\yuzu.rc` FileDescription/ProductName).
- 뼈대: **SnapRyu 2.0**. NXN1X 2.0은 SnapRyu 2.0을 뼈대로 만들었다. SnapRyu 2.0은 이름만 남았고 Ryujinx 기반이 아니다. Go 모드(폰 컨트롤러: 페어링·입력·진동·원격 플레이, `nxn1x_go_host.dll` + `source\src\common\nxn1x_go.cpp`)도 SnapRyu 2.0에서 왔다. 2.0에서는 Go 모드를 숨겨 두었다(`NXN1X_GO_MODE=1`로 표시). 앞으로 Go는 SnapRyu 2.0을 원본으로 두고 NXN1X는 그 DLL과 API 헤더만 받아 쓴다.

- GPU: Blackwell 계열, PCI 10de:2e03, 드라이버 621.24, Vulkan 1.4.357. Windows 11 ARM64 28000.3501, WDDM 3.2, 하드웨어 스케줄링 켜짐.
- CPU: P코어 10개 + E코어 10개의 혼합 구성. VRAM 예산 약 80 GB.
- 실행 파일: `build\release\bin\NXN1X.exe`. 진단 계측은 CMake 옵션 `NXN1X_DIAGNOSTICS`(기본 OFF)로 구분한다.
- 라이선스: Eden/yuzu 파생. 원저작자 표시와 GPL 라이선스를 유지한다.

출처: `docs\DLSS_ARM64_CAPABILITIES.md`, 메모리 노트 `host-tuned-for-adreno.md`, `source\CMakeLists.txt`, `source\README.md`.

### 1.2 Adreno 시기 (기록)

NXN1X는 원래 Qualcomm Snapdragon X Elite(Adreno GPU, Turnip/Mesa 드라이버, 균일 코어 12개)용 프로젝트였다.

- SnapRyu(당시 X Elite 프로젝트)를 복사해 시작했다. 지금의 뼈대는 SnapRyu 2.0이다(1.1 참고).
- 2026-11부터 N1X에서 실행한다. N1X로 옮긴 직후에는 NCE가 전혀 동작하지 않았고, 수정 후에도 성능은 처음부터 낮았다(나중에 생긴 회귀가 아님).
- 호스트 레이어(`host\dm\windows`)는 Adreno/Turnip에서 튜닝됐다. N1X에서 비용이나 버그가 나오면 Adreno 시기 기본값인지 먼저 확인한다. Adreno 호환은 더 이상 유지 대상이 아니다.

Adreno 시기 문서와 흔적:
- `source\README.md`: "Windows ARM64와 Qualcomm Adreno 환경" 대상의 옛 계획(제목은 2.0으로 맞춤, 내용은 Adreno 시기). NPU 프레임 생성(QNN/ORT), TotK 1080p 고정, `tools/build.ps1` 등.
- `README.md`(루트): Turnip/Mesa 소스, F1R/F2R 드라이버 오버레이, LSFG 통합을 담은 소스 스냅샷 설명.
- `docs\nvn-memory\`: 2026-11-16~18의 메모리 기반 작업(F2~F5Z 단계). 12 GB급 장치와 Turnip 전제. 대부분 "runtime pending" 상태로 끝났다.
- 코드에 남은 흔적: Switch 2 부팅 때 `TU_GS_ONCE=1`(Turnip 전용 환경 변수)을 설정한다(`source\src\core\nvn\nvn_layer.cpp` `OnAppLoad`). `source\src\video_core\renderer_vulkan\vk_npu_frame_gen.*`도 남아 있다.

출처: 메모리 노트 `host-tuned-for-adreno.md`, `source\README.md`, `README.md`, `docs\nvn-memory\FOUNDATION_DESIGN.txt`, `docs\nvn-memory\IMPLEMENTATION_PLAN.json`.

### 1.3 GPU 경로: Emulation과 NVN 호스트 레이어

NCE로 실행하는 게임의 GPU 경로는 설정 `nce_gpu_path`(Cpu 분류)로 고른다. 부팅 때 한 번 읽는다.

| 값 | 동작 |
|---|---|
| `Emulation` (코드 기본값) | Eden의 Maxwell GPU 에뮬레이션 |
| `Nvn` | 게임의 NVN 호출을 NVN 호스트 레이어(`source\src\nvn_host` → `host\dm`)로 보낸다 |

`Nvn`을 골라도 다음 경우는 Emulation으로 돌아가고 로그 한 줄을 남긴다(`core\nvn\nvn_layer.cpp` `OnAppLoad`, `Refuse`).
- NCE가 그 게임을 실행하지 않음
- 렌더러가 Vulkan이 아님
- 같은 프로세스에서 NVN 레이어가 이미 한 번 실행됨(NXN1X를 다시 시작해야 함)

Switch 1과 Switch 2의 차이:
- **Switch 1 (NVN1)**: 로그 `GPU path: NVN host layer`.
- **Switch 2 (NVN2)**: 로그 `GPU path: NVN host layer (Switch 2, NVN2)`. 게이트가 sdk 모듈의 NVN2 상수를 읽는다(`nvn_gate.cpp` ReadNvn2). 셰이더 브리지가 Ampere 코드를 Maxwell로 낮춘다(4.3절). 환경에 값이 없으면 `TU_GS_ONCE=1`, `DM_VK_TBUF_FRESH=1`을 설정하고 `NXN1X_NVN2_HOST=1`을 둔다.
- Switch 2 프로그램의 기본 해상도는 1x다(`ns2res1x`, `common\settings.cpp` `Switch2Res1x()`). 게임별 설정이 있으면 그것을 따르고, `NXN1X_NS2_RES_1X=0`이면 예전처럼 전역 설정을 쓴다.

미확인: 배포판 프로필에서 `nce_gpu_path`가 실제로 `Nvn`인지는 소스만으로 알 수 없다. 코드 기본값은 `Emulation`이다(`common\settings.h`).

출처: `source\src\common\settings.h`, `source\src\common\settings.cpp`, `source\src\core\nvn\nvn_layer.cpp`, `docs\RELEASE_2026-11-22.md` 3절.

---

## 2. 아키텍처

### 2.1 전체 구조

```
게스트 코드 (NCE, 호스트 CPU에서 직접 실행)
  │  nvnBootstrapLoader 훅 → H 게이트 (core\nvn\nvn_gate.cpp)
  ▼
NVN 호스트 레이어 (host\dm\gfx\nvn\hle_nvn.c)
  │  의미 있는 호출은 실제 객체로 구현, GPU 상태·draw는 명령으로 기록
  ▼
Vulkan replay (host\dm\windows\gfx_vk_exec.c, gfx_vk_eden.inc, gfx_vk_core.c, gfx_vk_tex.c)
  │  Eden 파이프라인 사용 (video_core\nvn 브리지 → Eden shader recompiler)
  ▼
present (gfx_vk_present.c) → core\nvn\nvn_output.cpp → Eden 컴포지터 (nvn_display.cpp)
  │  DLSS SR / DLSS-G (Streamline), 광학 흐름 프레임 보간 (fsr3_frame_gen)
  ▼
Vulkan 스왑체인
```

NVN 호스트 레이어는 한 C 정적 라이브러리 `nvn_host`로 빌드한다(`source\src\nvn_host\CMakeLists.txt`). 호스트 트리 자체의 빌드 방식(`dm\windows\build.py`의 GNU C, blocks, Darwin 호환 헤더)을 그대로 쓴다. 포함하는 파일은 다음과 같다.
- `gfx/nvn/hle_nvn.c`, `nvn_dirty.c`, `nvn_tex.c`, `nvn_capture.c`, `nvn_entry.S`, `nvn_thunks.S`
- `windows/gfx_vk_core.c`, `gfx_vk_exec.c`, `gfx_vk_present.c`, `gfx_vk_shader.c`, `gfx_vk_tex.c`, `gfx_vk_win_glue.c`, `nvn_memory_*.c`
- `nxn1x/nvnh_rt.c`, `nvnh_posix.c`, `nvnh_memory_guard.cpp`

`dm/host`(호스트 자체 프로세스용 커널·서비스), `posix_win.c`, 호스트의 창은 빌드하지 않는다. 스레드, 모듈 로드, 예외, 창, 성능 패널은 NXN1X(Eden)가 맡는다. `nvnh_rt.c`가 그 대체 함수를 둔다.

출처: `source\src\nvn_host\CMakeLists.txt`, `host\dm\nxn1x\nvnh_rt.c` 머리 주석.

### 2.2 CPU: Windows NCE

#### 2.2.1 구성

NCE 폴더는 `source\src\core\arm\nce\`다. Windows 빌드는 다음을 컴파일한다(`core\CMakeLists.txt` 약 1304-1335행).
- `arm_nce_windows.cpp`, `windows_*`, `x18_*`, `windows_nce_entry.asm`(armasm64), `core\hle\service\ro\ro_nce_windows.cpp`

`arm_nce.cpp`, `arm_nce.s`, `interpreter_visitor.cpp`는 비Windows(Linux/Android) 경로에서만 컴파일한다.

규모(2026-11 점검 시점):

| 항목 | 값 |
|---|---|
| NCE 폴더 전체 | 39개 파일, 15,864줄 |
| Windows 전용 | 약 8,410줄 |
| Linux/Android 전용(Eden 원본) | 4,366줄 |
| 공유(patcher 등) | 3,086줄 |
| `ArmNce::RunThread`(Windows) | 693줄(Eden 원본 59줄) |

출처: `design\NCE_CLEANUP_REVIEW.md` 0절, 5절.

#### 2.2.2 Windows에서 필요한 처리

| 관심사 | Windows 구현 |
|---|---|
| 예외 진입 | VEH(`AddVectoredExceptionHandler`). 예외는 fault가 난 스레드의 현재 스택, 즉 게스트 스택에서 처리된다 |
| fault 뒤 게스트 탈출 | `RedirectToHost` + `NtContinue`로 `WindowsNceHostStackBridge`에 들어감 |
| 게스트 진입 | `WindowsNceEnterGuest` / `EnterGuestContext`(armasm64) |
| x18 | Windows에서는 TEB 포인터라서 게스트가 가질 수 없다. 2.2.3절 |
| 생성 코드의 스레드 상태 포인터 | TEB TLS 슬롯(`LDR [x18,#slot]`) 또는 C getter. `yuzu\main.cpp:102`가 슬롯을 시작 시 예약 |
| TEB StackBase/StackLimit | `RtlRestoreContext`가 SP를 검사하므로 SP와 맞춰야 한다(tebfix, nceresume). 2.2.4절 |
| 다른 스레드 중단 | 코어별 워커가 Suspend/GetThreadContext/SetThreadContext/Resume. 전달 게이트 상태 기계와 BreakRescue |
| EL0 캐시 명령, CTR_EL0 | Windows가 막으므로 VEH에서 에뮬레이션. patcher가 DC CIVAC를 건너뛰고 CTR_EL0를 에뮬레이션 |
| 처리 못 하는 data abort | Dynarmic 한 단계 실행(JIT의 read-as-zero 의미), 마지막 수단으로 건너뜀 |
| fail-fast | `RaiseFailFastException`, 코드 0xE0CE0001, 0xE0CE0004, 0xE04E4352, 0xE04E4353 |

출처: `design\NCE_CLEANUP_REVIEW.md` 5.1절, 2.1절.

#### 2.2.3 x18과 FPCR

Windows ARM64에서 x18은 TEB를 가리키는 플랫폼 레지스터다. 게스트 코드가 x18을 쓰는 명령은 `Patcher::PatchText`(`patcher.cpp`)가 9단계로 처리한다.

| 단계 | 대상 명령 | 방식 |
|---|---|---|
| a | x18에 대한 CBZ/CBNZ/TBZ/TBNZ | x16을 게스트 스택에 저장, TEB 슬롯 조회, 네이티브 분기 |
| b | x18이 있는 SBFM/BFM/UBFM | 임시 레지스터 2개를 게스트 스택에 저장 |
| c | x18이 있는 LSE 원자 명령 | 0x130 저장 프레임, C getter 호출 |
| d | x18이 있는 LDXR/STXR 쌍, 단일 exclusive | 같은 0x130 프레임과 getter |
| e | 디코더로 검증한 레지스터 치환 | TEB 슬롯 spill(15단계) |
| f | 나머지 선형 x18 명령 | 문맥 저장 → 호스트 스택 → Dynarmic 한 단계(`WindowsNceExecuteDirectX18`) |
| g | 그 밖의 명령(분기, SP 변경) | 로드 시 BRK #0xF000, VEH redirect, Dynarmic 한 단계 |
| h | x18로 가는 MRS CNTFRQ/CNTPCT/CNTVCT/TPIDR | 시스템 레지스터 핸들러의 특수 처리 |
| i | a/b/e 단계나 MRS 도우미 안에서 중단 포착 | 레시피 재실행(`TryRecoverX18Substitution`) |

FPCR: 직접 x18 경로(f단계)는 호스트 FPCR로 실행한다. `EDEN_NCE_X18_HOST_FPCR=1`은 그 처리를 감사하는 진단이다(`x18_host_fpcr.h`). H 게이트도 호스트의 TEB 스택 경계와 FPCR로 들어간다(`nvn_gate.cpp` 머리 주석).

2026-11-22 DK 측정에서는 x18 비용이 병목이 아니었다. 직접 x18 단계가 초당 5-9만 번, 약 190 ns로 초당 10-16 ms였다(메모리 노트 `dk-bananza-perf.md`).

출처: `design\NCE_CLEANUP_REVIEW.md` 3.1절, 4.1절, `source\src\core\nvn\nvn_gate.cpp`.

#### 2.2.4 예외 복귀(tebfix → nceresume)

N1X에서 0xd fast-fail(`RtlRestoreContext`의 SP 검사)이 반복됐다. 원인과 수정은 다음 순서로 진행됐다.

| 빌드 | 원인 | 수정 |
|---|---|---|
| tebfix, tebfix2 | 게임이 다른 스택(fiber)에서 fault 후 재개했는데 TEB 경계가 옛 스택 기준 | SP가 속한 할당으로 경계를 맞춤 |
| x18fix → tebfix3 | 브리지가 게스트 x18을 TEB로 착각 | 대상 스레드의 TEB를 x4로 넘기고 `mov x18,x4` |
| tebfix4 | 자기 fault 전달 중에 다른 스레드의 break가 PC를 브리지로 옮김 | 브리지 구간도 맞춤 대상에 포함 |
| tebfix5 | VEH 안의 로그가 OutputDebugString 예외를 일으킴 | SP만 보는 규칙 하나로 통일, VEH 경로의 로그 제거 |
| nceresume | 검수 R1-R17 반영 | 공통 `ResumeChecked`(전환 구간 PC 되돌림, 맞출 수 없으면 0xE0CE0001 fail-fast), 바인딩 seqlock 등 |

HISTORY에 따르면 그 뒤로 NCE 크래시는 없었고 tebfix5의 "left" 카운터는 0이다. 남은 일은 P7 단계별 주입 테스트와 STP/exclusive 재실행 테스트다.

출처: `design\HISTORY_20261120.md` 3절, `design\ASTRA_NCE_RESUME_REVIEW_REQUEST.md`, `design\NCE_RESUME_REVIEW.md`, `design\ASTRA_NCE_RESUME_REVIEW.md`.

#### 2.2.5 patcher, 로더 순서, IPS

- NSO 로더(`core\loader\nso.cpp`)는 계획 패스(`PatchText`)와 로드 패스를 나눈다.
- 2026-11-22 이전에는 계획 패스에 프로그램 ID가 없어서, 프로그램 ID로 고르는 타이틀 패치가 NCE 계획 뒤에 들어갔다. 그 패치가 추가한 SVC/MRS/x18 명령은 NCE 패치를 받지 못했다(HWAoI의 "stubsvc" 버그).
- 2026-11-22 배포부터 타이틀 패치와 IPS 패치를 NCE 계획 전에 적용한다(`nso.cpp:202`, 로그 `titleplan:`, `NXN1X_TITLE_PATCH_PLAN=0`이면 예전 순서).
- 같은 배포에서 쓰이지 않는 patcher 래퍼를 지우고, CASP를 재시도하게 했다. 계획할 수 없는 x18 LSE는 경고와 함께 건너뛴다. self-test는 `NXN1X_NCE_SELFTEST`로 켠다.

출처: `docs\RELEASE_2026-11-22.md` 2절, `design\NCE_CLEANUP_REVIEW.md` 1절, `docs\NCE_CLEANUP_AUDIT_2026-11-22.md`.

#### 2.2.6 crash ring과 크래시 보고

- **crash ring**: VEH에 도달한 예외를 진입과 복귀 때 한 번씩, 256칸 파일 매핑 링 `crashring.bin`(exe 옆)에 기록한다. fast-fail 0xd가 남기는 유일한 기록이다. 컴파일 정의 `NXN1X_CRASH_RING`(기본 1)로 넣고 뺀다. `crashring_dump`가 내용을 출력한다(`arm_nce_windows.cpp:1325-1439`).
- **abort 보고**: SIGABRT, CRT invalid-parameter, purecall, `std::terminate`에서 `nxn1x_abort_<pid>.txt`를 exe 옆에 쓴다.
- **크래시 보고**: `nxn1x_crash_<pid>.txt`(예외, module+offset 주소, x0-x30/sp/pc, 프레임 포인터 체인)와 `nxn1x_crash_<pid>.dmp`를 exe 옆에 쓴다(`host\dm\nxn1x\nvnh_rt.c:324`). `crashsym` 빌드부터 PDB로 NXN1X.exe 프레임 이름을 붙인다.
- llvm-symbolizer와 llvm-objdump(clangarm64)로 NXN1X.exe 오프셋을 PDB에서 읽을 수 있다.

출처: `source\src\core\arm\nce\arm_nce_windows.cpp`, `host\dm\nxn1x\nvnh_rt.c`, `BUILD_LOG.txt`(crashsym), 메모리 노트 `dk-bananza-perf.md`.

#### 2.2.7 기본 경로에 남은 우회와 진단

`design\NCE_CLEANUP_REVIEW.md`(읽기 전용 점검, 변경 없음)가 기본 경로에서 실행되는 항목을 정리했다. 주요 항목은 다음과 같다.
- **s39 잠금 탈취**: 생성 코드의 `LockContext`가 2^23번 시도 후에도 잡혀 있는 잠금을 가져온다. TotK 정지(11-13) 대응 우회이고 기본 켜짐(`DM_S39`).
- **BreakRescue**: 5 ms마다 폴링하는 감시 스레드. **불일치**: 주석은 "`EDEN_NCE_BREAK_RESCUE=1`일 때만 켜짐"인데 코드는 release에서 항상 켜져 있다(`windows_cross_thread_break.h:178` 대 `:235-245`).
- **HWAoI 풀 재시도 스텁**: allocator의 슬롯 48 전체 풀 종료를 `svc #0xb`(1 ms sleep)로 최대 128번 재시도한다. 원인을 코드에서 찾지 못한 우회로 분류된다(`hwa_pool.h`).
- **DK dkctl 스냅숏 스텁**: 기본 켜짐(`DM_DK_CTL_LOG=0`이면 끔).

출처: `design\NCE_CLEANUP_REVIEW.md` 1절, 2절, 6절.

### 2.3 NVN 호스트 레이어

#### 2.3.1 진입: 훅과 H 게이트

`core\nvn\nvn_gate.cpp` 머리 주석의 설명:
1. **훅**: NCE patcher의 첫 패스가 `nvnBootstrapLoader` 정의의 첫 명령을 패치 섹션으로 가는 분기로 바꾼다. 게이트가 꺼져 있으면(장치 인계 전이나 Refuse 뒤) 원래 코드가 실행되고 부팅은 Emulation 경로로 간다.
2. **게이트**: `nvnBootstrapLoader`는 NVN 함수마다 게이트 블록의 항목을 돌려준다. 항목은 `&nvn_slot[i]`를 x16에 넣고 H 게이트로 들어간다. H 게이트는 게스트 스택에서 그 게스트 스레드의 호스트(fiber) 스택으로 옮기고, 호스트의 TEB 스택 경계와 FPCR로 `nvn_slot[i].fn`을 호출한다. x0-x8, v0-v7은 그대로 오간다.

NVN 호출은 게스트 스레드 위에서 호스트 함수로 실행된다. 예: `nvnSyncWait`는 S 게이트 함수로, 코어 0의 호스트 스레드에서 그 스레드가 코어를 잡은 채 실행된다(메모리 노트 `dk-bananza-perf.md`).

#### 2.3.2 HLE와 명령 기록 (`host\dm\gfx\nvn\hle_nvn.c`)

`hle_nvn.c` 머리 주석의 설명:
- `main`이 import한 `nvnBootstrapLoader`를 대신한다. sdk의 NVN 드라이버는 실행하지 않는다.
- NVN 이름 523개가 모두 카운팅 thunk(`nvn_thunks.S`)를 돌려준다.
  - 의미가 있는 것(장치, 메모리 풀, 텍스처와 샘플러, 큐, 창, sync, 이벤트, 명령 버퍼 메모리)은 NVN 의미대로 구현하고 객체도 실제로 만든다.
  - CPU에 보이는 결과가 없는 것(상태 객체, 바인딩, draw, clear 등)은 명령으로 기록했다가 submit 때 재생한다.
  - 나머지(main이 로드만 하고 부르지 않는 307개와 헤더에 원형이 없는 것)는 `nvn_unimpl_entry`로 가서, 호출되면 실패와 이름을 출력한다.
- GPU 주소는 CPU 주소와 같다. 메모리 풀은 게임이 준 실제 메모리를 쓴다.
- 명령은 게임이 준 명령 메모리에 기록하고 submit 때 재생한다.

이 코드는 nxnative 계열(Mac/iOS의 Metal 재생)에서 왔다. 주석의 "zero-copy MTLBuffer" 같은 Metal 용어가 그 흔적이다.

출처: `host\dm\gfx\nvn\hle_nvn.c` 머리 주석.

#### 2.3.3 Vulkan 재생 (`host\dm\windows\gfx_vk_*.c`)

Android 포트의 Vulkan 백엔드를 Windows로 옮긴 것이다.

| 파일 | 역할(머리 주석 기준) |
|---|---|
| `gfx_vk_core.c` | 장치, 메모리 풀, 제출, 완료. 제출은 슬롯 링(명령 풀, descriptor 풀, scratch, GPU가 지난 뒤 해제할 객체). 제출은 timeline semaphore에 펜스를 신호하고, 완료 스레드가 CPU에 보일 결과(이벤트 쓰기, 카운터 보고)를 적용 |
| `gfx_vk_exec.c` | 명령 재생(`mtl_exec`). 상태 레코드는 NVN 상태 shadow를 갱신하고, draw/dispatch는 파이프라인·descriptor set·draw로 바뀐다. render pass는 SetRenderTargets 뒤 첫 draw/clear에서 열고 draw가 아닌 것 앞에서 닫는다 |
| `gfx_vk_eden.inc` | Eden 모드(`vk_eden_mode`, 기본)의 draw/dispatch. 2.3.4절 |
| `gfx_vk_tex.c` | 형식, 텍스처, 뷰, 샘플러, ID 테이블. 같은 풀·오프셋·설명의 텍스처는 이미지 하나를 공유. packaged 텍스처는 CPU에서 deswizzle해 업로드 명령 버퍼로 채우고, dirty 추적이 쓰기를 보면 다시 채움 |
| `gfx_vk_shader.c` | 번들 경로(호스트 자체 번역기)의 셰이더. Eden 모드에서는 쓰지 않는 경로 |
| `gfx_vk_present.c` | present. NXN1X에서는 `mtl_present`가 각 present를 `core\nvn\nvn_output.cpp`에 넘김 |
| `gfx_vk_win_glue.c` | Android 포트 이름과 현재 호스트 인터페이스 연결 |
| `gfx_vk_fg_auto.inc`, `gfx_vk_fg_capture.inc`, `gfx_vk_fg_math.h`, `nvn_fg_input.h` | DLSS-G 입력(카메라, depth, motion) 탐지와 캡처 |

replay 스레드:
- 기본은 게임 스레드에서 기록한다(`DM_VK_ASYNC=0`, `DM_VK_PARALLEL=0`).
- `DM_VK_ASYNC=1`(s46)은 별도 replay 스레드에서 기록한다.
- `DM_VK_PARALLEL=<n>`은 병렬 기록이다(5.15절).
- DK는 `nvnh_rt.c`의 타이틀 기본값으로 ASYNC=1, PARALLEL=4를 쓴다(`dkpar` 빌드).

출처: 각 파일 머리 주석, 메모리 노트 `dk-bananza-perf.md`, `BUILD_LOG.txt`(dkpar+optdiag).

#### 2.3.4 Eden 파이프라인 연결 (`source\src\video_core\nvn\`)

`gfx_vk_eden.inc`는 Eden 모드에서 draw마다 Maxwell 경로가 같은 셰이더와 상태에 만들 Eden `GraphicsPipeline`을 쓴다(`nvn_eden_graphics_pipeline`).
- set 0은 Eden 순서(단계별 uniform, storage, texel, storage texel, texture, image)로 update template을 통해 채운다.
- push constant와 동적 상태도 Eden 것을 쓴다.
- 그림이 달라지는 replay 고유 규칙은 유지한다: 뷰포트 뒤집기(`set_dynamic`)와 그에 따른 winding, Bayonetta 3의 depth EQUAL 완화(r57).

브리지 파일:

| 파일 | 역할 |
|---|---|
| `nvn_shader_bridge.{h,cpp}` | NVN 셰이더를 Eden shader recompiler로 번역하는 C 인터페이스. NVN 셰이더 하나가 Eden environment 하나이고, Maxwell 경로와 같은 해시(CityHash64)를 써서 두 경로가 `vulkan.bin`의 같은 항목을 찾는다 |
| `nvn_pipeline_key.{h,cpp}` | NVN 고정 기능 상태 → Eden `FixedPipelineState`. 렌더 타깃 형식 매핑 |
| `nvn_shader_env.{h,cpp}` | NVN 셰이더 environment. NVN2 Ampere 코드 lowering 연결 |
| `renderer_vulkan\nvn_bridge_device.cpp` | Eden의 Vulkan 장치·스케줄러·파이프라인 캐시를 호스트에 인계. 그래픽 큐의 외부 동기화(`Scheduler::submit_mutex`)를 호스트와 공유 |

Eden 방식 셰이더 경로로 바꾼 결정의 이점: Switch 1 안정, 파이프라인 캐시 재사용, 다른 Eden 포크(Lichii, nxnative PR)와 비교하기 쉬움.

출처: `host\dm\windows\gfx_vk_eden.inc`, `source\src\video_core\nvn\nvn_shader_bridge.h`, `source\src\video_core\renderer_vulkan\nvn_bridge_device.cpp`, 메모리 노트(2026-11-21, Eden 방식 셰이더 전환 결정에 관한 피드백 노트).

### 2.4 셰이더 번역

#### 2.4.1 Ampere → Maxwell lowering (Switch 2)

`source\src\shader_recompiler\frontend\ampere\lower_to_maxwell.h`:
- 기존 Maxwell 프런트엔드로 의미를 표현할 수 있는 연산만 낮춘다. 원래 바이트를 재해석하지 않고 새 명령을 만든다. 스케줄/제어 워드는 environment 어댑터가 붙인다.
- 옵션: geometry, tensor(compact matrix), tessellation 단계, 검증된 trap 목록.
- UR28 이상의 uniform 레지스터는 R216 아래에 예약해 매핑한다.
- 인코딩 참고: Mesa NAK `sm70_encode.rs`(MIT).
- 호출 지점: `video_core\native_lowering.cpp` `LowerNativeProgram`, `video_core\shader_environment.cpp:667`.

FE 포팅 점검에서 확인한 항목(이미 코드에 있음): scaled shared memory 주소, IMAD.HI 쌍과 IMAD.X carry, NVN2 workgroup 수 c0+4, SDK 22 bindless TEX 형식 0x361/0xf60/0x370.

HMMA: 게스트 HMMA는 Ampere 번역기를 거친다. cooperative-matrix emitter(`emit_spirv_tensor.cpp`)가 있다고 해서 하드웨어 Tensor Core를 쓴다는 근거는 아니다. 현재 셰이더 프로필에는 cooperative HMMA 플래그를 켜는 곳이 없다.

출처: `lower_to_maxwell.h`, `docs\RELEASE_2026-11-22.md` 7.6절, `docs\DLSS_ARM64_ARCHITECTURE.md`.

#### 2.4.2 handle program과 handle table

- bindless 텍스처 handle을 draw마다 계산해야 하면 "handle program"이 된다(`shader_recompiler\environment.h` `handle_program`, `ir_opt\texture_pass.cpp`). 호스트는 `nvn_eden_eval_handle`(`HandleOp::Index` 포함)로 값을 채운다.
- `htarr`(2026-11-22): `HostTranslateInfo::support_handle_table_arrays`가 설정된 적이 없어서 invocation별 handle table 경로가 죽은 코드였다. non-uniform sampled image array indexing을 지원하는 NVN 호스트 장치에서 켠다(`vk_pipeline_cache.cpp:727`, `NXN1X_HANDLE_TABLE_ARRAYS=0`이면 끔).
- `cidx`(2026-11-22): 셰이더가 계산한 인덱스(table stride를 곱한 피연산자)도 원소 인덱스로 받는다. 인덱스는 `HANDLE_TABLE_SIZE` 64로 clamp한다(`texture_pass.cpp:881`, `NXN1X_HANDLE_CIDX=0`이면 끔).
- 진단 `handletree`(`NXN1X_HANDLETREE=1`)는 추적할 수 없는 bindless handle의 식을 로그로 남긴다.

출처: `docs\RELEASE_2026-11-22.md` 7.2절, 8절.

#### 2.4.3 그 밖의 셰이더 처리

- `nanguard`: 모든 fragment 프로그램의 float 색 출력에서 NaN을 0으로 바꾸고 ±65000으로 clamp한다(Longshot GuardFragmentOutputs 이식, GPL-3.0). `DM_VK_NAN_GUARD=0`이면 끔.
- `EDEN_MAXE`(한 파이프라인의 set 0 항목 작업 배열)를 1024에서 4096으로 올렸다(`maxe4k`). 1024는 한 indexed texture array의 한도(`texture_pass.cpp:99 DESCRIPTOR_MAX_COUNT`)였는데 파이프라인 전체 한도로 잘못 쓰이고 있었다.
- 비동기 셰이더: 빌드 중인 파이프라인의 draw를 건너뛰면 글자가 빠지는 문제가 있었다. 그래서 깊이 테스트 없는 draw와 compute는 컴파일을 기다린다(`glyphwait`).
- 파이프라인 캐시: `vulkan.bin`, `vulkan_pipelines.bin`(버전 20). 드라이버 캐시(61 MB)는 플레이 중 빌드가 20초 조용할 때, 늦어도 5분마다 저장한다(`pcache1`, `NXN1X_PCACHE_PERIODIC=1`이면 예전처럼 30초마다).

출처: `BUILD_LOG.txt`(nanguard), `docs\RELEASE_2026-11-22.md` 7.1절, `design\HISTORY_20261120.md` 4절, `design\OOT_BARRIER_DESIGN.md` 16절.

### 2.5 메모리

#### 2.5.1 N1X 드라이버의 import 메모리 문제

NXN1X는 게임 메모리 풀을 `VK_EXT_external_memory_host`로 import한다(zero-copy, GPU 주소 = CPU 주소). 드라이버가 import를 거절한 풀은 GPU 사본을 제출마다 갱신한다(`gfx_vk_core.c` 머리 주석). N1X 드라이버 621.24에서 측정한 import 메모리 성능은 다음과 같다.

| 접근 | import / device-local | 출처 |
|---|---|---|
| 고정 기능 vertex/index fetch | 100-250배 느림(membench), 76-94배(membench3 조건) | membench, membench3 |
| atomic | 약 25배 | membench5 |
| 같은 주소 반복 읽기 | 약 36배 | membench5 |
| SSBO store | 약 1.8배 | membench5 |
| SSBO/UBO/texel 읽기, 셰이더 load | 같음 | membench3/4 |
| 복사(import → device-local) | 125-190 GB/s | membench |
| texel store, 버퍼→이미지, indirect args, xfb, query copy | 1.0-1.1배 | membench5 |
| attachment 쓰기, transfer dst | 미측정 | `GPU_MEMORY_HYBRID_DESIGN.md` 2절 |

- 드라이버 비공개(내보낼 수 없는) 할당만 빠르다. 우리가 고른 주소에서 CPU가 보게 만드는 모든 방법(OPAQUE_WIN32 섹션, 우리 섹션 import, host-pointer import 변형)은 고정 기능 fetch가 느리다.
- 큰 페이지는 시험하지 못했다(SeLockMemoryPrivilege 없음).
- 드라이버 할당 HOST_VISIBLE 메모리를 게스트 원본으로 쓰는 안은 "현재 구조에서 차단"으로 판정됐다. 게임이 같은 페이지를 두 번째 주소에 매핑할 권리, `SetHeapSize`, `MapView`, `nvnMemoryPoolBuilderSetStorage`의 포인터 계약 때문이다.

출처: 메모리 노트 `n1x-imported-vertex-fetch.md`, `design\GPU_MEMORY_CACHE_DESIGN.md` 1절, `design\GPU_MEMORY_HYBRID_DESIGN.md` 1절, `design\ASTRA_HOST_VISIBLE_STATIC_REVIEW.md`.

#### 2.5.2 대응 방식

| 방식 | 스위치 | 내용 | 상태 |
|---|---|---|---|
| vbmir (정점 미러) | `NXN1X_VB_MIRROR=0`이면 끔 | 풀 청크별 영구 device-local 미러, 64 KB 블록 추적. CPU dirty(nvn_dirty)나 GPU 쓰기(sub_gpu_wrote) 때만 다시 복사 | 기본 켜짐. HWAoI 16 → 60 fps |
| vbvol | (vbmir 일부) | 보호를 4 KB 페이지로 줄이고 volatile 페이지 추가. fault 0-200/s | 기본 경로 |
| 버텍스 풀링 | `NXN1X_VTX_PULL=1` | VS가 fetch table(BDA)로 import 풀에서 직접 읽음, 인덱스는 미러 | B단계 구현, 기본 OFF |
| copy 모드 | `NXN1X_POOL_MODE=copy` | 모든 풀의 device-local 사본, 제출마다 CPU 변경 업로드, GPU 변경 병합 | 구현됨, 기본 아님 |
| hybrid | `NXN1X_POOL_MODE=hybrid` | atomic이 닿는 청크만 사본 | hybrid1/2 실행 결과 느림 |
| hy3 (atomic staging) | `DM_VK_HY3=0/1/2` (기본 2) | atomic dispatch의 바인딩만 device-local staging으로 옮기고 GPU가 병합 | 구현, OoT에서 효과 확인 |
| fstage | `DM_VK_FSTAGE=0`이면 끔 | fragment 셰이더가 쓰는 SSBO를 draw마다 staging | 구현, DK 타이틀 확인 |
| import (기본) | `NXN1X_POOL_MODE=import` | 기존 zero-copy | 기본값 |

현재 구조 결정(2026-11-20 23:49): **"혼합 = 버텍스 풀링 + import + atomic 예외 경로"**를 장기 기본 경로로 둔다. 드라이버가 고쳐지지 않을 수 있다는 전제다.

copy 모드는 GPU 쪽을 빠르게 하지만 CPU 쓰기 추적 비용이 크다.
- Bayonetta 3 depth pass: 40-57 ms → 0.6 ms.
- DK 게임플레이: copy 모드에서 8 fps. 제출 120번에 약 19.5 GB를 읽어 비교한다. 그래서 copy 모드는 DK의 해법이 아니다.

출처: 메모리 노트 `n1x-imported-vertex-fetch.md`, `bayo3-depth-pass-import-writes.md`, `dk-bananza-perf.md`, `design\GPU_MEMORY_HYBRID_DESIGN.md` 7.0절, `design\ATOMIC_STAGING_DESIGN.md` 8.2절, `design\ASTRA_PHASE_B_REVIEW_REQUEST.md`.

#### 2.5.3 쓰기 추적(nvn_dirty)과 그 함정

- `host\dm\gfx\nvn\nvn_dirty.c`는 페이지를 읽기 전용으로 보호해 CPU 쓰기를 감지하고 stamp를 남긴다.
- 함정: 코어의 `memory.cpp` `MapRegion`/`ProtectRegion`, uncached/debug-off, HostMemory의 부분 unmap 후 재매핑이 감시 페이지를 nvn_dirty 모르게 RW로 바꿨다. 그래서 copy 모드에서 정점이 깨졌다.
- 수정 `vfix`: 그런 경로가 바꾼 직후 `NvnLayer::BeforeGuestWrite`를 부른다.
- 오래된 GPU 데이터 같은 버그는 `DM_VK_PC_VERIFY`(`RUN_PCVERIFY_COPY.cmd`)로 tracker 상태와 Windows 보호를 먼저 비교한다.

텍스처 소유권 규칙:
- `gwrange`: GPU 쓰기 여부를 풀 단위가 아니라 텍스처 바이트 범위 단위로 본다(`gfx_vk_tex.c` `storage_gpu_written`, `DM_VK_GPUW_RANGE=0`이면 끔).
- `gpuown`: CPU가 다시 쓴 등록 텍스처는 GPU가 CPU의 마지막 쓰기 뒤에 썼을 때만 GPU 소유로 본다(`hle_nvn.c` `nvn_ft_poll`, `DM_NVN_GPU_OWN_AFTER_CPU=0`이면 끔). OoT Deku Tree 구멍의 원인이라 실행 스크립트에서 꺼 둔다(5.1절).
- `relovl`: GPU 전용 이미지를 더 새 텍스처 아래에 영원히 두던 문제를 고쳤다(11분에 이미지 4.7만 개, 86 GB). `DM_VK_KEPT_OVERWRITE_DROP=0`이면 끔.

출처: 메모리 노트 `core-writable-bypasses-nvn-dirty.md`, `oot-terrain-gwrange.md`, `oot-deku-terrain-gpuown.md`, `oot-speed-relovl.md`.

#### 2.5.4 명령 메모리

- StarFox가 약 115초에 abort하던 원인은 NVN command memory 상한 1 GB였다. 게임이 메모리를 콜백으로만 주는데, 콜백 응답 전에 빌려준 호스트 세그먼트가 회수되지 않아 65,489개까지 쌓였다.
- 수정: 세그먼트를 콜백이 응답한 청크에 묶고 재사용 때 회수한다(combined1120). 274초까지 정상, 세그먼트 773개로 일정.

출처: `design\HISTORY_20261120.md` 4절.

### 2.6 출력: 컴포지터, DLSS, 프레임 생성

#### 2.6.1 present 경로

- 호스트의 `mtl_present`(`gfx_vk_present.c`)는 창과 스왑체인을 따로 만들지 않고 각 present를 `core\nvn\nvn_output.cpp`에 넘긴다.
- vsync마다 `HardwareComposer::ComposeLocked`가 차례가 온 present를 게임 레이어로 합성한다.
- 렌더러가 이미지를 복사하고(`renderer_vulkan\nvn_display.cpp`), 그 복사가 GPU에서 끝나면 돌려준다. present는 순서대로 돌려준다.
- 호스트는 `vkQueueSubmit`과 present를 같은 그래픽 큐에서 `Scheduler::submit_mutex` 아래 실행한다.

출처: `source\src\core\nvn\nvn_output.cpp`, `nvn_display.cpp`, `nvn_bridge_device.cpp` 머리 주석.

#### 2.6.2 DLSS SR (Streamline)

- Streamline 2.14.16 Vulkan interposer 하나를 쓴다. SR 플러그인이 NGX를 평가한다. NGX DLSS/DLSS-G DLL은 310.9.16.0이다. 여섯 개 ARM64 DLL을 exe 옆 `streamline\`에 둔다.
- 코드: `renderer_vulkan\nvn_streamline.{h,cpp}`(`NXN1X_STREAMLINE_SDK`로 빌드).
- 2026-11-19 probe: SR 오프스크린 출력은 통과했다. 하지만 동기화 검증에서 `nv.ngx.dlss.Evaluate` 안의 WRITE_AFTER_WRITE 오류 2건이 나와, 그 시점 문서는 "production integration blocked"로 적었다.

NVN2 게임의 DLSS(`OP_DLSS`):
- `hle_nvn.c`의 `nvnCommandBufferEvaluateDlss`가 텍스처 역할 6개, 크기, 모션 스케일, jitter, pre-exposure, history reset을 기록한다.
- `nvnDlssNetworkBuilderGetOptimalSettings`의 답 순서는 optimal, **smallest**, **largest**다(NGX 순서가 아님). 범위는 output/3..output(`dlssrange`, `hle_nvn.c:3567`, `DM_NVN_DLSS_RANGE=0`이면 옛 순서).

OoT(`OP_DLSS` 없음):
- 게임 자체 업스케일 네트워크를 DLSS SR로 바꾼다(`DM_VK_OOT_DLSS`). 6.1절.

설정과 단축키:
- 그래픽 고급의 "NVIDIA DLSS Super Resolution", F7(`dlsstoggle`).

**불일치**: `docs\DLSS_ARM64_ARCHITECTURE.md`는 "SR and FG are off in the game", "The game executable has no SDK DLL dependency"라고 쓴다. 그러나 이후 빌드(ootdlss7~11, dlsstoggle)와 메모리 노트는 게임 안에서 DLSS SR과 DLSS-G가 동작한다고 기록한다. 이 아키텍처 문서는 초기 감사 시점의 기록으로 보아야 한다(문서 첫 단락도 "describes the initial audit"라고 적는다).

출처: `docs\DLSS_ARM64_ARCHITECTURE.md`, `docs\DLSS_ARM64_CAPABILITIES.md`, `docs\RELEASE_2026-11-22.md` 1절, `BUILD_LOG.txt`, 메모리 노트 `oot-dlss-plan.md`, `xcde-dlss-range-order.md`.

#### 2.6.3 DLSS-G와 광학 흐름 프레임 보간

- **DLSS-G**: 스왑체인 interposer가 합성된 최종 색을 받는다. depth와 motion은 생산 명령 버퍼에서 태그하고, 카메라 상수와 태그는 원본 present ticket에 묶는다. 입력이 없거나 오래됐거나 모호하면 켜지지 않는다.
- **자동 입력 탐지**: 바인딩된 vertex UBO에서 rigid view, perspective projection, view-projection을 찾는다. depth/motion 샘플 16개를 카메라 재투영과 비교하고, 세 번 일치하면 RG attachment와 단위·부호를 확정한다(`gfx_vk_fg_auto.inc`, `gfx_vk_fg_math.h`).
- **광학 흐름 보간**: DLSS 입력이 없는 원본 NVN ticket에는 광학 보간을 쓴다(Lichii commit 60ee2ef의 `fsr3_frame_gen` 이식). HUD를 포함한 합성 색을 보간한다.
  - 분수 목표 계획: 예를 들어 원본 43 fps + 생성 17 = 60.
  - `fgfill`: DLSS-G가 생성하지 않을 때 광학 보간이 채운다.
  - `optfill`: NVN ticket이 없는 present에도 자체 원본 번호(bit 62)를 준다(`NXN1X_OPTFILL_ANY=0`이면 끔). DK에서 보간이 동작하게 한 수정이다.
- DLSS-G는 정수 배수만 지원한다. 60 Hz 목표에서 원본이 약 53 fps면 생성하지 않는다. 120/Auto 목표가 필요하다.

설정과 단축키:
- 그래픽 고급의 Frame Generation(Off / DLSS Frame Generation ×2 / LSFG ×2), "Frame Generation source"(Auto / DLSS-G only / optical only, 실행 중 변경 가능), F10(Toggle Frame Generation).
- 오버레이 FG 행은 DLSS-G, 광학, 원본 수를 0.5초 창으로 센다(`fgoverlay`).

**불일치**: `docs\DLSS_ARM64_TEST_REPORT.md` 안에서도 선택지가 바뀐다. 앞 절은 "OFF, Auto, 30, 60, 90 or 120 FPS"에 LSFG 비활성, 뒤 절은 "Off, NVIDIA DLSS Frame Generation ×2, or LSFG ×2"다. 어느 쪽이 현재 UI인지는 소스 UI 파일을 확인해야 한다(미확인).

출처: `docs\DLSS_ARM64_TEST_REPORT.md`, `design\OOT_DLSS_DESIGN.md` 1절, `BUILD_LOG.txt`(ootdlss10, fgoverlay, fgsource, optfill), 메모리 노트 `oot-dlss-plan.md`, `dk-bananza-perf.md`.

#### 2.6.4 그 밖의 출력 처리

- `opthold`: NVN 출력 복사의 크기와 형식이 원본 8개 동안 유지될 때까지 프레임 생성을 기다린다(`NXN1X_OPT_HOLD=0`이면 끔).
- `viewusage`: 프레임 생성이 켜지면 프레임 이미지에 STORAGE usage가 붙는다. sampled/attachment 뷰는 `VkImageViewUsageCreateInfo`로 자기 usage를 명시한다(`NXN1X_VIEW_USAGE=0`이면 끔).
- 스왑체인 재생성은 `submit_mutex` 아래 idle을 기다린다(`DM_VK_SWAPCHAIN_WAIT`).
- GPU checkpoint(`DM_VK_CHECKPOINTS`)는 첫 device lost 때 보고한다.
- `BlitScreen::WaitIdle`은 `scheduler.Finish()` 뒤 NVN submit mutex를 잡는다(검증 레이어가 본 경합 대응).

출처: `docs\RELEASE_2026-11-22.md` 7.5절, 메모리 노트 `oot-dlss-plan.md`, `docs\DLSS_ARM64_TEST_REPORT.md`.

### 2.7 오디오와 동영상

- NXN1X의 NVN 호스트 라이브러리는 `dm/host`를 빌드하지 않는다(`source\src\nvn_host\CMakeLists.txt`). 따라서 오디오와 게임 동영상은 Eden의 HLE 서비스가 처리하는 것으로 보인다. 다만 NXN1X 고유의 오디오·동영상 설계 문서는 없다(미확인).
- `host\dm\windows\movie_stub_win.c`(AVC는 검은 화면, AV1은 출력 없음)는 호스트 단독 Windows 포트용이다. `nvn_host` 빌드 목록에는 없다.
- `host\docs\reference\audio.md`, `movies.md`는 nxnative(호스트 단독 프로세스, Mac/iOS)의 문서다. NXN1X에 그대로 적용되지 않는다.
- `source\CMakeLists.txt`에는 Eden의 `ENABLE_CUBEB`(ON), `YUZU_USE_BUNDLED_FFMPEG` 옵션이 있다.

출처: `source\src\nvn_host\CMakeLists.txt`, `host\dm\windows\movie_stub_win.c`, `host\docs\reference\README.md`, `source\CMakeLists.txt`.

---

## 3. 게임 파일 형식

### 3.1 지원 형식과 판별 순서

`source\src\core\loader\` 로더: NSP, XCI, NCA, NAX, NSO, NRO, KIP, NUP, DeconstructedRomDirectory(덤프 폴더).
- 판별은 `loader.cpp`의 `IdentifyFile`이다. 2026-11-22부터 덤프 폴더 판별을 NSO 판별보다 먼저 한다(`loader.cpp:106`). 그래서 `exefs/main`이 단일 NSO가 아니라 덤프 폴더로 부팅된다.
- NSP/XCI/NCA는 Eden 원래 로더다. 키(prod.keys)가 필요하다.

출처: `source\src\core\loader\loader.cpp`, `docs\RELEASE_2026-11-22.md` 5절.

### 3.2 NUP (NX Package, SPEC v1)

`source\src\core\file_sys\nup_container.h` 기준:
- 파일 하나에 base 프로그램, 업데이트, DLC의 ExeFS/RomFS 트리를 그대로 담는다.
- 구조:
  - Superblock 0x80바이트, magic `"NUP\x01"`
  - TOC: 패키지, 디렉터리/파일 노드, children 배열, 문자열 풀
  - Blob 영역: 파일별 extent, raw 또는 zstd, 16바이트 정렬
- 패키지 종류: `Base`=0, `Update`=1, `Addon`=2. `standalone_update` 플래그가 있다.
- 패키지별 meta TLV: NPDM, 타이틀 이름(기본 + 지역화), publisher, source dump, 아이콘, display version.
- 정수는 little-endian이다. zstd 디코딩 캐시는 기본 512 MB이고, 그 절반보다 큰 멤버는 임시 파일로 푼다.
- 파일은 암호화되지 않아 키가 필요 없다.
- 로더: `core\loader\nup.cpp`(NACP 16개 언어 슬롯 매핑 포함).

base 없는 컨테이너: base 레코드가 없거나(모든 업데이트에 standalone_update) 파일 없는 base 레코드(title ID 0)가 있으면, 고른 업데이트가 게임 전체로 쓰인다. 플래그가 그것을 보장하지는 않는다(`pack_nup.py` 머리 주석).

SPEC 원문: https://github.com/a55uka/nup (SPEC.md v1, `pack_nup.py` 주석의 링크).

### 3.3 make_nup.py: 덤프 폴더 → NUP

`host\scripts\make_nup.py`:

```
python host\scripts\make_nup.py <dump dir> <out.nup> --title-id <16 hex> --version <n> --name "<이름>" [--base] [--display-version 1.2.0] [--resume]
```

| 인수 | 뜻 |
|---|---|
| `<dump dir>` | `exefs/` + `romfs/`가 있는 폴더 |
| `--title-id`, `--version`, `--name` | 필수 |
| `--base` | base 패키지로 만듦. 기본은 standalone update(kind 1, flag 1) |
| `--display-version` | 표시 버전 |
| `--resume` | 이전 실행이 `<out>.part`에 쓴 파일을 이어서 사용 |

- 파일은 raw(codec 0)로 저장하고 파일마다 SHA-256을 둔다. zstd 압축은 쓰지 않는다.
- meta 블록에는 NPDM, 타이틀 이름, 원본 덤프 라벨, display version을 넣는다.
- 데이터를 먼저 쓰고(해시 포함) TOC와 superblock을 마지막에 쓴다. 260자를 넘는 Windows 경로를 처리한다.
- 읽기 스레드는 32개다. 작은 파일이 대부분이기 때문이다.
- 실제 사례: Fire Emblem Fortune's Weave(v131072), 29.1 GB, 252,693개 파일, 모든 SHA-256 검증.

`host\scripts\pack_nup.py`는 반대 방향이다. nxnative의 IPA 빌드 스크립트용으로 NUP를 추출 레이아웃(ExeFS, RomFS 이미지, NACP, 아이콘, DLC 이미지)으로 푼다. zstd 모듈이 필요하다.

출처: `host\scripts\make_nup.py`, `host\scripts\pack_nup.py`, `docs\RELEASE_2026-11-22.md` 6절.

### 3.4 덤프 폴더 (`exefs/` + `romfs/`)

2026-11-22 배포에서 고친 항목:

| 이름 | 위치 | 내용 | 스위치 / 로그 |
|---|---|---|---|
| `dirfs` | `core\loader\deconstructed_rom_directory.cpp:296` | 느슨한 `exefs/main` 부팅에 FS 컨트롤러 등록이 없어 `fsp SetCurrentProcess`가 2002-1002로 실패하고 게임이 스스로 종료했다(FE 무한 로딩). 이제 `exefs/`의 `.romfs` 파일이나 옆 `romfs/` 폴더로 RomFS를 등록한다 | `NXN1X_DIRFS=0`, `dirfs:` |
| `maxopen` | `core\file_sys\vfs\vfs_real.cpp:43` | VFS는 CRT `FILE*`로 최대 8192개를 열어 두는데 CRT 한도는 2048이었다. RomFS 파일이 25만 개인 폴더에서 "Too many open files"가 났다. CRT 한도를 8192로 올리고 나머지 용도는 1024로 유지 | `NXN1X_VFS_MAXOPEN=<n>`, `maxopen:` |
| `fsfail` | `fs_i_filesystem.cpp:22`, `fsp_srv.cpp:275` | 실패한 게스트 FS 호출의 경로를 Warning으로 기록, 처음 64개(release에서는 LOG_DEBUG가 빠지므로) | `NXN1X_FS_FAIL_LOG=0`, 기본 켜짐 |

출처: `docs\RELEASE_2026-11-22.md` 5절.

---

## 4. 설계 문서 요약

`design\` 폴더의 문서 26개다. 대부분 2026-11-20~21에 썼다. "검수 요청(REQUEST)"과 "검수(REVIEW)"는 짝을 이룬다. 문서 안의 "Astra"는 외부 검수자, 이름 없는 REVIEW는 내부 검수다.

### 4.1 HISTORY_20261120.md

- **범위**: 2026-11-19~20의 N1X 성능과 크래시 작업 기록.
- **내용**:
  - import 메모리 저속 측정과 vbmir/vbvol
  - copy 모드 1·2단계(poolcopy → stage2a/2c/2b → combined1120)
  - NCE 0xd 크래시(tebfix~nceresume)
  - StarFox 명령 메모리 abort, 종료 크래시, 글자 빠짐(glyphwait)
- **결론**: NCE 크래시 해결. HWAoI copy 모드는 이동 중 52-57 fps(import + vbmir는 60 고정). 다음 방향은 hybrid. 열린 항목은 무한 로딩(sv1 덤프 없음), T15(GPU 쓰기 중 CPU가 옛 값을 다시 쓰는 경우, 알려진 한계), NCE P7 주입 테스트.

### 4.2 GPU_MEMORY_CACHE_DESIGN.md (copy 모드 1단계)

- **문제**: N1X에서 import 메모리의 GPU 접근이 느리다(2.5.1절).
- **방식**: NVIDIA에서는 모든 풀을 driver가 할당한 device-local 사본(copied pool)으로 운용한다.
  - CPU 쓰기: nvn_dirty 기반 4 KB 증분 업로드
  - GPU 쓰기: `gw[]` 범위로 되돌림
  - 드라이버별 정책: NVIDIA는 사본, Adreno/Mac은 import
- **상태**: 사본 위치 A(device-local)로 결정. 구현 결과 Bayonetta 3 depth pass 0.6 ms. HWAoI는 22-24 fps(CPU 비용)로 2단계가 필요해졌다.

### 4.3 ASTRA_GPU_MEMORY_CACHE_REVIEW_REQUEST.md / ASTRA_GPU_MEMORY_CACHE_REVIEW.md / GPU_MEMORY_CACHE_REVIEW.md

- **요청**: barrier 순서, `g_done`과 `nvnSyncWait` 구조의 빈틈, 바인딩 범위 지연 업로드의 누락 위험, 되돌림 양, Eden BufferCache 재사용 범위.
- **Astra 판정**: "A안의 방향은 타당하다. 정확성 계약을 보완한 뒤 구현한다."
  - P0: 되돌림 기본 경로를 별도 readback staging으로 고정할 것
  - P0: 4 KB dirty와 최신 데이터 소유권을 구분할 것
  - P0: 모든 alias와 접근 경로가 같은 사본을 쓰게 할 것
- **내부 판정**: "(A)의 비교 구현 진행, 기본값 전환 보류."
  - P1 R9: device-local → import 직접 GPU 복사는 CPU 변경을 덮는다
  - P1 R1/R2: 크기 미상 쓰기와 raw global 쓰기 로그 포화 시 누락
- 두 검수 모두 프로젝트 소스와 로그 없이 설계 문서만 본 결과라고 밝힌다.

### 4.4 GPU_MEMORY_CACHE_STAGE2_DESIGN.md / ASTRA_STAGE2_REVIEW_REQUEST.md / ASTRA_GPU_MEMORY_CACHE_STAGE2_REVIEW.md / GPU_MEMORY_CACHE_STAGE2_REVIEW.md

- **문제**: copy 1단계에서 HWAoI CPU 비용이 컸다. 앞 제출 대기 8-10 ms/프레임, 비교 6-7 ms, 쓰기 예외 1.6만/s.
- **원안**: 기준본을 "제출 시작 때 사본"에서 "에폭별 첫 쓰기 스냅숏 S[e]"로 바꿔 직렬화를 없앰. 4 KB dirty 비교, volatile 페이지(3회 연속 쓰이면 진입, 8회 쓰이지 않으면 복귀).
- **검수**: 두 검수 모두 원안의 병합 규칙에 반례가 있다고 판정했다. 예: GPU가 X, CPU가 Y를 쓰면 (2,8)이어야 하는데 (1,8)이 나온다. 4 KB 비교는 따로 진행해도 된다고 했다.
- **반영(8절)**: GPU 측 사전·사후 readback(pre/post)과 "G의 거울" shadow로 직렬화를 제거하도록 바꿨다.
- **결과(HISTORY)**:
  - stage2c: HWAoI 대기 12 → 0 ms, 27 → 37-57 fps
  - fault-ahead(순차 fault면 P+1..P+15를 한 번에 해제): 52-57 fps
  - self-test: case 0-12, T1-T17, V1-V7, FA1-FA3

### 4.5 GPU_MEMORY_HYBRID_DESIGN.md / ASTRA_GPU_MEMORY_HYBRID_REVIEW_REQUEST.md

- **문제**: copy는 GPU를 빠르게 하지만 CPU 쓰기 추적 비용이 있다(HWAoI 이동 중 fault 20-35k/s, mprotect 30-50k/s).
- **방식**: backing 범위(256 MB 정렬 공유 사본 또는 private 청크) 단위로 느린 GPU 쓰기(shader store, atomic)가 있는 범위만 copy, 나머지는 import로 둔다. 한 방향 전환, 드라이버가 빨라지면 자동으로 끔.
- **구조 결정(7.0)**: 혼합 = 버텍스 풀링 + import + atomic 예외 경로.
- **결과**:
  - hybrid1(atomic 청크 전환): 256 MB 블록 닫힘 규칙으로 452개 청크(493 MB)가 사본이 되어 4-5 fps.
  - hybrid2(정확한 범위, 읽기 전용 제외, `NXN1X_SSBO_WRITES=all`로 A/B): 55개 청크, 그래도 import + ootfadd보다 느림.
- **상태**: 전제 검증(1.1절 카운터)은 "미검증"으로 남았다. atomic 예외 경로는 4.6절의 hy3로 대체됐다.

### 4.6 ATOMIC_STAGING_DESIGN.md (hybrid3, hy3)

- **문제**: hybrid1/2에서 atomic 바인딩의 GPU 결과를 매 제출 CPU가 비교·병합하는 비용(replay pool-sync 17.5초, copyback 15.8초).
- **방식**: 게임 메모리는 import로 둔다. atomic dispatch 동안만 바인딩 범위를 device-local staging(S)과 스냅숏(P)으로 옮기고, GPU가 3-way 병합으로 되돌린다(`hy3_merge.comp`). CPU는 관여하지 않는다. 출발점은 기존 TotK 전용 `n1x_atomic`이다.
- **2단계(창 묶기)**: 한 제출 안의 연속 dispatch를 창으로 묶고, dispatch가 아닌 GPU 작업 앞에서 닫는다. 바인딩 한도 16 MB, 창 총량 64 MB. `DM_VK_HY3=0` 끔 / `1` 1단계 / `2` 기본.
- **결과(stage2s1, OoT)**:
  - 셀프테스트 HYBRID 93/93, COPY 73/73 PASS
  - dispatch → dispatch gap 46 µs → 18.9 µs
  - fps 27-34
  - 다음 병목은 part 경계 4.7 ms(GPU가 CPU의 다음 part를 기다림)
- **확장**: DK에서 fragment 셰이더 storage 쓰기로 넓힌 `fstage`(6.2절).

### 4.7 VERTEX_PULLING_DESIGN.md / ASTRA_REVIEW_REQUEST.md / ASTRA_VERTEX_PULLING_REVIEW.md / VERTEX_PULLING_REVIEW.md / ASTRA_PHASE_B_REVIEW_REQUEST.md

- **문제**: vbmir 미러는 CPU 쓰기 감지(페이지 보호와 fault)를 쓴다.
- **방식**: VS가 정점 속성을 고정 기능 입력 대신 import 풀에서 BDA로 직접 읽고, 인덱스는 device-local 미러에 둔다.
- **검수**: 두 검수 모두 "방향 타당, 현 설계 그대로 기본 적용은 불가." 받아들인 주요 지적은 다음과 같다.
  - 증명된 연속 범위만 테이블에 싣기
  - 검사 후 분기해서 로드(select로 가리지 않음)
  - 소비 단계 변경에 따른 배리어
  - 게임 draw 이중 실행 비교 폐기
  - push constant offset 48, 범위 56
- **A단계(membench4)**: 형식 전수 비교 0 불일치(NaN payload 제외). 풀링 + device-local IB가 현재 미러와 같은 속도(13.8 대 12.1 µs, 29.1 대 29.7 µs).
- **B단계**: Eden 경로에만 구현. 조건을 만족하는 direct draw만 대상이고, 기본 OFF(`NXN1X_VTX_PULL=1`). 변경 파일은 `runtime_info.h`, `emit_spirv.h`, `spirv_emit_context`, `emit_spirv_context_get_set.cpp`, `fixed_pipeline_state.h`, `vk_vertex_pulling.h` 등이다.
- **상태**: B단계 검수 요청까지 확인했다. 게임 비교(C단계) 결과 문서는 찾지 못했다(미확인).

### 4.8 ASTRA_HOST_VISIBLE_STATIC_REVIEW.md

- **질문**: 드라이버가 할당한 HOST_VISIBLE 메모리를 게스트 원본으로 쓸 수 있는가(import 저속을 피하는 대안).
- **판정**: 전체 게임 메모리에 대해서는 "현재 구조에서 차단." 근거는 `KPageTableBase::SetHeapSize`, `HostMemory::Impl::MapView`, `nvnMemoryPoolBuilderSetStorage`가 정하는 주소 계약이다. 게임이 같은 페이지를 두 번째 주소에 매핑할 권리가 남는다. 일반 `vkMapMemory`에는 지정 주소에 두 번째 view를 만드는 계약이 없다.

### 4.9 ASTRA_NCE_RESUME_REVIEW_REQUEST.md / ASTRA_NCE_RESUME_REVIEW.md / NCE_RESUME_REVIEW.md

- **문제**: 하루에 다섯 번 수정(tebfix2 → tebfix5)한 NCE 예외·redirect·재개 설계가 완전한가.
- **전제(W1-W5)**: `RtlRestoreContext`의 SP 수용 규칙, VEH가 게스트 스택에서 실행됨, `NtContinue`, `SetThreadContext` 등.
- **판정**:
  - Astra: "tebfix5는 개선이지만 전체 설계의 완성을 입증하지 못한다." 브리지는 유지하고, 경계 선택 정책과 최종 재개 상태 검증을 공통화할 것. 우선순위 1은 대상 thread·TEB·GuestContext 일치, `CONTEXT_FULL`에 x18이 포함되지 않는 점, 전환 단계별 재구성 규칙이다.
  - 내부: "브리지는 유지한다." P1은 `FitTebToSp` 실패 시에도 그대로 재개하는 분기, 핸들러 프롤로그 중 예외.
- **결과**: nceresume 빌드가 R1-R17을 반영했다(2.2.4절).

### 4.10 NCE_CLEANUP_REVIEW.md

- **범위**: NCE와 그 연결 코드의 읽기 전용 점검(빌드·실행 없음).
- **내용**:
  - 타이틀별 훅 13곳(HWAoI, XC2/BF2, MKW, DK, OoT)
  - 기본 경로의 진단·우회
  - x18 9단계의 중복
  - 스위치 목록
  - 플랫폼 의존 부분
  - 정리 제안 P1-P6
- **발견**:
  - 로더 계획 패스에 프로그램 ID가 없는 구조적 위험
  - BreakRescue 주석과 코드 불일치
  - 0x0400E95004038001을 NCE 코드는 "XC2 3.0.1", `k_system_control.cpp`는 "BF2"라고 부름
- **후속**: `docs\NCE_CLEANUP_AUDIT_2026-11-22.md`가 첫 삭제 대상을 패처의 미사용 private 오버로드 9개로 좁혔다(Windows ARM64 구문 검사 18/18 통과, 링크와 게임 실행은 미포함). 2026-11-22 배포에 정리가 들어갔다(2.2.5절).

### 4.11 OCCLUSION_REDESIGN.md

- **문제**: OoT Kokiri의 haze, 떨림, 깜빡임. 원인은 앞 프레임에서 가져온 대체 occlusion 값(occkey, occaddr, occland, 고정 4096)이었다.
- **참고**: HeyListen(GPL-3.0, 메커니즘만), NX2navi, Longshot, Eden 쿼리 캐시.
- **방식(`DM_VK_OCC2`)**:
  - s1: stream serial별 draw당 쿼리 슬롯(64K 링)
  - s2: GPU resolve와 stream 순서 착지, 대체값 제거
  - s3: `VK_EXT_conditional_rendering`으로 GPU에서 render condition 판정
  - s4/s5: 워커 occlusion segment, 채택(미착수)
- **결과**: s3에서 과노출과 깜빡임이 사라졌다(2026-11-21 사용자 확인, `RUN_OCC_S0`와 A/B). 비용은 31-40 fps 대 이전 경로 약 47 fps. `occ-s3fast`(Eden `queries_prefix_scan_sum` 재사용, 조건 블록 병합)는 진행 중.
- **상태**: 기본값은 아직 0. fps A/B와 Switch 1 / Zelda 회귀 확인 뒤에 바꾼다.

### 4.12 OOT_BARRIER_DESIGN.md

- **문제**: OoT에서 full barrier가 초당 13,265개. gap 하나에 full 2개가 들어간다.
- **원인**: OoT는 dispatch마다 guest barrier를 기록하고, OP_BARRIER가 `g_bar_seq`를 올려 r89의 compute 전용 barrier 조건을 깨뜨린다.
- **수정**:
  - `cbar1`(`DM_VK_CBAR=0`이면 끔): dispatch 사이 guest barrier 생략 조건
  - `cbar2`(`DM_VK_CBAR2=0`이면 끔): 연속 barrier 합치기
  - 진단 tsdiag/cpudiag
- **후속 절**:
  - s46 비동기 replay A/B: 게임 스레드 submit+present는 17-20 → 0.3-0.5 ms이지만 fps는 같음
  - s46pc: replay 스레드 TIME_CRITICAL + P코어 정책이 회귀를 일으켜 s46pc2에서 ABOVE_NORMAL로 바꿈
  - `nvnTextureFinalize` 지연(`DM_VK_FIN_DEFER=0`이면 끔)
  - pcache1(30초 캐시 저장 문제)
- **남은 것**: s48 range import 130-180 ms 튐과 장면 진입 2-3 s는 손대지 않았다.

### 4.13 OOT_DLSS_DESIGN.md

- **문제**: OoT는 NVN DLSS 명령을 쓰지 않고 자체 업스케일 네트워크(a335c1db → d99fbeeb)로 536 → 1440 업스케일한다. 광학 흐름 FG가 HUD 포함 화면을 보간해 잔상이 의심됐다.
- **방식**: NX2navi(같은 게임의 static recompilation, MIT)의 `vk_dlaa.c`와 `dlaa_mv.comp`에서 다음을 재사용한다.
  - 카메라 블록, 프레임 모양, 업스케일러 위치
  - jitter(Halton 2,3), 카메라 기반 MV, 품질 비율
  - 단계: D1a survey → D1b SR → D1c jitter → D2 DLSS-G
- **결과(메모리 노트)**: ootdlss7에서 DLSS SR이 모든 DRS 크기(1138x640..1920x1080 → 2560x1440)의 게임 네트워크를 대체했다. 사용자 평: "DLSS가 훨씬 좋네". 이후 ootdlss9~11(NaN guard, srfit, 출력 정지 감지), resfix(고정 장면 비율 0.5), nanguard, expbound/flashdiag.
- **불일치**: 설계서는 카메라를 "c3의 0x3500바이트 scene context"(NX2navi 기준)로 적는다. 메모리 노트는 구현이 "vertex UBO0 >=0x230 (Longshot layout: VP 0x30, proj 0x70, size 0x120)"을 런타임 검증으로 쓴다고 적는다. 설계 후 구현에서 바뀐 것으로 보이지만 설계서는 갱신되지 않았다.
- **불일치**: 설계서 3절은 단계마다 별도 빌드(D1a, D1b, ...)를 전제한다. `CLAUDE.md`의 "한 빌드" 원칙은 이 설계 뒤인 2026-11-21에 정해졌다.

### 4.14 PARALLEL_REPLAY_DESIGN.md

- **문제**: replay 스레드 하나가 모든 NVN 목록을 직렬로 기록한다(OoT 약 15 ms/프레임, 약 2,300 draw). GPU는 part를 기다리며 약 6 ms 쉰다.
- **방식**: HeyListen 모델을 따른다.
  - replay 스레드가 stream을 한 번 순서대로 걸으며 순서 의존 결정을 모두 한다(dry pass: 보고, 이벤트, render condition, uniform 갱신 목록, occlusion, 렌더 타깃 표시, part 분할).
  - draw 묶음을 segment로 잘라 워커 스레드가 NVN 상태 스냅숏에서 자기 primary 명령 버퍼에 다시 실행한다.
  - chunk는 stream 순서로 제출한다.
  - 1단계는 GPU 메모리를 쓰는 연산을 모두 replay 스레드에 둔다.
- **단계**: R1-R8 리팩터, S1a(워커 기반), S1b(dry pass, `DM_VK_PARALLEL_DRY=1`, `DM_VK_PARALLEL_SERIALSEG=1`), S1c(스레드 안전), S1d, S1e(워커 기록, `DM_VK_PARALLEL=<n>`).
- **결과**:
  - OoT S1b/S1d: SERIALSEG mismatch 0, 그림 동일
  - DK: `uppool`(슬롯별 업로드 명령 풀, `DM_VK_UPPOOL=0`이면 끔)로 병렬 기록 크래시를 해결(2026-11-22). PARALLEL=4에서 draw의 약 85%가 워커로 가고 20 → 26-30 fps
- **불일치**: 문서 머리는 "NVN1 titles first (Zelda Ocarina of Time is the target)"라고 쓴다. 그러나 `OOT_DLSS_DESIGN.md`와 메모리 노트는 OoT를 NVN2(Switch 2) 타이틀로 다룬다.
- **불일치**: 문서 상태 줄은 S1e를 "syntax-checked only, not run"으로 적지만, 메모리 노트에는 이후 게임 실행 결과가 있다. 문서가 갱신되지 않았다.

---

## 5. 게임별 상태

상태는 2026-11-22 기준이다. "확인"은 사용자가 게임에서 확인한 것, "로그"는 로그 판독만 한 것이다.

### 5.1 Zelda: Ocarina of Time (Switch 2, 0400486024DEC000)

| 항목 | 상태 | 근거 |
|---|---|---|
| Kokiri 지형 "물"/하늘 반사 | 해결(gwrange), 확인 | `oot-terrain-gwrange.md` |
| haze, 떨림, 깜빡임 | `DM_VK_OCC2=s3`에서 해결, 확인. 기본값은 아직 0 | `oot-occlusion-s3.md` |
| 60 fps 미만 슬로모션 | `ootspeed`(main+0x6e6cc4 step 패치, `DM_REALTIME_PATCH=0`이면 끔), 확인 | `oot-speed-relovl.md` |
| 장시간 느려짐 | `relovl`, 16분 실행에서 2-3 GB 유지, 확인 | 같음 |
| DLSS SR | ootdlss7 이후 동작, 확인. 스위치 `DM_VK_OOT_DLSS` | `oot-dlss-plan.md` |
| 화면 전체 흰 번쩍임 | nanguard + flashdiag 빌드 뒤 이번 주 OoT 작업 종료. expbound는 나중에 빠짐(flashdiag2) | `oot-wrapup-friday.md`, `BUILD_LOG.txt` |
| Deku Tree 바닥 구멍 / 유령 지형 | 미해결. 원인은 gpuown. 실행 스크립트(`RUN_PAR2`, `RUN_PAR4`, `RUN_DK*`)에서 `DM_NVN_GPU_OWN_AFTER_CPU=0`. 투명해지는 셰이더 FS c72b6ea746c00433 | `oot-deku-terrain-gpuown.md` |
| 줄무늬 텍스처 | 다음 주로 미룸. F3 stripediag 덤프가 빌드에 있음 | `oot-wrapup-friday.md` |
| 시작 직후 끊김 | Eden 파이프라인 약 1,800개를 처음 사용할 때 생성(30-90초, fps 32-53). prewarm 예정 | `pipeline-prewarm-todo.md` |
| device lost | 3회 기록(그중 1회 지역 전환), 원인 미확정 | `oot-dlss-plan.md` |
| ootdlss9 실행 "unplayable" | 프레임 간격은 고른데(45-49 fps) 원인 미확인 | `oot-dlss-plan.md` |

메모 `oot-wrapup-friday.md`: OoT는 이번 주로 마무리한다. 추가 버그와 다른 프로젝트의 수정 사례는 2026-11-27(금)에 넘겨받는다.

### 5.2 Donkey Kong Bananza (040074A01BF12000)

| 항목 | 상태 | 근거 |
|---|---|---|
| "GPU 208 ms" | 잘못 읽은 수치. 그 줄은 긴 프레임 하나였고, GPU 열은 submission timestamp 구간의 합이라 CPU replay 시간을 포함 | `dk-bananza-perf.md` |
| 해상도 | 게임별 설정 1X(resolution_setup=3), 전역은 2X | 같음 |
| 게임플레이 병목 | replay CPU. 메인 스레드 50 ms의 대부분은 `nvnSyncWait`(GPU fence 대기). x18 아님(EDEN_NCE_STATS) | 같음 |
| 타이틀 20 fps | 1440x810 R32F 전체 화면 draw 하나가 import 풀에 storage를 써서 38-40 ms. `fstage`로 타이틀 60 fps, GPU 0.7 ms | 같음 |
| copy 모드 | 타이틀은 60 fps지만 게임플레이 8 fps. 해법 아님 | 같음 |
| 병렬 기록 | uppool로 크래시 해결, 확인(약 16.8K present, device lost 없음). PARALLEL=4에서 26-30 fps(이전 약 20). replay 스레드는 여전히 21-23 ms/프레임 | 같음 |
| 기본값 | DK는 ASYNC=1, PARALLEL=4(`nvnh_rt.c`) | `BUILD_LOG.txt` dkpar |
| 광학 보간 | `optfill`로 동작, 확인 | `dk-bananza-perf.md` |
| DLSS SR(agl_ndsr 대신) | 다음 주 | 같음 |
| 다음 주 과제 | replay part 경계 대기(프레임당 part 끝 약 6번, 3.4 ms 대기) | 같음 |
| 게임 내장 NDSR | 원본 guest compute가 기본. `DM_VK_NDSR=0`이면 GSR1. 한때 보라색 화면이 보고됐고, storage 뷰 분리 수정 뒤 게임 확인 대기로 기록 | `DLSS_ARM64_TEST_REPORT.md` |

`DLSS_ARM64_TEST_REPORT.md`의 11-19 측정(2X, 3.1-5.5 fps, 중앙 frame 213 ms)은 그 뒤 빌드에 비해 오래된 수치다.

### 5.3 Xenoblade Chronicles DE / Xenoblade Chronicles 2 (Switch 2 Edition)

- **XCDE (0400FF500E34A001)**: 타이틀 배경이 검고 인트로가 멈췄다(낮/밤 분기에서 약 1초만 움직임).
  - 원인: DLSS GetOptimalSettings 답 순서. 게임은 3/4번 인수를 하한, 5/6번을 상한으로 보고 입력이 범위 안일 때만 DLSS를 평가한다(main+0x7742c0, main+0x7745e0). 타이틀은 1280x720으로 그려 DLSS가 건너뛰어졌고, 다음 compute pass가 쓰이지 않은 1920x1080을 4K로 늘렸다.
  - 수정 `dlssrange`. 2026-11-22 확인.
- **XC2**: 같은 증상에 3D도 없었다. 같은 수정이 적용된다(`RELEASE_2026-11-22.md` 1절). XC2 게임 확인 기록은 찾지 못했다(미확인).
- 0x0400E95004038001: 전용 64 GiB NX2 주소 영역을 예약한다(`k_system_control.cpp:133-136`). 이 ID의 이름은 코드 안에서 "XC2 3.0.1"과 "BF2"로 엇갈린다.
- 회귀 확인 대기: htarr/cidx/copymirror의 XCDE 영향.

출처: `docs\RELEASE_2026-11-22.md` 1절, 9절, 메모리 노트 `xcde-dlss-range-order.md`, `design\NCE_CLEANUP_REVIEW.md` 1절.

### 5.4 Mario Kart World (0400C3F00006E000)

- 소스에 있는 것:
  - 30 FPS / dynamic FPS 패치(`mkw_30fps`, `mkw_dynamic_fps` 설정, `nso.cpp:195-282`)
  - 이중 해제 probe(`NXN1X_NX2_PROBE`, 기본 꺼짐, "Temporary"로 표시)
- FPS 패치 UI 컨트롤은 그래픽 고급 패널에서 빠졌다. 설정 데이터는 유지한다.
- **불일치(작음)**: `nx2_probe_trap.h:41`은 probe가 꺼져 있어도 모든 프로세스에서 65,536칸 링을 정적 초기화한다(검수 추정 약 3.5 MiB).
- 현재 게임 상태 기록: 2026-11-22 배포의 htarr/cidx/copymirror 회귀 확인 대기 목록에 있다는 것뿐이다. 그 밖의 N1X 실행 기록은 찾지 못했다(미확인).
- `docs\nvn-memory\MKW_REFERENCE.txt`는 mkw-recomp 프로젝트의 메모리 설계 비교 자료다. 게임 상태 문서가 아니다.

출처: `design\NCE_CLEANUP_REVIEW.md` 1절, `docs\DLSS_ARM64_TEST_REPORT.md`, `docs\RELEASE_2026-11-22.md` 9절.

### 5.5 Hyrule Warriors: Age of Imprisonment (04003F40232DE000)

- 16 fps의 원인은 import 메모리의 정점 fetch였다. vbmir로 60 fps, GPU p50 10 ms, 프레임당 약 6 MB 복사, 미러 15개 1.27 GB.
- copy 모드에서는 GPU 이득이 없다(정점은 이미 미러). 이동 중 52-57 fps.
- 미해결 크래시: 0xc0000409(fast-fail 0xd, ntdll+0x34a10).
  - 오프닝 영상 중 1회, 재현 안 됨, 원인 미증명
  - 같은 형태가 Bayonetta 3에서 강제 종료 후 재실행 때 1회
- E코어 배치가 로딩 정지를 일으킨 적이 있다(`host-tuned-for-adreno.md`).
- NCE에 HWAoI 전용 우회가 있다: 풀 재시도 스텁, 진단 BRK, pooldiag(2.2.7절).
- 사용자가 `hyrule_30fps` / `hyrule_dynamic_fps` 옵션을 일부러 없앴다(근본 수정 원칙). 다만 `nso.cpp:284-341`의 frame-interval 전역 탐지(frame pacing 기능)는 남아 있다.

출처: 메모리 노트 `n1x-imported-vertex-fetch.md`, `bayo3-depth-pass-import-writes.md`, `root-fix-principle.md`, `design\HISTORY_20261120.md`, `design\NCE_CLEANUP_REVIEW.md`.

### 5.6 Fire Emblem Fortune's Weave (04007EF01F8AE000, v131072)

2026-11-22 배포에서 고친 것:

| 이름 | 증상 → 수정 | 스위치 |
|---|---|---|
| dirfs, maxopen | 덤프 폴더 무한 로딩 → FS 등록, CRT 파일 한도 | `NXN1X_DIRFS=0`, `NXN1X_VFS_MAXOPEN` |
| maxe4k | descriptor 1035개 파이프라인의 draw 건너뜀 → `EDEN_MAXE=4096` | 없음(상수) |
| htarr + cidx | 조명 텍스처 handle 계산 불가 32건 → 0건 | `NXN1X_HANDLE_TABLE_ARRAYS=0`, `NXN1X_HANDLE_CIDX=0` |
| copymirror | 캐릭터 카드 상하 반전 → `MIRROR_X/Y` 복사를 `vkCmdBlitImage`로, 확인 | `NXN1X_COPY_MIRROR=0` |
| progforget | finalize된 program이 큐 바인딩에 남음(nxnative 1440a73 이식) | `DM_VK_PROG_FORGET=0` |
| opthold, viewusage | 프레임 생성 관련 변경(device lost는 해결 안 됨) | `NXN1X_OPT_HOLD=0`, `NXN1X_VIEW_USAGE=0` |

열린 문제:
1. 튜토리얼 사각형이 콘솔보다 어둡다. 장면 휘도 137.25로 nxnative의 콘솔 일치값 134.4와 맞으므로 노출 1단계는 맞다. 3x1 노출 이미지는 아직 못 잡았다.
2. 시작부터 프레임 생성을 켜면 약 3초(tick 6)에 device lost. 끄고 시작하면 동작한다. 실행 중 F10으로 켜도 소용없다. 다음 단계는 컴포지터 큐의 NV checkpoint.

출처: `docs\RELEASE_2026-11-22.md` 5-9절, 메모리 노트 `fe-fortunes-weave.md`.

### 5.7 Bayonetta 3 (01004A4010FEA000)

- 2X import 모드에서 GPU 25-55 ms. 그중 depth-only pass 하나(1600x896, 500-850 draw)가 40-57 ms다. fragment 셰이더가 픽셀마다 import 풀 메모리에 `ssbo[objectId] = 65535`를 쓰기 때문이다.
- copy 모드에서 그 pass는 0.50-0.68 ms, GPU p50 7.4-9.0 ms, 프레임 18-20 ms. copy 모드 정점 깨짐은 vfix로 해결, 확인.
- copy 모드는 기본값이 아니다. stage 2가 기본값 전환의 전제였다.
- fstage는 셰이더당 present마다 16 draw 한도라서 Bayo3 depth pass(pass당 500+ draw)는 한도에 걸린다. pass 단위 창이 후속 과제다.
- 기록상 ID 불일치: 메모리 노트는 01004A4010FEA000, `NCE_CLEANUP_REVIEW.md`는 "Bayonetta 3 0400E88025B1A000"을 주석 근거로 언급한다. 두 ID가 다른 판(Switch 1판 / Switch 2 Edition)인지는 문서에 없다(미확인).

출처: 메모리 노트 `bayo3-depth-pass-import-writes.md`, `core-writable-bypasses-nvn-dirty.md`, `dk-bananza-perf.md`, `design\NCE_CLEANUP_REVIEW.md` 1절.

### 5.8 Switch 1 회귀 타이틀

- 2026-11-21 빌드에서 사용자는 Switch 1 게임이 잘 돌아간다고 했다. 원문 표현은 "젤다" 1/2와 Bayonetta 1/2다.
- Zelda 1은 occlusion 쿼리를 쓰지 않는 것으로 보인다(occlusion 변경의 대조군). Zelda 2는 쓰는 것으로 보인다.
- 그 밖에 쓰는 타이틀: HWAoI, DK, Pokémon Z-A(보류).
- 원칙: NVN1 경로에 닿는 기본값 전환(예: `DM_VK_OCC2=s3`)은 OoT와 Switch 1 타이틀 하나(Bayonetta 2 또는 Zelda)의 A/B 뒤에 한다.
- **불일치**: "Zelda 1/2"가 구체적으로 어느 타이틀인지(BotW/TotK 등) 메모리 노트에 없다(미확인).
- 기록된 기타 타이틀:
  - StarFox: 명령 메모리 abort 수정. 일회성 `vk::Exception(MasterSemaphore::Wait)`는 원인 미확정.
  - TotK(0100F2C0115B6000): s39 잠금 탈취 우회, `n1x_atomic`, s56 GPU→게스트 write-back(`DM_VK_S56`). 11-19 FG 탐지는 카메라 60개 관측, motion 후보 0으로 30 fps에 머묾.

출처: 메모리 노트 `regression-games.md`, `design\HISTORY_20261120.md` 4절, `docs\DLSS_ARM64_TEST_REPORT.md`, `design\NCE_CLEANUP_REVIEW.md`.

---

## 6. 설정·스위치·로그

### 6.1 데이터와 로그 위치

| 항목 | 위치 |
|---|---|
| 사용자 데이터 | exe 옆에 `user` 폴더가 있으면 포터블 모드, 없으면 `%APPDATA%\NXN1X` |
| 게임 로그 | `%APPDATA%\NXN1X\log\nxn1x_log.txt` (이전 실행은 `.old.txt`). 포터블 모드는 `user\log\nxn1x_log.txt` |
| crash ring | exe 옆 `crashring.bin`. 이전 실행의 링은 시작 시 이름을 바꿔 보관 |
| 크래시 보고 | exe 옆 `nxn1x_crash_<pid>.txt`, `.dmp`, `nxn1x_abort_<pid>.txt` |
| 파이프라인 캐시 | Eden `vulkan.bin`, `vulkan_pipelines.bin`, 드라이버 캐시 |
| DLSS-G 프로필 | `streamline/profiles/<16자리 소문자 hex 프로그램 ID>.fg` (검증된 게임 프로필은 배포에 없음) |
| 실행 스크립트 로그 사본 | `RUN_*.cmd`가 실행 뒤 로그를 `logs_poolcopy\`에 복사 |

- 세이브, 키, 펌웨어, 게임 파일은 배포물에 넣지 않는다.
- 진단 로그는 게임 타이밍에 영향을 줄 수 있다.

출처: `source\README.md`, 메모리 노트 `nxn1x-build.md`, `arm_nce_windows.cpp`, `nvnh_rt.c`, `docs\DLSS_ARM64_TEST_REPORT.md`, `RUN_DK.cmd`.

### 6.2 DM_CTL: 실행 중 스위치

- `DM_CTL=<파일>`로 실행하면 실행 중에 그 파일의 줄을 읽는다(`gfx_vk_exec.c:775`). `env K=V` 줄이 환경 스위치를 실시간으로 바꾼다. 예: `DM_VK_RC_OCC=0`, `NXN1X_VTX_PULL=1`은 다음 present부터.
- `DM_CTL_FROM_END=1`은 파일 끝부터 읽는다(`RUN_DK.cmd`).
- 저장소의 `ctl.txt`가 실행 스크립트용 제어 파일이다.
- `host\docs\env_switches.tsv`의 `DM_CTL` 설명은 "(Mac)"과 버튼, `shot`, `passdump`, `texdump`, `env KEY=VALUE`, `peek`, `memdump`, `drag` 명령을 적는다. 그중 어떤 명령이 Windows/NXN1X에서 동작하는지는 문서에 없다(미확인). `dlsstoggle` 빌드는 "DM_CTL line-count reader"를 추가했다.

출처: 메모리 노트 `nxn1x-build.md`, `RUN_DK.cmd`, `host\docs\env_switches.tsv`, `BUILD_LOG.txt`, `design\ASTRA_PHASE_B_REVIEW_REQUEST.md`.

### 6.3 주요 환경 스위치 (NXN1X)

아래는 2026-11-22 배포 문서, 설계서, 메모리 노트에 나온 스위치다. 원칙상 스위치는 A/B 측정과 회귀 시 되돌림용이고 해결책이 아니다(근본 수정 원칙).

**메모리 풀**

| 스위치 | 기본 | 뜻 |
|---|---|---|
| `NXN1X_POOL_MODE` | import | `import` / `copy` / `hybrid` |
| `NXN1X_VB_MIRROR` | 1 | `0`이면 vbmir 끔 |
| `NXN1X_VTX_PULL` | 0 | 버텍스 풀링 B단계 |
| `NXN1X_SSBO_WRITES` | — | `all`이면 모든 SSBO를 쓰기로 침(A/B) |
| `DM_VK_HY3` | 2 | atomic staging: 0 끔, 1 1단계, 2 창 묶기 |
| `DM_VK_FSTAGE` | 켬 | fragment storage staging |
| `DM_VK_GPUW_RANGE` | 켬 | gwrange |
| `DM_NVN_GPU_OWN_AFTER_CPU` | 켬(코드) | gpuown. 실행 스크립트는 0 |
| `DM_VK_KEPT_OVERWRITE_DROP` | 켬 | relovl |
| `DM_VK_S56` | TotK만 | GPU → 게스트 write-back |
| `DM_VK_PC_VERIFY` | 끔 | copy 모드 tracker와 Windows 보호 비교 |

**replay와 동기화**

| 스위치 | 기본 | 뜻 |
|---|---|---|
| `DM_VK_ASYNC` | 0 (DK 1) | 비동기 replay(s46) |
| `DM_VK_PARALLEL` | 0 (DK 4) | 병렬 기록 워커 수 |
| `DM_VK_PARALLEL_WKTEST` | — | 처음 N present는 워커 chunk를 replay 스레드에서 다시 기록 |
| `DM_VK_UPPOOL` | 켬 | 슬롯별 업로드 명령 풀 |
| `DM_VK_CBAR`, `DM_VK_CBAR2` | 켬 | barrier 생략·합치기 |
| `DM_VK_FIN_DEFER` | 켬 | texture finalize 지연 |
| `DM_VK_S46_POLICY` | — | replay 스레드 정책 진단(0-3) |
| `NXN1X_THREADCPU` | — | 스레드 CPU 진단 |
| `DM_VK_PROG_FORGET` | 켬 | progforget |

**occlusion과 렌더**

| 스위치 | 기본 | 뜻 |
|---|---|---|
| `DM_VK_OCC2` | 0 | `s1`/`s2`/`s3` 새 occlusion 경로 |
| `DM_VK_OCCDIAG` | 끔 | occlusion 계측 |
| `DM_VK_RC_OCC` | — | `0`이면 모든 occlusion 값을 unknown(→ draw) |
| `DM_VK_NAN_GUARD` | 켬 | fragment 출력 NaN 처리 |
| `NXN1X_COPY_MIRROR` | 켬 | 미러 복사 |
| `NXN1X_HANDLE_TABLE_ARRAYS`, `NXN1X_HANDLE_CIDX` | 켬 | handle table |
| `DM_VK_TBUF_FRESH` | NVN2 1 | GPU 쓰기 뒤 texel buffer 사본 재생성 |

**DLSS와 프레임 생성**

| 스위치 | 기본 | 뜻 |
|---|---|---|
| `DM_NVN_DLSS_RANGE` | 켬 | dlssrange |
| `DM_VK_OOT_DLSS` | — | OoT DLSS SR |
| `DM_VK_OOT_RESFIX` | 켬 | OoT 고정 장면 비율 |
| `DM_VK_NDSR` | 원본 | `0`이면 GSR1 |
| `NXN1X_OPT_HOLD`, `NXN1X_VIEW_USAGE`, `NXN1X_OPTFILL_ANY` | 켬 | 프레임 생성 보조 |
| `DM_VK_SWAPCHAIN_WAIT`, `DM_VK_CHECKPOINTS` | — | 스왑체인 대기, GPU checkpoint |

**CPU, 로더, FS, 해상도**

| 스위치 | 기본 | 뜻 |
|---|---|---|
| `DM_REALTIME_PATCH` | 켬 | DK·OoT realtime step 패치 |
| `NXN1X_TITLE_PATCH_PLAN` | 켬 | 타이틀/IPS 패치를 NCE 계획 전에 |
| `NXN1X_NCE_SELFTEST` | 끔 | 재개 규칙 self-test |
| `EDEN_NCE_STATS` | 끔 | NCE 종료 종류별 카운터, 상위 trap PC |
| `NXN1X_NS2_RES_1X` | 켬 | Switch 2 기본 1x |
| `NXN1X_DIRFS`, `NXN1X_FS_FAIL_LOG` | 켬 | 덤프 폴더 FS, FS 실패 로그 |
| `NXN1X_VFS_MAXOPEN` | — | 열린 파일 수 직접 지정 |
| `NXN1X_PCACHE_PERIODIC` | 끔 | 30초 주기 캐시 저장(예전 동작) |

**불일치**: `RELEASE_2026-11-22.md` 머리는 "Each change has one switch (environment variable)"라고 쓴다. 그러나 crash ring의 `NXN1X_CRASH_RING`은 환경 변수가 아니라 컴파일 정의다(`-DNXN1X_CRASH_RING=0`, `arm_nce_windows.cpp:1326`). `maxe4k`에는 끄는 스위치가 없다.

출처: `docs\RELEASE_2026-11-22.md`, `design\*.md`, 메모리 노트 전체, `source\src\core\nvn\nvn_layer.cpp`, `source\src\common\settings.cpp`.

### 6.4 env_switches.tsv

- `host\docs\env_switches.tsv`는 362개 항목이다(diagnostic 154, switch 122, path 54, player 23, test 9).
- 이 표는 nxnative(Mac/iOS Metal 호스트)의 스위치 목록이다. 항목 대부분이 Metal, iOS, Mac 전용이다. 예: `DM_NVN_TBDR`, `DM_NVN_ENC_THREAD`, `DM_PRESENT_LINK`, `DM_SHOW_FPS`.
- NXN1X에서 쓰는 `DM_VK_*`, `NXN1X_*` 스위치는 거의 없다. `DM_VK_CAPTURE`와 `DM_VK_LOG`도 Vulkan-on-Metal 레이어용이다.

공통 NVN 레이어(`hle_nvn.c`)에서 쓸 수 있는 것으로 보이는 진단:
- `DM_NVN_TSV`(프레임별 NVN 호출 표, XCDE 조사에 사용)
- `DM_NVN_COUNT`, `DM_NVN_FRAMES`, `DM_NVN_PASSDUMP`, `DM_NVN_CMDSTAT`, `DM_NVN_CBLOG`, `DM_NVN_WAITS`

다만 Windows Vulkan 백엔드에서 각각 동작하는지는 표에 적혀 있지 않다(미확인).

출처: `host\docs\env_switches.tsv`.

### 6.5 진단 도구

| 도구 | 사용 | 내용 |
|---|---|---|
| pass 시간 | `DM_VK_PASS_TIMES=1` | r138 "GPU busy" 줄, pass별 시간. 켜면 S1e 병렬 segment가 모두 거절됨 |
| ootgap / longframe | (로그) | GPU span = timed pass + part 경계 + gap 분해 |
| dispatch trace | `DM_VK_TRACE_DISPATCH` | dispatch 추적(XCDE 조사) |
| image watch | `DM_VK_WATCH` | 이미지 쓰기 감시(겹침 인식) |
| NVN 게이트 trace | `NXN1X_NVN_GATE_TRACE=1` | 게스트 코드 덤프. 디스어셈블해 호출 의미 확인 |
| NVN 호출 시간 | `NXN1X_S51=1`, `NXN1X_S49=1` | 메인 스레드를 호스트 NVN 호출과 게스트 코드로 분리 |
| NCE 통계 | `EDEN_NCE_STATS=1` | 종료 종류별 수와 상위 trap PC(`windows_nce_stats.h`) |
| gpuck | (빌드 기본) | device lost 때 GPU가 있던 pass와 draw 보고 |
| hangstk | (sv1 첫 "no present" 덤프) | 모든 호스트 스레드 PC/LR/프레임 체인, PDB로 이름 |
| sv1 | (NVN 경로) | present가 멈추면 게스트 스레드 덤프(로그만) |
| cpudiag / thrname | (진단 빌드) | 스레드별 CPU, nnsdk 스레드 이름 |
| crash ring | 기본 | 2.2.6절 |
| fsfail | 기본 켜짐 | 실패한 게스트 FS 경로 |
| handletree | `NXN1X_HANDLETREE=1` | 계산할 수 없는 bindless handle 식 |
| fmtdiag | `NXN1X_FMTDIAG=1` | fragment가 샘플링하는 텍스처의 NVN/Vulkan 형식 |
| expoprobe | `NXN1X_EXPOPROBE=1` | 휘도와 노출 값, 2초마다 |
| stripediag | F3 | 마지막 present 3개의 이미지 정보 덤프, packaged 텍스처 재업로드 |
| FG 입력 캡처 | Ctrl+Shift+F11 (`tools\test_ndsr.ps1 -FrameGeneration -CaptureFGInputs`) | 2프레임 32x32 샘플, UBO 4096바이트 |
| flashdiag(2) | (OoT) | DLSS 입력·출력의 밝기 급변 |
| copy self-test | `RUN_POOLCOPY_SELFTEST_COPY.cmd` | HYBRID / COPY 셀프테스트 |
| DLSS probe | `tools\dlss_probe.cpp` | Streamline 기능 probe, SR 오프스크린 readback |

같은 원칙이 진단에도 적용된다. 로그로 답할 수 있는 A/B를 따로 요청하지 않는다. 검증은 빌드 안의 런타임 검사, 감지 로그, 자동 폴백으로 한다.

출처: `docs\RELEASE_2026-11-22.md` 8절, 메모리 노트 `dk-bananza-perf.md`, `xcde-dlss-range-order.md`, `feedback-no-busywork.md`, `BUILD_LOG.txt`, `docs\DLSS_ARM64_TEST_REPORT.md`.

### 6.6 단축키와 UI

| 키 / 위치 | 동작 |
|---|---|
| F7 | DLSS SR 켜고 끄기 |
| F10 | 프레임 생성 켜고 끄기(시작 때 켜지 않으면 STORAGE usage가 없어 FE device lost에는 소용없음) |
| F3 | stripediag 텍스처 덤프 |
| Ctrl+Shift+F11 | FG 입력 캡처(진단 실행) |
| 보기 → Show Performance Overlay | FPS, 프레임 시간, CPU%, GPU%, DLSS/FG 행 |
| 그래픽 고급 | DLSS SR, Frame Generation, Frame Generation source, HDR 체크박스 |

- HDR 체크박스는 기존 HDR 정책을 쓴다. presenter는 여전히 SDR 스왑체인을 선호하므로 HDR10 출력이 보장되지 않는다.
- NVN 경로에서는 에뮬레이션 GPU 전용 컨트롤(텍스처 디코딩/재압축, anisotropy, buffer cache, GPU unswizzle)을 숨긴다.

**불일치(오버레이 의미)**: 오버레이와 s67 로그의 "GPU" 값(`mtl_gpu_ms`)은 submission timestamp 구간의 합이다. part 사이 GPU 대기가 포함되므로 실제 GPU 작업은 r138 "GPU busy"(`DM_VK_PASS_TIMES=1`)로 봐야 한다. 한편 `source\README.md`는 GPU%를 "가장 바쁜 Windows GPU 엔진 값"이라고 정의한다. 둘은 다른 지표다.

출처: `BUILD_LOG.txt`(dlsstoggle, stripediag), 메모리 노트 `oot-dlss-plan.md`, `dk-bananza-perf.md`, `docs\DLSS_ARM64_TEST_REPORT.md`, `source\README.md`.

---

## 7. 빌드

### 7.1 현재 방법

```powershell
$env:PATH = "C:\msys64\clangarm64\bin;C:\msys64\usr\bin;$env:PATH"
ninja -C build/release                      # 또는: ninja -C build/release bin/NXN1X.exe
```

- 결과물: `build\release\bin\NXN1X.exe`.
- 툴체인: MSYS2 clangarm64(Clang 23.1.5로 구문 검사한 기록), CMake/Ninja.
- 빌드 머신에 git이 없다. 변경 원본은 `_backup\<태그>\`에 보관하고, 빌드마다 `BUILD_LOG.txt`에 한 줄 남긴다.
- NXN1X가 실행 중이면 exe를 바꿀 수 없다. 빌드는 NXN1X가 꺼져 있을 때 한다. 같은 폴더를 다른 작업 스레드와 공유하므로 남의 빌드 위에 덮어 빌드하지 않는다.
- CMake 옵션:
  - `NXN1X_DIAGNOSTICS`(기본 OFF): 조사 trace와 상세 비용 카운터
  - `NXN1X_CRASH_RING`(컴파일 정의, 기본 1)
  - `NXN1X_STREAMLINE_SDK`: Streamline 연결
  - `ENABLE_LSFG`: 광학 흐름 빌드는 OFF로 빌드한 기록이 있다
- NVN 호스트는 `NVN_HOST_DM`(기본 `../host/dm`)에서 빌드한다. `hle_nvn.c`가 없으면 구성이 멈춘다. 게임 프로필 헤더(`game_profile.h`)를 생성한다.
- Streamline: 공식 ARM64 SDK 2.14.16을 `build/sl-sdk`에 풀고, 런타임 DLL 6개를 `build\release\bin\streamline\`에 둔다. SDK의 license 파일을 함께 보관한다.

출처: 메모리 노트 `nxn1x-build.md`, `root-fix-principle.md`, `dk-bananza-perf.md`, `docs\DLSS_ARM64_TEST_REPORT.md`, `docs\DLSS_ARM64_CAPABILITIES.md`, `docs\NCE_CLEANUP_AUDIT_2026-11-22.md`, `source\CMakeLists.txt`, `source\src\nvn_host\CMakeLists.txt`, `design\HISTORY_20261120.md` 6절.

### 7.2 작업 규칙 (`CLAUDE.md`)

- 조사용 빌드와 단계별 빌드로 나누지 않는다. 이전 구현(refer, Eden, nxnative, Longshot, NX2navi)이나 로그가 원인·방법을 이미 보여 주면 진단과 변경을 한 빌드에 넣는다.
- 위험한 부분은 그 한 빌드 안에서 세 가지로 지킨다.
  - 실행 중 검사
  - 검사 실패 시 원래 경로로 돌아가고 이유를 로그에 남김
  - 변경을 끄는 스위치 하나
- 그 밖의 운영 원칙(HISTORY 6절, 메모리 노트):
  - 근본 원인만 고친다. 30 fps 모드, 재시도, sleep, stub 같은 우회를 해결책으로 쓰지 않는다.
  - 원인을 증거로 확인한 뒤 고친다.
  - 다른 프로젝트(nxnative, BananaNative 등)가 이미 고친 문제는 그 수정을 먼저 대응시켜 이식한다.

출처: `CLAUDE.md`, `design\HISTORY_20261120.md` 6절, 메모리 노트 `root-fix-principle.md`, `feedback-one-build.md`, `feedback-port-known-fixes.md`.

### 7.3 예전 빌드 도구 (Adreno 시기)

- `tools\build_nx2.py`: CMake/Ninja + MSYS2 ARM64(루트 `README.md`).
- `tools\build_turnip_f1r.py`, `build_turnip_f2r.py`: Turnip/Mesa 증분 빌드. 다른 donor 빌드의 오브젝트에 의존하므로 깨끗한 빌드 명령이 아니다.
- **불일치**: `source\README.md`는 `tools/build.ps1`(`-Diagnostic`)를 빌드 도구로 적는다. 그러나 현재 `tools\`에 `build.ps1`이 없다. 정식판/진단판 두 exe(`bin/NXN1X.exe`, `bin/NXN1X-Diagnostic.exe`) 구성도 현재 빌드 기록과 맞지 않는다.

출처: `README.md`, `source\README.md`, `tools\` 목록.

---

## 8. 알려진 문제와 다음 과제

### 8.1 알려진 문제

| # | 문제 | 상태 | 근거 |
|---|---|---|---|
| 1 | FE 튜토리얼 사각형이 콘솔보다 어둡다 | 노출 이미지 캡처 필요 | `RELEASE_2026-11-22.md` 9절 |
| 2 | FE에서 시작부터 프레임 생성을 켜면 device lost | 미해결. 다음은 컴포지터 큐 NV checkpoint | 같음 |
| 3 | htarr/cidx/copymirror가 다른 Switch 2 게임(MKW, DK, XCDE)에 미치는 영향 | 회귀 확인 대기 | 같음 |
| 4 | OoT Deku Tree 바닥 구멍(gpuown) | 미뤄짐. 스크립트에서 gpuown 끔 | `oot-deku-terrain-gpuown.md` |
| 5 | OoT 줄무늬 텍스처 | 다음 주. F3 덤프 대기 | `oot-wrapup-friday.md` |
| 6 | OoT occlusion s3 비용(31-40 대 약 47 fps) | occ-s3fast 진행 중. 기본값 0 | `oot-occlusion-s3.md` |
| 7 | OoT device lost 3회 | 원인 미확정 | `oot-dlss-plan.md` |
| 8 | 시작 후 파이프라인 생성 끊김(OoT 약 1,800개) | prewarm 예정 | `pipeline-prewarm-todo.md` |
| 9 | DK 게임플레이 replay CPU 병목(replay 스레드 21-23 ms) | part 경계 대기가 다음 주 과제 | `dk-bananza-perf.md` |
| 10 | 0xc0000409 fast-fail 0xd(HWAoI 오프닝 1회, Bayo3 재실행 1회) | 원인 미증명 | `n1x-imported-vertex-fetch.md` |
| 11 | StarFox 일회성 `vk::Exception(MasterSemaphore::Wait)` | 진단 추가, 원인 미확정 | `HISTORY_20261120.md` |
| 12 | 무한 로딩 | sv1 덤프가 있는 로그가 아직 없음 | 같음 |
| 13 | T15: GPU 쓰기 진행 중 CPU가 옛 값을 다시 쓰는 경우 | 알려진 한계로 문서화 | 같음 |
| 14 | s48 range import 130-180 ms 튐, 장면 진입 2-3 s | 손대지 않음 | `OOT_BARRIER_DESIGN.md` 16절 |
| 15 | DLSS SR 동기화 검증 오류 2건(NGX Evaluate 내부) | 11-19 probe 기록. 이후 재검증 기록 없음 | `DLSS_ARM64_CAPABILITIES.md` |
| 16 | BreakRescue가 release에서 항상 켜짐(주석과 다름) | 결정 필요(P2) | `NCE_CLEANUP_REVIEW.md` |
| 17 | HDR 체크박스가 HDR10 출력을 보장하지 않음 | — | `DLSS_ARM64_TEST_REPORT.md` |
| 18 | Vulkan validation layer가 노트북에 없음 | 동기화 검증 제한 | `dk-bananza-perf.md` |

### 8.2 다음 과제

1. DK: replay part 경계 대기를 줄이고, agl_ndsr 자리에 DLSS SR을 넣는다(다음 주).
2. OoT: 2026-11-27에 추가 버그와 다른 프로젝트 수정 사례를 받는다. 줄무늬, Deku 지형(gpuown, TotK s56 같은 write-back 확인), occ-s3fast, s3 기본값 전환(OoT + Switch 1 A/B 뒤)이 남아 있다.
3. 파이프라인 prewarm: nxnative의 `prewarm_start`와 Eden `vulkan.bin` 목록을 재사용한다(새 기계 장치를 만들지 않음).
4. 메모리: 혼합 구조(버텍스 풀링 C/D단계, fstage의 pass 단위 창(Bayo3), 버텍스 풀링 기본값 판단)를 진행한다.
5. 정점 미러 개선안: draw가 읽는 범위만, 제출 전 GPU 복사, fence까지 수명 유지. 메모리 노트에는 "승인 대기"로 남아 있다.
6. NCE: P7 주입 테스트, STP/exclusive 재실행 테스트, 정리 제안 P2-P6.
7. Adreno 시기 기본값 재점검: `nvnh_rt.c`의 타이틀 키, `nvn_layer.cpp` NVN2 기본값(`TU_GS_ONCE` 등), Turnip 조건 코드, fp16 규칙.
8. FE: 노출 이미지 캡처, FG device lost용 컴포지터 큐 checkpoint.

### 8.3 문서 간 불일치와 공백

| # | 내용 | 관련 파일 |
|---|---|---|
| 1 | `source\README.md`는 Adreno + NPU 프레임 생성 + `tools/build.ps1` + 정식/진단 두 exe를 설명한다(옛 내용; 제목만 2.0으로 맞춤). 현재 대상(N1X), 빌드(`ninja -C build/release`), 프레임 생성(DLSS-G/광학)과 다르다. `tools\build.ps1`은 없다 | `source\README.md`, `tools\` |
| 2 | 루트 `README.md`는 Turnip/Mesa, LSFG가 있는 스냅샷이라고 설명한다. 현재 N1X 빌드와 관계가 문서에 없다 | `README.md` |
| 3 | `host\docs\`(development, playing, settings, frontends, env_switches.tsv, reference)는 nxnative(iOS/Mac, Metal) 문서다. NXN1X Windows용 사용자 문서는 없다 | `host\docs\*` |
| 4 | 코드 주석이 가리키는 `NVN_HOST_BRIDGE.md`, `docs/lane_b/SHADER_EDEN_B.md`, `docs/lane_a/A1_BUILD.md`, `docs/lane_a/apply_sv1.py`가 저장소에 없다 | `nvn_gate.cpp`, `nvn_shader_bridge.h`, `nvnh_rt.c`, `sv1.cpp` |
| 5 | `DLSS_ARM64_ARCHITECTURE.md`의 "SR and FG are off in the game"은 이후 구현과 맞지 않는다(초기 감사 기록) | 2.6.2절 |
| 6 | `DLSS_ARM64_TEST_REPORT.md` 안에서 FG 설정 선택지가 두 가지로 적혀 있다 | 2.6.3절 |
| 7 | `OOT_DLSS_DESIGN.md`의 카메라 위치(c3 0x3500)와 메모리 노트의 구현(UBO0 ≥0x230, Longshot layout)이 다르다 | 4.13절 |
| 8 | `PARALLEL_REPLAY_DESIGN.md`는 OoT를 NVN1 타이틀로, 다른 문서는 NVN2로 적는다. 상태 줄도 실행 결과로 갱신되지 않았다 | 4.14절 |
| 9 | `OCCLUSION_REDESIGN.md` 머리와 6절 제목은 s2/s3를 "syntax-checked only, not run"으로 적지만 메모리 노트에는 s3 게임 확인이 있다 | 4.11절 |
| 10 | 0x0400E95004038001이 코드에서 "XC2 3.0.1"과 "BF2"로 다르게 불린다 | `arm_nce_windows.cpp`, `k_system_control.cpp` |
| 11 | Bayonetta 3 ID가 01004A4010FEA000(메모리)과 0400E88025B1A000(NCE 검수의 주석 근거)으로 다르다. 같은 게임의 다른 판인지 문서에 없다 | 5.7절 |
| 12 | BreakRescue 주석("기본 꺼짐")과 코드(release 항상 켜짐)가 다르다 | `windows_cross_thread_break.h` |
| 13 | RELEASE 문서는 "모든 변경에 환경 변수 스위치"라고 하지만 `NXN1X_CRASH_RING`은 컴파일 정의이고 `maxe4k`에는 스위치가 없다 | 6.3절 |
| 14 | `BUILD_LOG.txt`는 optfill(11-22 약 08:40)에서 끝난다. 11-22 배포 항목(dlssrange, NCE 정리, dirfs, maxe4k, htarr 등)이 없다 | `BUILD_LOG.txt` |
| 15 | GPU 값의 의미가 두 가지다: 오버레이/s67 "GPU"(submission 구간 합, CPU replay 포함)와 README의 GPU%(Windows GPU 엔진) | 6.6절 |
| 16 | `nce_gpu_path` 코드 기본값은 Emulation이다. 배포 프로필에서 NVN 경로가 기본인지 문서가 없다 | 1.3절 |
| 17 | 버텍스 풀링 C단계(게임 비교) 결과, XC2 게임 확인, MKW의 N1X 실행 기록을 찾지 못했다 | 4.7절, 5.3절, 5.4절 |
| 18 | NXN1X 고유의 오디오·동영상 문서가 없다(Eden 서비스로 추정) | 2.7절 |
| 19 | `docs\nvn-memory\`의 작업 공간 경로(`Documents/ChatGPT/NXN1X`)와 전제(12 GB 장치, Turnip)는 현재 트리와 다르다. 대부분 runtime pending으로 끝났다 | `docs\nvn-memory\FOUNDATION_DESIGN.txt` |
| 20 | Switch 1 회귀 타이틀 "Zelda 1/2"가 구체적으로 어떤 게임인지 기록이 없다 | `regression-games.md` |

---

## 9. 변경 이력

| 날짜 | 문서 | 내용 |
|---|---|---|
| 2026-11-16~18 | `docs\nvn-memory\` | Adreno 시기 NVN 메모리 기반 작업(F2-F5Z) |
| 2026-11-19 | `docs\DLSS_ARM64_CAPABILITIES.md`, `docs\DLSS_ARM64_TEST_REPORT.md` | N1X 기능 probe, Streamline 통합, FG 입력 자동 탐지, 광학 흐름 보간 |
| 2026-11-19~20 | [`design\HISTORY_20261120.md`](../design/HISTORY_20261120.md) | import 메모리 측정, vbmir, copy 모드 1·2단계, NCE 크래시 수정, StarFox 명령 메모리 |
| 2026-11-20~21 | `design\*.md` | hybrid, atomic staging, 버텍스 풀링, occlusion, barrier, OoT DLSS, 병렬 replay 설계 |
| 2026-11-21 | `BUILD_LOG.txt` | OoT DLSS(ootdlss7~11), resfix, fgoverlay/fgsource, nanguard, flashdiag, cbguard, upswap |
| 2026-11-22 | `BUILD_LOG.txt`, 메모리 노트 | DK: fstage, hangstk, parck, uppool, dkpar, optfill |
| 2026-11-22 | [`docs\RELEASE_2026-11-22.md`](RELEASE_2026-11-22.md) | 첫 배포: dlssrange, NCE 정리와 로더 순서, ns2res1x, 아이콘, 덤프 폴더 부팅(dirfs, maxopen, fsfail), make_nup.py, FE 수정(maxe4k, htarr/cidx, copymirror, progforget, opthold, viewusage) |
| 2026-11-22 | [`docs\NCE_CLEANUP_AUDIT_2026-11-22.md`](NCE_CLEANUP_AUDIT_2026-11-22.md) | NCE 정리 사전 점검 |

빌드별 상세는 `BUILD_LOG.txt`(빌드당 한 줄, 시간은 UTC)에 있다. 변경 전 원본은 `_backup\<태그>\`에 있다.
