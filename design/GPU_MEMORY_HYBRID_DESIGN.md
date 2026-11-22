# GPU 메모리 혼합(hybrid) 설계: backing 범위별 import / copy

2026-11-20. NPARK 결정(20:32, 20:35): 코드 전에 설계부터. 범위는 짧게 잡는다.

## 0. 왜

- copy 모드(device-local 사본)는 GPU 쪽 import 저속을 없앤다. 바요3 depth pass는 40-57 ms에서 0.6 ms가 됐다.
- 대신 copy 모드에는 CPU 쓰기 추적(쓰기 보호, fault, 재보호) 비용이 든다. HWAoI는 이동 중 fault가 20-35k/s, mprotect가 30-50k/s(약 300 ms/s)이고, 이것이 이동 중 fps 하락의 주원인이다(19:02/20:13 로그).
- import 모드에는 그 비용이 없다. import 메모리는 GPU가 게임 페이지를 직접 읽으므로 사본을 맞출 필요가 없다. 대신 GPU의 특정 접근이 느리다(1절).
- 그래서 느린 접근을 하는 범위만 copy로, 나머지는 import로 둔다.

## 1. 전제: N1X 드라이버에서 import가 느린 접근 (nvidia_report\membench, 2026-11-20)

| 접근 | import / device-local | 근거 |
|---|---|---|
| 고정 기능 vertex/index fetch | 100-250배 느림 | results.txt: 950 µs 대 3.8 µs, 2814 µs 대 14.3 µs |
| storage buffer 읽기 | 같음 | results.txt: 4.0 µs 대 3.8 µs |
| shader 저장(SSBO store) | 약 1.8배 | 앞선 membench(요약 기록) |
| atomic | 약 25배 | 같음 |
| 같은 주소 반복 읽기 | 약 36배 | 같음 |
| import → device-local 복사 | 125-160 GB/s | results.txt |
| **색/깊이 attachment 쓰기** | **미측정** | 2절 |
| **transfer dst(copy/fill이 import에 씀)** | **미측정** | 2절 |

- 게임 실측: 바요3 depth pass(fragment shader가 픽셀마다 SSBO에 씀)는 import에서 40-57 ms, copy에서 0.6 ms였다.
- HWAoI의 vertex fetch는 이미 vbmir(device-local mirror)로 해결돼 있다(16 → 30 fps).

### 1.1 전제 검증: 게임별 backing 범위 분류 — **미검증**

혼합의 효과는 "CPU fault가 GPU가 느리게 쓰지 않는 범위에 몰려 있는가"에 달려 있다.

- 이미 있는 숫자: combined1120 빌드(20:35, 아직 실행 전)의 로그 줄 `combined1120, last 120 submissions`가 "fault난 페이지 중 GPU 기록 페이지 대 CPU 전용 페이지" 비율을 센다. 단, 페이지 단위이고 GPU 쓰기 종류는 구분하지 않는다.
- 더 필요한 카운터. 다음 빌드에서 넣고, 지금은 넣지 않는다:
  1. backing 범위(256 MB pcr와 private 청크)별로 이번 실행에서 GPU 쓰기가 몇 번 바인딩됐는지, 종류별로 센다: shader store(SSBO/image store), atomic, 색/깊이 attachment, transfer dst(copy/fill/ubd), transform feedback. 바인딩 지점은 gfx_vk_exec.c와 gfx_vk_eden.inc의 sub_gpu_wrote 호출부와 attachment 바인딩이다.
  2. 같은 범위의 고정 기능 VB/IB 사용 횟수(vbmir 대상)
  3. fault를 범위의 분류별로 집계: "GPU 쓰기 없음", "transfer만", "shader/atomic/attachment"
- 판단 기준(HWAoI 이동 구간): fault의 60% 이상이 "GPU 쓰기 없음" 또는 "transfer만" 범위에 있어야 혼합이 의미가 있다. 이 숫자를 얻기 전에는 전제가 성립하는지 모른다.

## 2. 측정 공백

membench에 두 항목을 추가해 import 대 device-local로 측정한다.
- (a) 색과 깊이 attachment 쓰기(render pass store)
- (b) vkCmdCopyBuffer와 vkCmdFillBuffer의 dst가 import인 경우
- 결과에 따라 규칙 3절의 "느린 쓰기" 목록이 정해진다.

## 3. 규칙

- **단위는 backing 범위입니다.** pool이 아닙니다. alias(같은 페이지의 여러 VA)가 사본을 공유하기 때문입니다(pcr: 256 MB 정렬 범위). private 청크는 그 자체가 단위입니다.
- **시작은 import**입니다. 지금의 import 경로와 vbmir/vbvol을 그대로 씁니다. import 범위에는 쓰기 보호를 걸지 않습니다. vbmir이 쓰는 블록 단위 추적만 남습니다.
- **copy로 바꾸는 조건**: 그 범위에 "느린 GPU 쓰기"(shader store, atomic, 그리고 2절에서 느리다고 나온 attachment 쓰기)가 처음 바인딩될 때입니다. transfer dst와 shader 읽기만 있는 범위는 import에 둡니다(2절 결과에 따라 조정).
- **전환 시점**: 그 바인딩을 기록하는 제출의 경계. 실제로는 다음 제출 시작이고, 2.3절 참조.
- **한 번 바꾸면 그 실행 동안 copy로 고정**합니다(단방향). 이유:
  - 되돌리면 그 범위의 GPU 결과를 모두 병합하고 보호를 풀고 사본을 버려야 합니다. 비용이 크고, 경계 상태가 하나 더 생깁니다.
  - 느린 쓰기를 한 번 한 범위는 대개 매 프레임 다시 씁니다(바요3 depth pass, 계산 버퍼).
  - 범위 수가 적어서(pcr 256 MB 단위) copy로 고정해도 메모리 걱정이 없습니다.

### 3.1 raw global memory(bindless 전역 저장)

**정정 (11-20 23:50, 외부 검토 IMPORTED_MEMORY_ISSUES 4.5절 확인)**: 아래 원래 문제 설명의 원인은 틀렸습니다.
- 그 실행(logs_poolcopy\ab_copy_205455.txt)의 192.35초 창 값은 `stage2c … pre copies 332 (723.0 MB), post 723.0 MB`입니다.
- 같은 창의 `raw global memory stores … 0 write logs`도 0입니다.
- 실행 전체에서 `raw global memory bound (every chunk marked GPU-written) N times`의 N이 0이 아닌 줄은 하나도 없습니다. 그래서 pc_pre_all(gfx_vk_core.c vk_gmem_map, vk_gml_note)은 불리지 않았습니다.
- 723 MB는 120번 제출 누계이고, 프레임당 양이 아닙니다.
- 실제 경로는 compute dispatch의 storage buffer입니다. gfx_vk_eden.inc:612(`if (st == 5) vk_gpu_wrote(a, n)`)와 gfx_vk_exec.c:2720이 is_written과 관계없이 모든 SSBO를 쓴 것으로 칩니다.
  - 크기를 모르면 청크 끝까지 칩니다.
  - 이것이 vk_gpu_wrote → sub_gpu_wrote → pc_pre_note로 pre/post를 만들고, g_gw와 vbmir 미러 무효화도 일으킵니다.
  - 이 실행의 프레임에는 compute pass가 많습니다. 17 pass 중 14개가 dispatch입니다.
- 읽기 전용 바인딩의 몫은 hybrid1 빌드의 진단 줄 `hybrid1 binds`로 셉니다. 그것을 빼는 수정은 숫자를 본 뒤 NPARK 승인을 받아서 합니다.
- 아래 "raw 바인딩만으로는 바꾸지 않는다"는 규칙은 그대로 맞습니다. 혼합의 전환 기준은 "is_written=1 바인딩의 atomic"입니다. compute라서 쓴 것으로 치는 가정은 전환 기준에 쓰지 않습니다.
- Eden 모드(기본)에는 raw 전역 메모리의 쓰기 로그(DMSH_GMEM_LOG)가 없습니다. 그 로그는 DM_VK_EDEN=0 경로에만 있습니다. 그래서 Eden의 raw atomic은 "쓴 곳 모름 → 전환 안 함"(H11 규칙)입니다.

**원래 문제 설명(원인 부분은 위 정정으로 대체)**

- (틀림) 이 게임은 raw global memory를 바인딩합니다(로그: "raw global memory bound (every chunk marked)"). → 그 카운터는 실행 내내 0이었습니다.
- copy 경로가 120번 제출당 723 MB를 pre/post 복사한 것은 사실입니다. 원인은 위의 compute SSBO입니다.
- 그 결과:
  - F0 copyback 한 번에 약 200 ms
  - mprotect 41.5k/s
  - 3D가 약 4 fps(프레임 240 ms, GPU 43 ms). 타이틀 영상은 60 fps입니다.
- 혼합 설계에서 "raw 바인딩 = 모든 범위가 느린 쓰기를 받을 수 있음"으로 보면, 첫 raw 바인딩에 모든 범위가 copy로 바뀌고 같은 비용이 생깁니다. 그래서 그렇게 하면 안 됩니다.

**선택지**

| | 방법 | 장점 | 단점 |
|---|---|---|---|
| (a) | raw로만 닿는 범위는 import에 둠 | 동기화 비용 0. raw 저장은 import에 직접 감 | raw store가 import에서 느림(SSBO store 기준 약 1.8배), raw atomic이면 약 25배 |
| (b) | 실제로 쓰인 범위만 찾아서 그 범위만 copy | 정확 | 쓰인 범위를 아는 방법이 필요 |
| (c) | 게임별 측정으로 결정 | 단순 | 게임별 규칙(원칙상 피함), 새 게임마다 측정 필요 |

**권장: (a)를 기본으로, (b)는 이미 있는 write log로만 합니다.**

- 근거 1: raw 저장이 import에서 받는 손해는 SSBO store 기준 약 1.8배입니다. raw 때문에 생기는 copy 비용(청크 전체 pre/post 723 MB, 3D 4 fps)은 그보다 몇 자릿수 큽니다.
- 근거 2: 이미 raw 저장을 하는 셰이더는 쓴 페이지를 write log(DMSH_GMEM_LOG, gml)에 남깁니다. 지금도 그 로그로 copy-back 대상을 정합니다. 그래서 "실제로 쓰인 범위"는 새 장치 없이 알 수 있습니다.
- 규칙:
  - raw 바인딩만으로는 어느 범위도 바뀌지 않습니다.
  - raw 셰이더 중 **전역 atomic**을 쓰는 것이 있으면(셰이더 정보, dmsh facts), 그 셰이더의 write log에 나온 페이지가 속한 범위만 다음 제출에서 copy로 바꿉니다. atomic이 import에서 약 25배 느리기 때문입니다.
  - 일반 raw store만 쓰는 범위는 import에 남습니다.
  - write log가 넘친 경우(쓴 페이지를 다 모름)에는 바꾸지 않습니다. import는 정확성에 문제가 없고 속도만 손해입니다.
- copy로 바뀐 범위에 raw 저장이 오면, 지금처럼 그 청크 전체를 pre/post 대상으로 합니다. 하지만 바뀐 범위만이므로 양이 작습니다.
- 확인할 숫자(다음 빌드 카운터):
  - raw 바인딩 횟수
  - write log에 나온 페이지 수와 범위 분포
  - 전역 atomic을 쓰는 raw 셰이더 수
- (정정) OoT의 723 MB pre/post는 raw가 아니라 compute SSBO 때문이었습니다. 위 정정을 보세요. OoT import 모드의 저속(dispatch 0xd987a9de 앞 122 ms)은 OoT 스레드가 원인을 찾았습니다. import pool에 f32 atomic add를 CAS 반복문으로 하는 것이 원인입니다. 그래서 atomic 예외 경로(hybrid1)의 첫 검증 대상입니다.
- 다만 지금도 import 모드(RUN_AB_IMPORT.cmd)로 OoT를 돌려 비교하면 이것을 바로 확인할 수 있습니다.

## 4. 전환과 일관성

어떤 바이트도 두 모드에서 동시에 보이면 안 됩니다.

1. **전환 절차**: 범위 R에 느린 쓰기 바인딩이 나오면, 그 제출 X는 R을 import로 그대로 씁니다. 바인딩을 기록 중인 명령 버퍼는 R의 import 버퍼를 가리키고 있습니다. R에 "다음 제출부터 copy" 표시를 남깁니다.
2. **다음 제출 시작**: 앞선 모든 제출의 완료를 기다립니다. 지금의 drain과 같고, 혼합 전환 한 번에 한 번뿐입니다. 그 뒤:
   - R의 사본을 만듭니다(pc_make).
   - 게임 페이지에서 전체 동기화합니다(기존 full-resync 경로).
   - 보호를 겁니다.
   - 이후 바인딩은 사본 버퍼로 갑니다(vk_buf_at이 R을 copy로 봄).
3. **진행 중 제출**: 2번의 drain 덕분에, 전환 시점에 R을 import로 쓰던 제출은 모두 끝났습니다. 그 GPU 쓰기는 import라 이미 게임 페이지에 있습니다.
4. **import 범위로의 GPU 쓰기**(copy, fill, transfer): 지금과 같이 게임 페이지에 직접 씁니다. 사본이 없으니 병합할 것도 없습니다.
5. **counter/event 보고**: 완료 스레드의 쓰기는 모드와 무관합니다. combined1120의 "뒤 GPU 쓰기 바이트 보존"은 두 모드에 모두 적용됩니다.
6. **CPU 쓰기 추적**: copy 범위만 필요합니다(dirty 4 KB, volatile, fault-ahead). import 범위의 페이지는 pool 동기화를 위해 보호하지 않습니다. nvn_dirty의 텍스처와 vbmir용 보호는 지금처럼 남습니다.
7. **alias**: 판정과 사본은 backing 범위 단위이므로, 같은 페이지를 가리키는 모든 VA가 함께 바뀝니다.

## 5. 되돌리기와 드라이버 재확인

- **스위치 하나로 전부 import**: `NXN1X_POOL_MODE=import`이면 모든 범위가 처음부터 끝까지 import입니다. 지금의 import 경로와 같고 다른 차이가 없어야 합니다(테스트 H1).
- `NXN1X_POOL_MODE=copy`(전부 copy, 지금 동작)는 A/B용으로 둡니다. 새 기본값은 `hybrid`입니다.
- **드라이버 버전 확인**:
  - 시작할 때 드라이버 버전(VkPhysicalDeviceProperties.driverVersion)을 기록합니다.
  - 새 드라이버가 나오면 nvidia_report\membench(2절 항목 포함)를 다시 돌립니다.
  - shader store, atomic, 고정 기능 fetch가 device-local과 비슷해지면(예: 1.5배 이내), 그 드라이버 버전에서는 copy 전환을 자동으로 건너뜁니다. 표에 등록된 드라이버 버전 이상이면 import만 씁니다. vbmir도 같은 기준으로 끕니다.
  - 표는 코드 상수로 둡니다. 측정 결과와 함께 갱신합니다.

## 6. 예상 효과 (전제 미검증, 현재 숫자에서 추론)

| 게임 | 지금(copy, async) | 혼합 예상 |
|---|---|---|
| 바요3 | 53-60 fps. depth pass SSBO 범위가 copy | 그 범위는 copy로 바뀌어 같은 효과. 나머지 범위는 import로 가서 fault 15-46k/s 중 일부가 사라짐(비율 미측정) |
| HWAoI | 이동 37-49, 정지 53-57 fps | fault가 "GPU 쓰기 없음" 범위에 많으면 이동 중 하락이 크게 줄어듦. vertex fetch는 vbmir가 이미 처리. 비율 미측정 |
| StarFox | 47-58 fps, GPU 12-14 ms | 느린 쓰기 범위가 적으면 거의 import로. 효과는 작을 것으로 예상 |

## 7. 위험

- 1.1절 전제가 틀리면(fault가 느린 쓰기 범위에 몰려 있으면) 효과가 없습니다. 먼저 카운터로 확인합니다.
- 전환 시 drain 한 번 + 전체 동기화(범위 크기만큼, 최대 256 MB)로 프레임이 한 번 끊깁니다. 로딩 중에 몰릴 가능성이 큽니다.
- vk_buf_at 등 범위를 가리키는 모든 곳이 전환 뒤 사본을 써야 합니다. 이미 만든 descriptor set이나 디스크 캐시된 바인딩에 import 버퍼가 남아 있으면 안 됩니다. 전환 때 descriptor 캐시를 무효화해야 합니다.
- 2절 측정 전에는 attachment 쓰기를 어느 쪽으로 볼지 정할 수 없습니다.

## 7.0 구조 결정 (NPARK, 11-20 23:49)

**혼합 = 버텍스 풀링 + import + atomic 예외 경로.** 드라이버가 고쳐지지 않을 수 있다는 전제로, 임시 우회가 아닌 장기 기본 경로로 만듭니다.
- import: 기본입니다. CPU 쓰기 추적 비용이 없습니다.
- 버텍스 풀링(design\VERTEX_PULLING_DESIGN.md, gfx_vk_eden.inc vtxpull): 고정 기능 vertex fetch를 셰이더 로드로 대체합니다. import에서 셰이더 읽기는 정상 속도입니다. 풀링할 수 없는 draw는 vbmir로 갑니다(지금 상태).
- atomic 예외 경로(hybrid1, 7.1): atomic이 닿는 단위만 device-local 사본으로 둡니다.
- 전환된(사본) 청크에서는 고정 기능 fetch가 device-local이라 빠릅니다. 그래서 풀링과 vbmir가 둘 다 필요 없습니다. 두 경로 모두 zero-copy 청크에만 적용됩니다.

## 7.1 hybrid1 구현 (11-21, `NXN1X_POOL_MODE=hybrid`, 범위: atomic 예외 경로만)

NPARK이 정한 구조는 "import 기본 + 정점(지금은 vbmir, 나중에 vertex pulling) + atomic 예외 경로"이고, hybrid1은 그중 atomic 예외 경로입니다.

- **전환 기준**: is_written=1이면서 셰이더가 atomic을 실행하는 SSBO 바인딩입니다.
  - Eden `StorageBufferDescriptor::is_atomic`을 global_memory_to_storage_buffer_pass가 GlobalAtomic*에서 표시합니다.
  - nvn_eden_bind.is_atomic을 거쳐 gfx_vk_eden.inc eden_ssbo → vk_hy_ssbo로 전달됩니다.
  - store만 하는 바인딩, transfer 쓰기, raw store로는 전환하지 않습니다.
- **단위**: 청크가 걸친 NXN1X backing 256 MB 블록, 그리고 청크의 guest 범위입니다.
  - 전환된 청크는 두 가지를 모두 단위 목록에 더합니다.
  - hy_apply가 공유하는 모든 청크를 닫힘이 될 때까지 함께 전환합니다. 그래서 alias와 같은 VA의 청크가 같이 바뀝니다.
- **시점**: 요청은 기록 중에 남기고, 다음 제출 시작(mtl_sub_begin)에서 copied pools 스캔 전에 적용합니다.
  - import 버퍼는 vk_defer_free_buffer로 GPU가 지난 뒤 해제합니다.
  - 새 사본은 synced=0입니다. 그래서 스캔이 앞선 모든 제출을 기다린 뒤 게임 페이지에서 전체를 채웁니다.
- **재사용**: 사본은 pc_make이고, 그 뒤는 copy 모드 경로(stage2c, stage2b, fault-ahead) 그대로입니다.
  - serial을 올려 tbuf view, vbmir 미러, pce 엔트리를 무효화합니다.
  - g_pool_gen도 올려 descriptor memo와 pool table을 무효화합니다.
- **import 거절**: hybrid에서는 거절된 청크가 legacy 사본이 아니라 device-local 사본이 되고, 그 단위도 전환됩니다.
- **Eden 모드 raw atomic(cbuf 0xFF)의 제한**: 쓰기 로그가 없어 어디에 썼는지 모릅니다. 그래서 전환하지 않고(H11 규칙) `hybrid1: shader … runs an atomic through raw global memory` 줄과 카운터만 남깁니다.
- **쓰기 로그 경로(vk_gml_note2)**: DM_VK_EDEN=0 경로에만 있습니다. 그 경로의 번들에는 atomic 정보가 없어서, 실제로는 셀프테스트(H9-H11)만 이 경로를 씁니다.
- **로그**:
  - `hybrid1: chunk … is a device-local copy`: 전환마다, 원인과 셰이더 해시 포함
  - `hybrid1 conversions`: 120번 제출마다
  - `hybrid1 binds`: 모든 모드에서. 바인딩 종류별 수와 바이트, 그리고 각 종류가 만든 pre 복사, g_gw, 미러 무효화 몫입니다. compute 읽기 전용 몫을 보는 진단이고 동작은 바꾸지 않습니다.
- **이번에 하지 않은 것**:
  - 드라이버 표(5절, H8)
  - 셰이더 store로 인한 전환
  - is_written=0 바인딩을 pre/post에서 빼는 것. 숫자를 본 뒤 NPARK 승인이 필요합니다.

## 7.2 hybrid2 (11-21 00:35, NPARK 00:30 승인: 전환 단위 + 읽기 전용 제외)

- 원인(OoT 00:25, ab_hybrid_002501):
  - atomic이 직접 닿은 청크는 24개였는데, 256 MB 블록 닫힘 규칙 때문에 452개 청크(493 MB)가 사본이 됐습니다.
  - 그 사본에서 compute SSBO를 모두 쓰기로 쳐서 pre/post가 120번 제출당 각각 4.9 GB였습니다. 그중 읽기 전용이 65%입니다.
  - 그 결과 replay pool-sync가 10초 중 7.2초, copyback이 5.4초였습니다.
- (a) 전환 단위: 256 MB 블록 대신 청크의 정확한 backing 범위와 guest 범위로 바꿨습니다. 페이지를 실제로 공유하는 청크만 함께 바뀝니다(H13). device-local 메모리 할당 자체는 지금처럼 256 MB 단위(pc_range)로 공유합니다.
- (b) SSBO 쓰기 등록(vk_ssbo_bound):
  - is_written=1인 바인딩만 쓰기로 칩니다.
  - dispatch의 경우 write_extent를 그 dispatch의 workgroup과 grid로 푼 범위까지만 칩니다(nvn_eden_ssbo_reach). indirect dispatch는 grid를 모르므로 범위를 모를 수 있고, 그때는 바인딩 전체로 칩니다.
  - copy, hybrid, import 모든 모드에 적용됩니다.
  - 예외(지금처럼 유지): indirect SSBO(포인터를 메모리에서 읽는 것)와 raw pool table.
  - CPU → GPU 업로드는 바꾸지 않았습니다(R6).
  - `NXN1X_SSBO_WRITES=all`이면 예전처럼 모두 쓰기로 칩니다(A/B용).
- 로그: `hybrid2 storage buffer writes`(120번 제출마다)에 쓰기로 치지 않은 읽기 전용 바인딩 수와 MB, 범위로 잘린 MB가 나옵니다.
- 테스트:
  - 154 H13
  - 160-165 R1-R6: 읽기 전용은 pre와 목록에 없음, reach 256 B, reach 모름이면 전체, indirect는 그대로, reach 안의 store가 게임 페이지에 반영, 읽기 전용 바인딩 아래의 CPU 쓰기 업로드

## 8. 테스트 (self-test, 코드 작성 시 먼저)

hybrid1 셀프테스트 번호는 140 H1(import 모드), 141 H0, 142 H2, 143 H3, 144 H4, 145 H5, 146 H6, 149 H9, 150 H10, 151 H11, 152 H12(Eden raw atomic 카운터), 153(A와 C 전환 뒤 기존 copy 케이스)입니다. H7(case 8b)은 기존 case 8이 두 모드에서 다룹니다. H8은 미구현입니다.


- H1: `NXN1X_POOL_MODE=import`이면 모든 범위가 import로 남고 사본이 하나도 만들어지지 않는다.
- H2: import 범위에 첫 SSBO store 바인딩 → 다음 제출에서 copy로 전환. 전환 직전 import로 쓴 GPU 값과 CPU 값이 모두 사본에 들어간다.
- H3: 전환 중 진행 중 제출(hold 훅): 앞 제출의 import GPU 쓰기가 전환 뒤 사본에 들어 있다.
- H4: alias 두 VA 중 하나로 느린 쓰기 → 두 VA 모두 copy로 바뀌고 서로의 쓰기를 본다.
- H5: transfer dst만 있는 범위는 import에 남고, 그 GPU 쓰기가 게임 페이지에 바로 보인다.
- H6: 전환된 범위는 그 실행 동안 copy로 남는다(단방향).
- H7: counter 보고와 뒤 GPU 쓰기의 순서(case 8b)가 두 모드 모두에서 맞다.
- H8: 드라이버 표에 등록된 버전이면 전환이 일어나지 않는다(표를 가짜 값으로 바꿔 확인).
- H9: raw global memory 바인딩(일반 store, write log 정상)만으로는 어느 범위도 copy가 되지 않고, raw 저장이 게임 페이지에 바로 보인다.
- H10: 전역 atomic을 쓰는 raw 셰이더의 write log에 나온 페이지의 범위만 copy로 바뀌고, 다른 범위는 import에 남는다.
- H11: write log가 넘친 raw 저장 뒤에도 범위가 바뀌지 않고 값이 정확하다(import).
- 기존 T1-T17, V1-V7, FA1-FA3는 copy 범위에서 그대로 통과해야 한다.
