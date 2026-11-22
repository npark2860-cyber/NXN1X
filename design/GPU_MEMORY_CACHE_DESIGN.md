# GPU 메모리 캐시 설계 (N1X 근본 수정)

작성: 2026-11-20. 상태: **설계안. 아스트라 검수와 NPARK 승인 전까지 구현하지 않습니다.**
경로 기준: 파일 경로는 `nxn1x\`에 대한 상대 경로입니다. `host\dm\...`은 NVN 호스트, `source\src\...`는 Eden 소스입니다.

## 0. 한 줄 요약

N1X(NVIDIA 621.24)에서는 GPU가 **import한 게스트 메모리**(VK_EXT_external_memory_host)를 직접 다룰 때 여러 경로가 느립니다.
이를 고치려면 NVIDIA에서 **모든 풀을 드라이버가 할당한 GPU 사본(copied pool)으로 운용**해야 합니다.

copied pool 경로는 호스트에 이미 있습니다. 다만 지금은 import가 거부된 풀에만 쓰는 비상 경로이고, 느린 부분이 남아 있습니다.
이 설계는 그 경로를 다음 세 가지로 주 경로로 끌어올립니다.

1. **CPU 쓰기 업로드**를 vbmir/vbvol 방식(4 KB 페이지, volatile 페이지)으로 증분화합니다.
2. **GPU 쓰기 되돌림**(copy-back)은 기존 `gw[]` 범위 방식을 유지하고, 필요하면 비동기로 바꿉니다.
3. **드라이버별 정책**을 둡니다. NVIDIA는 사본을, Adreno와 Mac은 지금처럼 import를 씁니다.

정점 미러(vbmir)와 버텍스 풀링의 적용 조건은 5절에 있습니다.

## 1. 측정된 사실 (기존 벤치, 2026-11-20)

| 경로 | import / 정상 | 출처 |
|---|---|---|
| 고정 기능 정점·인덱스 fetch | 20~250배 느림 (grass draw 202.8 vs 12.1 µs, 3633 vs 29.7 µs) | membench4, membench2 |
| 같은 정점 버퍼 반복 읽기 (캐시 안 됨) | 35.9배 | membench5 #5 |
| atomicAdd (1 MB에 4 M회) | 25.0배 | membench5 #2 |
| SSBO store (16 MB) | 1.8배 | membench5 #2 |
| texel buffer store, 복사, 버퍼→이미지, indirect args, xfb, query copy | 1.0~1.1배 (정상) | membench5 |
| 셰이더 load (SSBO, BDA, UBO, texel) | 정상 | membench3, membench4 |
| **드라이버가 할당한 호스트 메모리 (type 2)** | **정상 (8.7 µs, device-local 9.8 µs)** | membench2 |
| 내보낸(exportable) 드라이버 메모리, OPAQUE_WIN32 섹션 import | 느림 (438~1133 µs) | membench2 |
| 같은 주소에 픽셀마다 몰아 쓰기 | 측정하지 않음 (NPARK 지시로 실험 취소) | — |

결론:
- 느린 것은 "호스트 메모리"가 아니라 **"사용자 포인터를 import한 메모리"**입니다.
- N1X 추가 벤치에서 드라이버 host-visible type 2/3은 반복 정점 fetch에서 device-local과 동급이었습니다. 연속 SSBO store는 각각 2.8/2.7배, 분산 atomicAdd는 2.0/1.7배의 GPU 시간이 들었습니다. [측정 조건과 원자료](GPU_MEMORY_CACHE_REVIEW.md)
- 게스트 주소에 드라이버 메모리를 그대로 놓는 경로는 없습니다(`VK_EXT_map_memory_placed` 미지원, 내보낸 메모리도 느림). 그래서 사본이 필요합니다.

게임에서 확인된 영향:
- HWAoI: 정점 fetch 경로 하나 때문에 16 fps였습니다. vbmir로 60 fps, GPU 65%가 됐습니다.
- Bayonetta 3 (2X): 깊이 패스 하나가 GPU 시간 40~57 ms를 차지합니다. 같은 크기에 draw가 600~900개인 색상 패스는 0.7~2 ms입니다.
  - 그 패스의 fragment 셰이더들은 import 메모리에 픽셀마다 `ssbo[object] = 65535`를 씁니다.
  - atomdiag: `4f5eed91`과 `481193a6`이 프레임당 약 200회 바인딩되고, 100% import입니다.
  - 덤프 디스어셈블로 이 패턴을 확인했습니다.

## 2. 왜 Eden buffer cache를 우회했나 (코드 근거)

**사실 1. NVN 경로는 Eden의 Maxwell 엔진 에뮬레이션을 거치지 않습니다.**
- `source\src\core\nvn\nvn_layer.h:2-3`: 게임의 NVN 호출을 게임 자신의 NVN 드라이버와 Maxwell GPU 에뮬레이션 대신 호스트 레이어로 보낸다고 적혀 있습니다.
- `RendererVulkan`은 NVN 레이어에 장치, 스케줄러, 파이프라인 캐시만 넘깁니다. `NvnBridge::Attach(..., device, scheduler, rasterizer.GetPipelineCache())` (`source\src\video_core\renderer_vulkan\renderer_vulkan.cpp:300-302`, 인터페이스는 `nvn_bridge_device.h:30-36`).
- `nvn_shader_bridge.cpp:21-42`는 셰이더 리컴파일러, 파이프라인 캐시, 장치 헤더만 include합니다. buffer cache는 참조하지 않습니다.
- buffer cache는 `RasterizerVulkan`이 소유합니다(`vk_rasterizer.h:225-226`). 바인딩 정보는 `maxwell3d->regs`에서 읽습니다(`vk_rasterizer.cpp:646-694`). NVN 타이틀에서는 이 경로가 쉬고 있습니다.

**사실 2. 호스트는 Mac/Android 통합 메모리에서 시작했고, zero-copy를 기본으로 설계했습니다.**
- `host\dm\windows\gfx_vk_core.c:4-6`:
  - NVN에서는 GPU 주소가 곧 CPU 주소입니다. `VK_EXT_external_memory_host`로 풀 페이지를 import해서 "게임 포인터가 곧 GPU 메모리"가 되게 합니다.
  - 거부된 풀만 GPU 사본을 쓰고, 이를 "정확하지만 느린" 경로로 적어 두었습니다.
- Mac: `newBufferWithBytesNoCopy` (`host\dm\gfx\nvn\nvn_mtl.m:576`); `host\docs\reference\nvn-objects.md:121-142`에 "zero-copy 버퍼는 메모리를 더 쓰지 않는다"고 적혀 있습니다.
- Android: AHardwareBuffer (`gfx_vk_core.c:5`, `:878-879`).
- copied pool의 과거 비용:
  - 제출마다 전체를 복사했을 때 "초당 몇 프레임"이었습니다(`gfx_vk_core.c:1669-1671`, `:1814-1816`).
  - Turnip에서 제출마다 풀 전체를 되돌렸을 때 DK가 프레임당 7~8초였습니다(`:1887-1891`).
  - 이 경험 때문에 import가 기본이 되었습니다.

**사실 3. Eden BufferCache 클래스는 그대로 재사용할 수 없습니다.** 메커니즘만 이식합니다.
- GPU VA와 DAddr 키를 씁니다. `MaxwellDeviceMemoryManager`, `gpu_memory->GpuToCpuAddress`에 묶여 있습니다(`buffer_cache_base.h:212`, `:475`; `buffer_cache.h:433` 외). NVN은 호스트 가상 주소를 씁니다.
- `Update*`와 `Bind*`는 `maxwell3d->regs`와 `kepler_compute`가 있어야 동작합니다(`buffer_cache.h:1138-1212` 외).
- CPU 쓰기 추적이 fastmem 보호(`RasterizerMarkRegionCached`, `memory.cpp:495-512`)를 씁니다. NVN 풀은 `nvn_dirty`가 이미 보호하고 있고, 그 fault가 먼저 처리됩니다(`memory.cpp:1271`). 보호 주인이 둘이 되어 충돌합니다.
- Eden 스케줄러, 마스터 세마포어, FenceManager(Host1x syncpoint)에 의존합니다(`vk_buffer_cache.cpp:417-431`; `fence_manager.h:82-137`). NVN 호스트는 자체 제출 링, timeline semaphore, 완료 스레드를 씁니다(`gfx_vk_core.c:1820-1864`, `:3405-3501`).

**이식할 가치가 있는 Eden 메커니즘:**
- GPU-modified 범위 집합과 committed/uncommitted 구분(`buffer_cache.h:1824-1843`)
- 펜스에서 비동기 다운로드(`CommitAsyncFlushesHigh` / `PopAsyncBuffers`, `buffer_cache.h:845-1016`)
- 커밋 뒤의 더 새로운 GPU 쓰기는 유지하는 규칙(`:950-1011`)
- CPU 쓰기가 들어오면 대기 중인 다운로드 취소(`ClearDownload`, `:410-426`)

**코디네이터 설명 검증 결과:** "Mac/Adreno 통합 메모리에서 zero-copy가 가장 빨랐고, NVN HLE는 Maxwell 엔진(buffer cache 위치)을 거치지 않으며, 공유하는 것은 셰이더 리컴파일러뿐"이라는 설명은 맞습니다.
- 정정 1: 공유하는 것은 셰이더 리컴파일러뿐 아니라 **파이프라인 캐시, Vulkan 장치, 스케줄러, 큐 락**까지입니다.
- 정정 2: Android 쪽 근거는 Adreno 일반이 아니라 AHardwareBuffer입니다.
- 정정 3: Turnip에는 `external_memory_host`가 없어서 copied pool을 썼습니다.

## 3. 이미 있는 부품 (재사용 대상)

| 기능 | 현재 코드 | 이 설계에서 |
|---|---|---|
| 풀 chunk와 GPU 버퍼 | `mtl_pool_add`, `pool_make_0`, `vk_pool_of`, `vk_buf_at` (`gfx_vk_core.c:878-966`, `:1349-1401`, `:1532-1568`) | 그대로 씁니다. NVIDIA는 zero-copy 대신 사본을 고릅니다. |
| copied pool 사본 할당 | `pool_shadow_required`, `pool_shadow_allocate` (`:1280-1344`) | 메모리 타입을 정책으로 정합니다(5절). |
| CPU 쓰기 감지 | `nvn_dirty` (`host\dm\gfx\nvn\nvn_dirty.c:91-175`, `:277-325`), 4 KB, 세대 스탬프 | 그대로 씁니다. |
| 증분 업로드 | `vk_pools_sync`, `pool_sync_one`(64 KB 블록), `pool_sync_copy` (`:1684-1796`) | 4 KB 페이지와 volatile 처리로 바꿉니다(vbvol 교훈). |
| GPU 쓰기 추적 | `vk_gpu_wrote`, `sub_gpu_wrote`의 `gw[64]`, `vk_sub_wrote`, `vk_gml_note` (`:3201-3249`) | 크기 미상·로그 포화와 부분 제출을 처리합니다. |
| 되돌림 (GPU → 게스트) | `pools_copy_back`, `snap_merge` (`:902-919`, `:1920-1939`) | 완료 스레드를 활용하고, 입력을 세대별 readback slice로 바꿉니다. |
| 펜스 대기 | `g_done` 공개(`:3488`)와 `mtl_wait`를 사용하는 ticket 대기 | 실제 전송과 병합을 확인한 뒤 공개합니다. 이벤트 관찰 순서는 별도로 검증합니다. |
| 전송 동기화 | `vk_pools_sync` (`:2978-2995`), `pool_sync_host_barrier` (`:1865-1886`) | Switch 1·2 모두에서 실제 소비자와 readback까지 의존성을 연결합니다. |
| 정점 미러 | vbmir/vbvol (`gfx_vk_exec.c:383-498`) | 사본의 정점·인덱스 바인딩은 5절을 따릅니다. |

기존 완료 스레드와 풀 조회를 재사용합니다. device-local 사본의 전송, 쓰기 세대, alias 처리는 새로 검증할 부분입니다.

## 4. 방식

### 4.1 메모리 위치 정책 (드라이버별)

- **NVIDIA (`vk_is_n1x()` 또는 vendor 0x10de)**
  - GPU가 사용하는 풀의 렌더링·컴퓨트 바인딩은 device-local 사본을 사용합니다.
  - 게스트 원본을 전송에 사용할 조건은 4.2절을 따릅니다. NCE가 접근하는 게스트 주소는 유지합니다.
- **Adreno/Turnip, Mac, 그 외**: 지금처럼 import(zero-copy)를 씁니다. import가 거부되면 사본으로 가는 것도 지금과 같습니다.
- **전환 스위치**: `NXN1X_POOL_MODE` = `import` / `copy` / `auto`(기본값, 드라이버로 판정).
  - 풀이 만들어질 때 정해지는 값이라 present 단위로 바꿀 수는 없습니다. 실행 단위 A/B입니다.
  - 근본 수정의 검증용 스위치이고, 결과가 확인되면 auto 하나만 남깁니다.

### 4.2 사본의 메모리 타입

두 후보 중 (A)를 선택했습니다. 아래 성능은 N1X에서 측정한 연산에 한정됩니다.

| | (A) device-local | (B) 드라이버 host-visible (이 기기에서 type 2/3, HOST_CACHED 우선 시 type 3) |
|---|---|---|
| CPU → 사본 업로드 | GPU 복사(staging 또는 import 원본 → 사본, vbmir 방식) | CPU memcpy (지금의 `pool_sync_copy`) |
| GPU → 게스트 되돌림 | 사본 → staging GPU 복사, 완료 후 CPU memcpy | 완료 후 CPU memcpy (지금의 `pools_copy_back`) |
| GPU 접근 속도 | 벤치 비교 기준 | 반복 정점 fetch 약 1.0배, 분산 atomicAdd 1.7~2.0배의 소요 시간 |
| 용량 | heap0 86.5 GB | heap1 24 GB 안 |
| 구현량 | 큼 (staging, 복사 기록) | 작음 (기존 copied pool 그대로) |

- **결정 (NPARK, 2026-11-20 15:02): (A) device-local로 갑니다.** 메모리 용량은 제약이 아닙니다("이 컴에서 메모리가 부족하면 그건 이 세상 게임이 아닌 거야").
  - 렌더링 사본은 device-local에 둡니다. (B)의 추가 측정은 참고 자료입니다.
- 업로드: 게스트 페이지 → 사본 GPU 복사입니다. vbmir가 검증한 방식이며, 업로드 명령 버퍼와 barrier를 씁니다.
  - 전송이 끝날 때까지 원본 범위의 내용과 수명을 보장할 수 있으면 import 버퍼를 업로드 원본으로 사용할 수 있습니다. 나머지는 해당 세대의 데이터를 host-visible staging에 보관합니다.
- 되돌림의 검수 권고안은 device-local → 제출별 readback staging → 완료 스레드에서 게스트에 병합하는 순서입니다. 게스트 페이지에 대한 직접 GPU 복사는 해당 바이트에 대한 CPU 쓰기와 alias 접근을 배제할 수 있을 때만 별도로 검증합니다. 상세 순서는 [R9 검수](GPU_MEMORY_CACHE_REVIEW.md)에 있습니다.
- 구현량은 (B)보다 크지만 vbmir 코드(`gfx_vk_exec.c:383-498`)가 그 절반입니다.

### 4.3 CPU 쓰기 → 사본 (업로드)

- 단위: **4 KB 페이지**. 지금의 64 KB 블록은 vbvol에서 fault 폭주의 원인이었습니다.
- 감지: `nvn_dirty` 세대 스탬프를 씁니다. 업로드 기준값을 채취하기 전에 보호와 세대를 설정해 그 사이에 발생한 CPU 쓰기도 다음 세대에 남깁니다.
- **volatile 페이지**: 3회 연속 dirty일 때 보호를 푸는 정책을 비교합니다. CPU와 GPU가 함께 쓰는 페이지에서는 최신 GPU 바이트를 보존할 수 있는 범위만 다시 올립니다.
- 시점: 현재 `vk_pools_sync`는 `mtl_sub_begin`에서 실행됩니다. 이후에 드러난 첫 바인딩과 중간 제출 전에 필요한 업로드가 실제 소비보다 앞서도록 기록합니다.
- 범위: **GPU가 이번 제출에서 실제로 쓰는 범위만** 올립니다. 바인딩된 범위(vbmir의 지연 방식)이고, 풀 전체를 스캔하지 않습니다.
  - 기존 `pool_sync_one`은 dirty 페이지를 모두 스캔합니다. 지연 방식으로 바꾸려면 바인딩 지점(`vk_buf_at` 호출부) 모두에 "이 범위를 이번 제출에서 쓴다"는 표시가 필요합니다.
  - 바인딩 지점: vertex/index, SSBO, UBO, texel buffer, indirect, xfb, copy/fill, raw global memory BDA 테이블.
  - raw global memory(BDA 테이블, `gfx_vk_core.c:1612-1639`)는 셰이더가 임의 주소를 읽습니다. 그래서 그 풀은 바인딩 범위를 알 수 없고, **dirty 페이지 전체 업로드**로 대체합니다.

### 4.4 GPU 쓰기 → 게스트 (되돌림)

- 추적: 기존 `vk_gpu_wrote` / `sub_gpu_wrote` / `vk_sub_wrote` / `vk_gml_note`를 출발점으로 씁니다. 크기 미상 SSBO와 raw global 로그 포화 시 누락되는 범위를 처리해야 합니다.
- `gw[64]` 한도를 키웁니다(구간 트리 또는 페이지 비트맵). 지금은 넘치면 가장 가까운 항목에 합쳐서 범위가 커집니다.
- 되돌림: 완료 스레드는 현재 기록된 범위를 처리한 뒤 `g_done`을 공개합니다(`gfx_vk_core.c:3488`). (A)에서 성공한 대기가 올바른 값을 보려면 실제 제출된 쓰기가 readback에 포함되고 해당 게스트 세대에 병합되어야 합니다. 이벤트 폴링과 mapped-range invalidate는 별도의 관찰 경로입니다.
- CPU와 경합: 기존 `snap_merge`는 32비트 단위입니다(`:881-919`). 동일 단어의 다른 바이트를 바꾼 경우와 snapshot 누락을 처리하도록 병합 규칙을 보강합니다.
- 안전망 (게임이 기다리지 않고 읽는 경우):
  - 일반 펜스를 기다리지 않아도 이벤트 값을 관찰한 뒤 payload를 읽는 경우가 있습니다. 게스트가 사용하는 공개 지점을 따릅니다.
  - 사본 방식에서는 되돌림이 늦어질 수 있습니다(다음 완료까지).
  - **위험 항목 R2**로 두고, 필요하면 GPU dirty 페이지를 읽기 보호해 fault로 강제 되돌리는 방식을 2단계에 둡니다. Eden의 `OnCPURead`(`gpu.cpp:317-329`)에 해당합니다.

### 4.5 비용 추정 (검증 항목)

- 업로드: 프레임당 CPU가 바꾸는 바인딩 범위입니다. HWAoI의 vbmir 실측은 프레임당 0.01~0.1 MB에 volatile 0.4 MB였습니다. 상수와 동적 버퍼가 더해지므로 수 MB/프레임으로 예상합니다.
- 되돌림: GPU가 쓴 범위입니다. Bayo3 가시성 플래그는 수 KB이고 SSBO 바인딩은 16~30 MB입니다. **바인딩 범위가 아니라 실제로 쓴 페이지만** 되돌려야 합니다.
  - 지금 `gw[]`는 바인딩 범위로 기록합니다. 그래서 30 MB/프레임 memcpy와 페이지 비교가 생길 수 있습니다. 페이지 비교는 바뀐 페이지만 쓰지만 읽기 비용은 남습니다.
  - **위험 항목 R1**: 필요하면 GPU 쓰기 범위를 줄입니다. 쓰기 보호 대신 page compare, 또는 2단계에서 GPU 쪽 dirty 비트맵을 둡니다.
- 메모리: GPU가 쓰는 풀만큼 device-local을 더 씁니다. 제약이 아니므로 한도는 두지 않습니다(8절). 참고값은 바요3 1,684 MB(실측), HWAoI 약 5 GB(기록)입니다.
- 지연: (A)의 업로드와 readback은 GPU 전송입니다. staging을 쓰면 업로드용 CPU 복사와 완료 스레드의 병합 시간도 측정합니다. 게임이 결과를 기다리면 그 시간이 원본 프레임 지연에 포함됩니다.

## 4.6 검수 P1 대응 (2026-11-20 아스트라 검수 반영)

검수 판정은 "(A) 비교 구현 진행, 기본값 전환 보류"입니다([검수](GPU_MEMORY_CACHE_REVIEW.md)).

**지적별 분류**

| 지적 | 분류 | 비고 |
|---|---|---|
| P1 직접 되돌림이 CPU 쓰기를 덮음 | 수용 | 아래 P1-1 |
| P1 크기 미상 SSBO, raw 로그 포화 | 수용 | P1-2 |
| P1 `p->shadow` 포인터 전제 | 수용 | P1-3 |
| P1 bridge, alias, 겹침, 재등록 | 수용 | P1-4. backing 단위 사본으로 해결합니다. |
| P1 이벤트, 카운터, query, invalidate 공개 시점과 부분 제출 | 수용 | P1-5 |
| R9 전송·공개 4단계 순서 | 수용 | P1-1과 P1-5에 반영 |
| 되돌림 양: `gw` 확대만으로는 부족, 단계별 바이트 측정 | 수용 | 6절 로그 항목에 추가 |
| 업로드 시점: 정확성 기준 경로부터 | 수용 | 1단계 기준 경로는 제출 시작의 dirty 전체 업로드 |
| `snap_merge` 32비트 한계, volatile 페이지가 GPU 바이트를 덮음 | 수용 | 3-way 바이트 병합. volatile 정책은 2단계로 미룹니다. |
| fault 비용: `pool_tracked` 4 MiB 생략, 측정 항목 | 수용 | 7절 기록 항목 |
| Eden 메커니즘 이식, 활성 경로 기록 | 수용 | 7절 |
| 합성 테스트 먼저, 게임 A/B는 동일 조건 | 수용 | 6절 1a단계 |
| (B) type 2/3 추가 측정 | 참고 | (A) 결정으로 기록만 남깁니다. |

반박할 지적은 없습니다. P1 다섯 건에 대한 설계 결정은 아래와 같습니다.

**P1-1 직접 되돌림이 CPU 쓰기를 덮는 문제 (R9)**
- GPU → 게스트 직접 복사는 하지 않습니다. 사본 → **제출별 readback slice**(host-visible, HOST_CACHED, 제출 링과 같은 수명) GPU 복사만 합니다.
- **기준값(baseline)**: 제출을 기록하는 시점에 GPU 쓰기 후보 범위의 게스트 바이트를 CPU가 slice 옆 baseline 영역에 복사합니다. 제출 시작에 게스트와 사본이 같다는 것은 업로드와 앞선 병합이 보장합니다.
- **병합 (완료 스레드, 바이트 단위 3-way)**: 결과(R), 기준(B), 현재 게스트(G)로 판정합니다.
  - R≠B이고 G=B: GPU 값을 씁니다.
  - G≠B: CPU가 그 사이에 쓴 것이므로 CPU 값을 유지합니다. Switch에서도 동기화 없는 경합이고, 나중 행위인 CPU를 택합니다.
  - 병합 중 CPU 재쓰기는 기존 `dm_guest_write_prepare`와 페이지 단위 보호로 직렬화합니다.
- `snap_merge`(32비트, 슬롯 충돌 시 건너뜀)를 이 규칙으로 대체합니다. 같은 단어의 다른 바이트 경합도 보존됩니다.
- 비일관 매핑은 timeline 대기 **뒤에** invalidate합니다.

**P1-2 크기 미상 SSBO와 raw 로그 포화 (R1/R2)**
- 크기를 확정할 수 없는 쓰기 가능 바인딩(`RK_SBUF_UBO`, `g_sbufu_w`, `gfx_vk_exec.c:2720`)은 **그 주소를 담은 backing 범위 전체**를 후보로 기록합니다.
- raw global 쓰기 로그(`gml_copy_back`, `gfx_vk_core.c:1904`)가 포화되면, 그 제출에서 BDA 테이블이 노출한 **모든 사본 범위**를 후보로 바꿉니다. 손실 없는 대체입니다.
- 비용은 커질 수 있지만 정확성이 우선입니다. 발생 횟수와 MB를 로그에 남깁니다.
- 로그를 실행 후에야 읽을 수 있으므로 후보 범위 전체를 readback slice에 복사해 둡니다. 판독은 완료 뒤 병합 단계에서 합니다.

**P1-3 `p->shadow` 포인터 전제 (R9)**
- 사본 상태를 포인터 하나로 표현하지 않습니다. 풀(또는 backing 범위)마다 세 가지를 분리합니다.
  - `dev_buf`: 렌더와 컴퓨트 바인딩용, device-local, 매핑 없음
  - 업로드 원본: import 버퍼 또는 staging slice
  - readback slice: 제출별 링
- "사본인가"는 `copy_mode` 플래그로 판정합니다. `p->shadow`를 직접 읽는 사이트(`memcmp`, `memcpy`, `snap_merge`, `gfx_vk_core.c:1290`, `:1351`, `:1920` 외)를 전수 조사해서, 사본 모드에서는 readback slice와 baseline만 읽게 바꿉니다.
- slice는 그 제출의 병합이 끝난 뒤에 재사용합니다.

**P1-4 alias, bridge, 겹치는 풀, 재등록 (R4)**
- 사본은 풀 chunk가 아니라 **NXN1X backing 메모리의 정렬 범위 단위**로 만듭니다.
  - 기존 s48/s23이 이미 chunk를 backing 범위에 대응시킵니다(`gfx_vk_core.c:464-577`).
  - 같은 backing을 가리키는 모든 view와 풀은 **같은 device-local 버퍼의 같은 offset**을 씁니다.
  - 그래서 alias 사이의 일관성(한 view에서 쓰고 같은 제출의 다른 view에서 읽기)은 GPU 메모리 하나로 자동 보장됩니다.
- view 경계에 걸친 bridge도 backing이 연속이면 연속 범위입니다. backing이 불연속이면 지금처럼 bridge 버퍼를 쓰고, 그 범위는 copy 모드에서 제외합니다(로그로 계수).
- 권위 있는 사본은 backing 범위 하나뿐이므로 "오래된 사본이 새 값을 덮는" 경우가 생기지 않습니다.
- 해제 후 같은 VA로 재등록하는 경우: backing 할당 세대를 키에 넣고, 세대가 바뀌면 사본을 무효화합니다(첫 사용 시 전체 업로드).
- `pool_make_0`의 bridge 분기는 copy 모드 판정 **뒤로** 옮깁니다.

**P1-5 이벤트, 카운터, query, invalidate의 공개 시점 (R2/R3)**
- 공개 지점과 처리 순서:

  | 공개 지점 | 처리 |
  |---|---|
  | 펜스, `nvnSyncWait`, `nvnQueueFinish` | 제출 끝 readback → 병합 → `g_done` (기존 순서) |
  | NVN 이벤트, 카운터 (완료 스레드가 게스트에 씀, `gfx_vk_core.c:3452-3479`) | **병합 뒤에** 씁니다(기존 순서 확인). CPU가 이벤트를 본 직후 payload를 읽어도 최신입니다. |
  | GPU가 같은 제출 안에서 소비하는 이벤트·데이터 | 사본끼리 GPU 안에서 처리합니다(barrier). 공개는 필요 없습니다. |
  | CPU가 제출 도중에 관찰하는 이벤트 (`nvnEventGetValue`가 게스트를 직접 읽음, `hle_nvn.c:1100`) | 그 이벤트 신호를 **제출 분할 지점**으로 삼습니다. 기존 part 분할을 쓰고, 앞부분의 readback과 병합을 끝낸 뒤 이벤트를 씁니다. |
  | `nvnBufferInvalidateMappedRange` (지금은 빈 함수, `hle_nvn.c:2878`, `:2911`) | 그 범위에 대기 중인 readback이 있으면 병합까지 기다립니다. |
  | query 결과, s56 렌더 타깃 착지 | 게스트 명령 순서대로 병합합니다. 같은 backing이면 뒤 명령이 이깁니다. |

- **부분 제출 실패**: part별 접수(`nvnc_accept`)를 기준으로, 접수된 part의 후보 범위와 readback만 병합합니다. signal-only 복구 fence는 전송 완료로 계산하지 않습니다.
- **readback 의존성**: Switch 1과 2 모두에서 기록합니다. `pool_sync_mode()`와 `pool_sync_host_barrier`가 Switch 1에서 빠지는 현재 동작(`gfx_vk_core.c:1865-1886`)은 copy 모드에서 적용하지 않습니다.
- **업로드 의존성**: 소비자 전체(vertex/index, shader storage·uniform·texel, indirect, transfer)를 dst stage/access에 넣습니다. `vk_vbm_up_close`는 vertex/index만 다룹니다.

**검수 권고에 따른 업로드 기준 경로**
- 1단계 기준 경로는 **제출 시작 시점의 CPU dirty 페이지 전체 업로드**(바인딩 지연 없음)입니다. 이것이 정확성 기준입니다.
- 바인딩별 지연 업로드와 volatile 정책은 그다음 최적화로, 기준 경로와 바이트 단위로 대조한 뒤에 켭니다.

## 5. 정점 미러와 풀링과의 관계

- **vbmir**: 사본 풀에서는 정점과 인덱스가 이미 정상 메모리에 있으므로 **필요 없습니다.** `vbm_bind`는 zero-copy chunk에서만 동작합니다(`gfx_vk_exec.c:463`). 그래서 NVIDIA 사본 모드에서는 자동으로 꺼지고 코드도 그대로 둡니다. auto 모드가 검증되면 정리를 논의합니다.
- **버텍스 풀링(B단계, 기본 OFF)**: zero-copy chunk에서만 pull합니다(`vpull_decide`의 `p->zero_copy` 조건). 사본 모드에서는 자동으로 fallback(POOL 사유)하므로 충돌하지 않습니다.
  - import를 계속 쓰는 드라이버에서 의미가 있는지 검토가 필요합니다. Adreno는 고려 대상이 아니라고 하셨으므로 **보존만 하고 확장하지 않습니다.**
- **s48 range import, bridge chunk** (`gfx_vk_core.c:470-577`): import 계열 장치이므로 사본 모드에서는 쓰지 않습니다. bridge(두 view에 걸친 범위)는 사본에서 어떻게 표현할지 확인이 필요합니다(**위험 항목 R4**).

## 6. 단계

1. **0단계 (조사, 코드 변경 없음)**: copied pool 경로가 NVIDIA에서 지금 어떻게 동작하는지 읽고, `pool_shadow_required`를 강제할 수 있는 지점을 확인합니다.
2. **1a단계 (합성 테스트, 게임 전)**: 검수가 요구한 재현 항목마다 import 경로와 copy 경로의 기대 바이트를 대조합니다. Switch 1과 2 동기화를 모두 확인합니다.
   - 크기 미상 SSBO
   - raw 로그 포화
   - 같은 페이지에서 분리된 CPU/GPU 쓰기
   - alias 사이 GPU→GPU 읽기
   - view 경계의 간접 인자
   - 같은 VA 재등록
   - 이벤트 관찰 직후 payload
   - query와 s56 덮어쓰기 순서
   - 부분 제출 실패
3. **1단계**:
   - `NXN1X_POOL_MODE=copy`로 N1X device-local 사본 (A)을 비교할 수 있게 합니다. 기본 전환은 아래 회귀 검증 뒤에 판단합니다.
   - 4.6절의 P1 대응 전부를 포함합니다.
   - 한 backing의 alias·쓰기 세대와 실제 제출된 범위의 기록, readback staging 및 완료 순서를 먼저 검증합니다.
   - 4 KB 단위로 CPU 쓰기를 추적합니다. volatile 및 바인딩별 지연 업로드는 정확성 기준 경로와 비교합니다.
   - 로그(약 10초마다): 업로드 MB/프레임, 되돌림 MB/프레임, volatile 페이지, fault/s, 사본 총량
4. **2단계** (1단계 측정 결과에 따라):
   - 1단계 실측에서 비용이 큰 전송과 비교 범위 최적화
   - 비동기 되돌림
   - GPU 쓰기 범위 축소
   - CPU 읽기 fault 안전망

## 7. 검증 (게임 실행으로)

| 게임 | 확인 항목 |
|---|---|
| HWAoI | fps(현재 60), GPU ms, 화면, vbmir 없이 같은 성능인지 |
| Bayonetta 3 (2X) | 깊이 패스 ms(현재 40~57 → 목표 수 ms), fps, 가시성 결과(오브젝트 사라짐 없음) |
| 젤다무쌍, 기타 실행 가능 타이틀 | 회귀 없음, 메모리 사용량 |
| DK | 제외(지시) |

각 게임에서 `NXN1X_POOL_MODE=import` 실행과 `copy` 실행을 비교합니다. 조건은 같은 저장 지점, 카메라, 해상도, 캐시 상태입니다. GPU ms는 PASS_TIMES로 봅니다.
함께 기록할 것:
- 업로드·readback GPU 시간
- 완료 스레드의 병합 시간
- CPU fence 대기 p95
- fault/s와 같은 주소 재진입 횟수
- 가장 긴 프레임
- vbmir, 풀링, atomic 경로의 활성 여부

## 8. 위험

- **R1** 되돌림 양: SSBO 바인딩 범위 전체를 되돌리면 프레임당 수십 MB를 비교해야 합니다. 실제로 쓴 페이지만 되돌리는 수단이 필요할 수 있습니다.
- **R2** 게임이 기다리지 않고 GPU 결과를 읽는 경우: import에서는 "결국 보였지만" 사본에서는 다음 완료까지 늦습니다. 펜스 없이 폴링하는 게임이 있으면 문제가 됩니다.
- **R3** 업로드 누락: 바인딩 지점 중 하나라도 "이번 제출에서 쓴다" 표시를 빠뜨리면 오래된 데이터를 읽습니다. raw global memory는 전체 dirty 업로드로 대체합니다.
- **R4** bridge chunk, s48 range, 겹치는 풀(`pool_rank`)의 사본 표현.
- (R5 삭제) 메모리 사용량은 위험 항목이 아닙니다. 이 기기는 128 GB이고, NPARK가 "이 컴에서 메모리가 부족하면 그건 이 세상 게임이 아닌 거야"라고 판단했습니다(15:02).
  - **메모리 예산 한도, 부분 사본, LRU 축출 같은 절약 장치는 두지 않습니다.** GPU가 쓰는 풀은 전부 사본을 만듭니다.
  - 참고값: 바요3에서 GPU가 쓰는 풀은 1,684 MB(로그 실측, 등록 풀 2,653.6 MB 중)입니다. HWAoI 풀은 기록상 약 5 GB입니다(`gfx_vk_core.c:936-940`, 실측 아님).
- **R6** CPU 쓰기 fault 비용: 4 KB 보호가 많아지면 vbvol 이전처럼 fault가 폭주할 수 있습니다. volatile 처리로 완화합니다.
- **R7** (해제) 사본은 (A) device-local입니다. (B)의 벤치 결과는 1절에 기록했습니다.
- **R9** 되돌림이 GPU 복사로 바뀌므로, 복사가 끝난 뒤에야 `g_done`이 공개되는 순서를 지켜야 합니다. 복사는 같은 제출 안에서 하고 barrier가 필요합니다.
- **R8** NCE fault 경로: tebfix 이후 안정적이지만, 보호 페이지 수가 늘면 fault 빈도가 늘어납니다.
