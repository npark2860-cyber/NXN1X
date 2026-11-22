# N1X GPU 메모리 캐시 설계 검수

2026-11-20 · 대상은 같은 날 15:02에 device-local 사본 (A)로 결정한 설계입니다.

**판정은 (A)의 비교 구현 진행, 기본값 전환 보류입니다.** GPU가 사용하는 풀 전체에 사본을 두는 결정과 양립합니다. 아래 문제는 사본 용량과 무관하며, CPU와 GPU가 같은 게스트 데이터를 다른 시점에 읽고 쓸 때 발생합니다.

## 구현 전에 고쳐야 할 사항

| 우선순위 | 발견한 문제와 영향 | 설계에 필요한 처리 |
|---|---|---|
| P1 · R9 | device-local에서 import 원본으로 GPU가 직접 복사하면 이미 게스트 페이지가 덮입니다. 예를 들어 GPU가 한 페이지의 첫 4바이트만 바꾸고 CPU가 다음 4바이트를 갱신했을 때, 바인딩 전체를 직접 되돌리면 완료 스레드 진입 전에 CPU 변경이 사라집니다. 뒤의 `snap_merge`로 복구할 수 없습니다. | 일반 readback 목적지는 제출별 staging으로 둡니다. CPU가 볼 수 있는 값은 세대와 바이트 범위를 확인해 병합합니다. import 직접 쓰기는 복사 범위와 배타적 소유권을 양쪽 alias에서 증명한 경우만 최적화합니다. |
| P1 · R1/R2 | 그래픽의 쓰기 가능 `RK_SBUF_UBO`에서 크기 미상이면 `g_sbufu_w++`만 합니다. raw global 쓰기 로그가 포화되면 `gml_copy_back`은 포화 횟수를 세고 남은 슬롯만 순회합니다. 넓은 host barrier와 성공한 fence가 있어도 누락 주소는 readback에 들어오지 않습니다. | 주소·크기를 확정할 수 없는 쓰기는 해당 명령이 접근 가능한 backing 집합에 보수적으로 기록합니다. 포화 시에도 손실 없는 대체 경로가 필요합니다. GPU 로그로 정확한 페이지가 **실행 후** 결정된다면 첫 제출에서 후보 범위를 staging에 복사하거나, 로그 판독 후 별도 전송의 완료까지 기다려 공개합니다. |
| P1 · R9 | 현재 사본은 CPU가 매핑한 `p->shadow`를 전제로 합니다. 이 포인터로 사본 유무까지 판정하고, `memcmp`·`memcpy`·`snap_merge`에서도 직접 읽습니다. N1X의 type 1 device-local은 host-visible 속성이 없습니다. 할당 타입만 바꾸면 경로가 건너뛰거나 유효한 CPU 주소를 얻지 못합니다. | 렌더용 device-local 버퍼, 업로드 원본, CPU가 읽을 readback slice를 분리합니다. 사본 유무와 CPU 매핑 여부도 분리합니다. slice는 게스트 병합이 끝난 뒤에 재사용합니다. |
| P1 · R4 | `pool_make_0`의 bridge 분기는 zero-copy 비활성화 검사보다 먼저 실행됩니다. sub-view가 거절되면 버퍼 없이 반환할 수도 있습니다. 겹치는 두 사본에 같은 `gw` 범위를 적용하면 뒤에 처리한 오래된 사본이 새 값을 덮을 수 있습니다. | 게스트 물리 backing과 offset, 할당 세대를 기준으로 권위 있는 사본을 결정합니다. 한 view에서 쓰고 같은 제출의 다른 view에서 즉시 읽는 경우에도 최신 값이 보여야 합니다. 경계에 걸친 단일 descriptor·간접 인자·BDA에는 연속 범위를 제공하고, 해제 후 동일 VA 재등록은 별도 세대로 취급합니다. |
| P1 · R2/R3 | `g_done`의 release와 `mtl_wait`의 acquire는 해당 ticket이 성공적으로 기다린 경우에 적용됩니다. `nvnEventGetValue`는 게스트 주소를 직접 읽고 buffer invalidate는 빈 함수입니다. CPU가 완료 이벤트를 본 직후 payload를 읽거나 GPU가 같은 제출 안에서 CPU 이벤트를 기다리면 제출 끝 병합 하나로는 부족할 수 있습니다. | 이벤트·카운터·query·mapped-range invalidate별 데이터 공개 시점을 정합니다. GPU 내 소비와 CPU 통지를 구분하고, 중간 공개가 필요한 명령은 해당 지점에서 전송과 공개를 실행하도록 설계합니다. s56과 buffer write가 같은 backing을 가리키면 뒤에 실행한 게스트 명령이 이겨야 합니다. |

소스 근거는 [그래픽 SSBO 바인딩](../host/dm/windows/gfx_vk_exec.c), [사본 할당·raw write log·완료 스레드](../host/dm/windows/gfx_vk_core.c), [NVN 이벤트 및 invalidate](../host/dm/gfx/nvn/hle_nvn.c)입니다. 특히 `gfx_vk_exec.c:2720`, `gfx_vk_core.c:902, 1290, 1351, 1904, 1920, 2980, 3425, 3488`, `hle_nvn.c:1100, 2878, 2911`에서 확인할 수 있습니다.

## R9 · 전송과 공개 순서

같은 Vulkan 큐를 사용하는 기준 경로입니다.

1. **업로드 원본 확정.** 복사가 끝날 때까지 내용과 수명이 유지되는 import 범위만 직접 읽습니다. CPU가 먼저 재사용할 수 있는 페이지나 import가 거부된 범위는 그 세대의 staging slice에 복사합니다. 비일관 매핑은 제출 전에 atom 경계에 맞춰 flush합니다.
2. **GPU 소비 직전까지 의존성 연결.** 업로드의 transfer write가 실제 소비자의 stage/access에 보이게 합니다. 기존 `vk_vbm_up_close`는 vertex/index fetch만 포함합니다. shader storage·uniform·texel·indirect·transfer 소비자도 각각 포함해야 합니다. 이미 사용 중인 사본 범위를 덮을 때는 앞선 GPU 접근과 다음 transfer write 사이의 의존성도 필요합니다.
3. **마지막 GPU 쓰기 → transfer read → readback.** 셰이더·transfer·해당 확장 쓰기에서 복사 읽기로 이어지는 의존성을 설정하고 staging으로 복사합니다. staging의 transfer write를 host read에 공개한 뒤 해당 작업을 포함하는 timeline 값을 signal합니다. 현재 `pool_sync_mode()`는 Switch 1에서 0이고 `pool_sync_host_barrier`는 즉시 반환합니다. (A)의 readback 의존성은 양쪽 SDK에서 실행해야 합니다.
4. **GPU 완료 → CPU 병합 → 게스트 공개.** timeline wait 성공 후 비일관 staging을 invalidate하고, 세대가 유효한 결과만 게스트에 반영합니다. 이벤트·카운터·s56의 선후 관계를 지키고 마지막에 `g_done`을 공개합니다. submit이 분할되어 앞부분만 접수되고 마지막이 거절되면, 실제 실행된 부분의 쓰기와 전송을 별도로 정산해야 합니다. signal-only 복구의 fence를 누락된 전송의 완료로 계산하면 안 됩니다.

현재 코드에서 GPU 완료, 게스트 반영, `g_done` 공개는 서로 다른 시점입니다. host-coherent 매핑에서도 실행 순서와 수명이 필요합니다. 비일관 매핑은 invalidate 자체가 GPU를 기다려 주지 않습니다. [Khronos의 host invalidate 규정](https://docs.vulkan.org/refpages/latest/refpages/source/vkInvalidateMappedMemoryRanges.html) · [Vulkan 동기화 예제](https://docs.vulkan.org/guide/latest/synchronization_examples.html)

## R1·R3·R6·R8 및 기존 경로 재사용

- **되돌림 양.** `gw`는 실제 쓰인 바이트가 아닌 후보 범위입니다. `sub_gpu_wrote`는 배열이 차기 전에도 가까운 범위를 합칩니다. 크기 확대만으로는 빈 구간 비교가 줄지 않습니다. 바인딩 크기, GPU 전송, CPU 비교, 최종 게스트 반영 바이트를 각각 측정해야 정확한 dirty bitmap의 이득을 판단할 수 있습니다.
- **업로드 시점.** `mtl_sub_begin`에서 동기화할 때는 이후 리플레이할 바인딩이 아직 확정되지 않습니다. 미리 파악한 CPU 변경 범위를 올리는 기준 경로부터 검증하고, 지연 방식은 first use와 early submit 경계에서 비교하십시오. descriptor 재사용, 셰이더 임의 BDA, 간접·mesh draw, texel/texture 전송, XFB와 CPU가 해석하는 descriptor 원본까지 포함합니다. 같은 범위를 읽는 두 draw 사이의 유효한 변경을 제출 맨 앞 업로드로 합치면 첫 draw 결과도 바뀝니다.
- **페이지 충돌.** `snap_merge`는 32비트 단위이며 스냅샷 슬롯이 충돌하거나 세대가 맞지 않으면 건너뜁니다. CPU와 GPU가 같은 단어의 서로 다른 바이트를 수정해도 한쪽 값이 사라질 수 있습니다. 보호 해제 상태의 volatile 페이지를 무조건 전부 업로드하면 최신 GPU 바이트도 덮습니다. 동기화된 시점의 기준값과 세대, 변경 범위를 함께 관리하고 병합 도중의 CPU 재쓰기까지 처리해야 합니다.
- **fault 비용.** 기존 `pool_tracked`는 기본 4 MiB 이하 사본에서 보호를 생략합니다. 코드에는 작은 bridge의 반복 fault와 멈춤 이력이 있습니다. 보호 재설정과 기준값 채취 사이의 CPU 쓰기를 놓치지 않게 하고, 미추적 페이지의 정확성 경로를 유지하십시오. fault/s 외에 동일 주소 재진입, CPU 대기 p95, 가장 긴 프레임을 기록하십시오. 현재 `nvn_dirty`의 쓰기 fault 처리와 읽기 fault 기반 다운로드는 별도 구현입니다. [페이지 추적 코드](../host/dm/gfx/nvn/nvn_dirty.c)
- **Eden과 정점 미러.** NVN의 자체 큐·주소 체계를 유지하면서 Eden의 범위 추적, 제출 확정, 새 쓰기가 오래된 다운로드보다 우선하는 규칙을 이식하는 방향에 동의합니다. 기존 `nvnc_accept`도 제출 부분별 접수 근거로 활용할 수 있습니다. `vbm_bind`와 `vpull_decide`에는 zero-copy 조건이 있습니다. 사본 전환은 atomic 최적화의 적용 조건에도 영향을 줄 수 있으므로 게임 A/B에 각 경로의 활성 여부를 함께 남기십시오. [정점 풀링 조건](../host/dm/windows/gfx_vk_eden.inc)

## 측정 근거와 검증 순서

N1X에서 ARM64 독립 벤치로 측정했습니다. 수치는 같은 실행에서 type 1 device-local과 비교한 GPU 시간 배율이며 9회 중앙값입니다. 각 실행의 클럭 조건이 달라 서로 다른 열의 절대 시간을 직접 나눠 게임 성능 향상률로 해석할 수 없습니다.

| 연산 | import / local | type 2 / local | type 3 / local |
|---|---:|---:|---:|
| 연속 SSBO store 16 MB | 1.5× | 2.8× | 2.7× |
| 1 MB에 분산한 atomicAdd 400만 회 | 21.6× | 2.0× | 1.7× |
| 160 KB 정점 버퍼를 99개 draw에서 재사용 | 15.7× | 1.0× | 1.0× |

type 2의 flags는 `0x6`, type 3은 `0xe`입니다. 기존 사본 할당기는 `HOST_CACHED`를 우선하므로 두 타입이 모두 허용되면 type 3을 고릅니다. R7은 (A) 선택으로 해제됐고 위 수치는 연산별 근거로 남깁니다. R5는 최신 요청에 따라 검수 범위에서 해제됐습니다. [벤치 소스](../build/gpu-cache-review/r7.c) · [import 원자료](../build/gpu-cache-review/type0.txt) · [type 2 원자료](../build/gpu-cache-review/type2.txt) · [type 3 원자료](../build/gpu-cache-review/type3.txt)

1. 먼저 합성 테스트에서 import 경로와 device-local 경로의 **기대 바이트 값**을 대조합니다. 미상 크기 SSBO, raw 로그 포화, 같은 페이지의 분리된 CPU/GPU 쓰기, alias 간 GPU→GPU 읽기, view 경계의 간접 인자, 같은 VA 재등록, 이벤트 관찰 직후 payload, query/s56 덮어쓰기 순서, 부분 제출 실패를 각각 재현합니다. Switch 1과 2에서 동기화 검증도 함께 실행합니다.
2. 이후 같은 저장 지점·카메라·해상도·캐시 상태에서 게임 A/B를 비교합니다. 기존 바요네타 3 NVN 2X 로그의 1600×896 패스 #6은 540 draw에서 **42.09 ms**, 576 draw에서 **51.90 ms**였습니다. 인접한 색상 패스 #18은 각각 **0.76 ms**, **0.54 ms**였습니다. 패스 병목은 확인되지만 이 로그만으로 같은 주소에 집중되는 fragment store의 기여분은 분리할 수 없습니다.
3. 바요네타 3에서는 같은 패스의 GPU ms와 가시성 결과, 젤다무쌍에서는 기존 60 FPS 장면의 원본 FPS와 프레임 시간 p50/p95를 비교합니다. 둘 다 업로드·되돌림 GPU 시간, 완료 스레드의 비교·병합 시간, CPU fence 대기를 기록해 병목이 다른 곳으로 이동했는지 판단합니다.

이번 검수에서 실행한 작업은 독립 벤치이며, device-local 캐시를 연결한 게임 실행·화면·FPS는 후속 구현 검증 항목입니다.
