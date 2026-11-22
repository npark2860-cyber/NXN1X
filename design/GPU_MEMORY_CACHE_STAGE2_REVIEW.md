# GPU 메모리 캐시 2단계 설계 검수

2026-11-20 · [검수 요청](ASTRA_STAGE2_REVIEW_REQUEST.md) · [검토한 설계](GPU_MEMORY_CACHE_STAGE2_DESIGN.md)

**판정. 4 KB dirty 페이지 선별은 기존 대기와 shadow를 유지한 채 먼저 비교할 수 있습니다. 제출 간 대기 제거와 volatile 페이지의 동시 병합은 설계 수정이 필요합니다.** 에폭별 첫 쓰기 스냅숏은 값의 변화는 보여 주지만 그 값을 쓴 주체와 쓰기 순서는 기록하지 않습니다.

## 1. 두 제출만으로 병합이 틀리는 반례 · P1

설계 36~37행과 55~58행의 규칙을 같은 페이지의 두 바이트 X, Y에 적용했습니다. X는 GPU만 쓰고 Y는 CPU만 씁니다. 두 제출은 같은 큐에서 실행되고 완료 스레드도 제출 순서를 지킵니다.

| 순서 | 동작 | 게스트 (X,Y) |
|---|---|---|
| 1 | 초기값은 (0,0). 제출 N이 GPU에서 X=1을 기록. N의 readback은 (1,0) | (0,0) |
| 2 | N의 병합 전 CPU가 Y=7을 기록. 첫 쓰기 직전 스냅숏 S[N]=(0,0) | (0,7) |
| 3 | N+1이 Y=7을 업로드하고 GPU에서 X=2를 기록. N+1의 readback은 (2,7) | (0,7) |
| 4 | 여전히 N의 병합 전 CPU가 Y=8을 기록. S[N+1]=(0,7) | (0,8) |
| 5 | N을 병합하면 X=1이 반영되고 Y=8은 보존됨 | (1,8) |
| 6 | N+1을 병합할 때 X는 G=1, B=0, R=2. 설계의 G==B 조건이 거짓이라 X=2를 건너뜀 | **(1,8)** |

기대값은 **(2,8)**입니다. 문서의 식을 독립적인 두 바이트 모델로 실행해 (1,8)을 확인했습니다. GPU 완료 시각이나 동일 바이트 CPU/GPU 경합에 의존하지 않는 반례입니다. 반대로 이전 GPU 병합이 만든 차이를 업로드에서 CPU 변경으로 분류하면 그 값이 다음 GPU 쓰기를 덮을 수도 있습니다. 설계 58행의 “이미 사본과 같은 값”은 여러 제출이 진행될 때 성립하지 않습니다.

**설계 수정.** 물리 backing·할당 세대별로 CPU가 쓴 범위와 GPU 결과를 게스트에 공개한 범위를 구분하고, 각 제출이 사용한 세대를 보존해야 합니다. 값 비교만으로는 같은 값 재기록이나 A→B→A도 구분할 수 없습니다. 쓰기 범위를 관측할 수 없는 페이지에는 의존하는 작업만 이전 병합까지 기다리는 기준 경로가 필요합니다. 특히 N에는 첫 CPU 쓰기가 없고 N+1에서 처음 썼다면 S[N] 자체가 없습니다. 그 경우에 대한 판정과 스냅숏의 마지막 사용 시점도 명시해야 합니다.

## 2. 첫 쓰기 관측과 volatile · P1

- **실제 누락 후보.** `Memory::WriteBlock`과 `WriteBlockUnsafe`는 backing에 쓰기 전에 `BeforeGuestWrite`를 호출합니다. 반면 `Write8/16/32/64`와 `WriteExclusive*`는 `ObserveGuestWrite`를 부르고, 이 함수는 범위를 `nvnc_cpu`에 전달할 뿐 `nvn_dirty`의 pre-write 스냅숏 훅을 부르지 않습니다. 보호된 게스트 VA를 거치지 않는 backing 쓰기라면 첫 쓰기 증거가 빠집니다. exclusive는 시도 전에 통지하므로 실패한 연산까지 실제 쓰기로 계산해서도 안 됩니다. [메모리 쓰기 진입점](../source/src/core/memory.cpp) · [두 통지 함수](../source/src/core/nvn/nvn_layer.cpp)
- **alias와 GPU 되돌림.** `prepare_write_range`는 전달받은 VA와 겹치는 등록 범위를 순회합니다. 다른 VA의 보호 상태는 같은 물리 페이지라는 이유만으로 바뀌지 않습니다. s56은 `nvn_dirty_gpu_write`로 stamp를 남긴 뒤 backing alias로 쓸 수 있고, `pc_merge`와 이벤트·카운터도 게스트 페이지를 갱신합니다. GPU 결과의 게스트 반영, 호스트가 새로 만든 값, 게스트 CPU 쓰기를 구분해야 합니다. 이벤트·카운터·s56이 **device-local에 아직 없는 새 값**이라면 다음 GPU 소비 전 반영이 필요합니다. 모든 완료 스레드 쓰기를 일괄 제외하는 방식도 틀립니다. [페이지 추적](../host/dm/gfx/nvn/nvn_dirty.c) · [s56 경로](../host/dm/gfx/nvn/hle_nvn.c)
- **에폭 전환.** 설계 50행처럼 staging 채취 후 재보호하면 그 사이의 쓰기가 이번 staging에도 다음 첫 쓰기 기록에도 없을 수 있습니다. 재보호·세대 교체·채취를 모든 쓰기 alias와 조정해야 합니다. `dm_guest_write_prepare`와 실제 store 사이, `pc_merge`의 비교와 게스트 store 사이도 같은 문제입니다. 이미 쓰기 가능한 페이지에서 CPU가 계속 쓰는 동안 4 KB를 복사하면 두 시점이 섞일 수 있습니다.
- **3회 진입·8회 복귀.** 임계값은 성능 조절에만 사용합니다. 무보호 페이지에서 “8회 쓰이지 않음”은 단순 stamp 또는 스냅숏 동일성으로 판정할 수 없습니다. 값 차이만 관측했다면 “8회 차이가 없음”으로 기록하고, 복귀 시 기준값 확정·재보호·세대 교체를 한 절차로 처리해야 합니다. GPU 쓰기와 겹치고 CPU 변경 범위를 알 수 없는 페이지에는 volatile 양방향 병합을 적용하기 어렵습니다. CPU 전용 업로드 페이지부터 시험하는 편이 검증 가능합니다.
- **fault 훅의 제약.** 현재 훅은 페이지를 writable로 바꾸기 전에 실행되며 fault 경로에는 동적 할당과 잠금을 쓰지 않습니다. 충돌 없는 저장소는 제출 측에서 준비하고, 포화 시에도 기준본 보존 또는 안전한 직렬 경로로의 전환을 보장해야 합니다. 다른 사용자가 같은 페이지를 텍스처 추적에 사용하는 동안 volatile 정책이 그 보호를 해제하지 않도록 해야 합니다. [nvn_dirty.c](../host/dm/gfx/nvn/nvn_dirty.c)

## 3. 제출 수명, 읽기 의존성, 부분 실패 · P1

**R9.** 기존 `pc_readback`은 모든 풀의 `p->rb_sub`와 `p->rb_whole`을 재설정합니다. 완료 시에는 현재 풀 목록에서 `p->rb_sub == s->pc_nsub`를 검사합니다. N+1을 먼저 기록하면 N의 readback slice를 따로 할당했더라도 N 결과가 건너뛰어집니다. N+1이 그 풀에 쓰지 않아도 재설정됩니다. 각 제출이 자기 readback 범위, slice, backing ID·세대, 실제 접수된 명령 부분, raw write log 및 수명을 소유하도록 바꿔야 합니다. 해제·같은 VA 재등록도 끝까지 식별해야 합니다. [pc_readback과 pools_copy_back](../host/dm/windows/gfx_vk_core.c)

GPU 의존성은 이전 접근 → 업로드 transfer write → 소비자, 마지막 write → readback transfer → host read 순서입니다. 같은 큐의 제출 순서만으로 메모리 의존성이 모두 생기지는 않습니다. 비일관 매핑은 해당 slice의 flush/invalidate 성공과 atom 경계를 확인하고, readback은 CPU 병합 완료 후에 반환합니다. 기존 slot 재활용에는 GPU fence만 기다리는 설정도 있어 완료 스레드가 참조하는 데이터 수명은 별도로 보장해야 합니다. [Vulkan 동기화 예제](https://docs.vulkan.org/guide/latest/synchronization_examples.html)

**링 포화 외의 대기.** 다음 제출의 호스트 리플레이가 앞 제출의 결과를 CPU에서 해석하거나, 완료 스레드의 s56·counter가 넣을 값을 다음 GPU 제출이 소비한다면 데이터 의존성이 있습니다. 해당 범위의 공개를 기다리거나 같은 큐에서 직접 전달하는 경로가 필요합니다. 기존 `pool_sync_one`에도 allocation별 `wait_fence(use)`가 있습니다. 최상위 `mtl_wait`만 제거하면 의존성을 빠뜨리면서 병목 일부는 그대로 남을 수 있습니다. [제출 시작·풀 동기화·완료 경로](../host/dm/windows/gfx_vk_core.c)

**S4.** 현재 `pc_dropped`는 모든 사본을 미동기화로 표시합니다. 예를 들어 N의 GPU 쓰기가 접수됐고 N+1의 앞부분도 이미 실행됐는데 N+1의 마지막 버퍼가 거절된 경우, 오래된 게스트 페이지 전체를 업로드하면 실제 실행된 쓰기까지 덮습니다. 복구 중에는 영향을 받는 backing의 새 제출을 보류하고, 접수된 부분의 완료와 필요한 별도 readback을 끝낸 뒤 실패한 부분의 CPU 변경만 재적용해야 합니다. 접수 여부를 확정할 수 없는 장치 오류는 별도 오류 상태로 처리해야 합니다. 기존 case 9는 `vkEndCommandBuffer` 뒤 결과를 강제로 실패시키며 앞부분을 분할 제출하지 않습니다. 실제 queue submit 실패와 부분 성공은 따로 시험해야 합니다. [vk_cmd_split·pc_dropped·selftest](../host/dm/windows/gfx_vk_core.c)

## 4. 먼저 진행할 성능 작업과 통과 조건

1. **대기를 유지하고 4 KB 선별만 A/B 비교.** 현재 64 KB는 dirty 후보를 고르는 단위이고 내부 `memcmp`는 이미 4 KB입니다. HWAoI 로그에는 120제출에 704.0 MiB 비교, 657.6 MiB 동일이 기록됐습니다. 차이 46.4 MiB는 달라진 페이지·구간의 크기입니다. 실제로 달라진 바이트 수는 이 로그로 측정할 수 없습니다. 후보 탐색 횟수, 검사한 페이지, 읽은 바이트, GPU에 보낸 바이트와 복사 명령 수를 분리해 측정하십시오. [1단계 HWAoI 로그](../logs_poolcopy/ab_copy_160639.txt)
2. **검증을 결정적으로 실행.** 테스트에서 N과 N+1을 동시에 대기 상태로 만들고, 둘의 readback과 CPU 쓰기를 정해진 순서로 해제해야 합니다. 위 (2,8) 반례, S[N] 부재, CPU 동일값 재기록, 다른 VA의 CPU 쓰기, 실패 exclusive, staging 채취 도중 store, 완료 병합 도중 store, volatile 진입·복귀, 링 재사용·동일 VA 재등록, 앞부분 접수 후 뒷부분 거절을 포함합니다. case 3은 CPU에 `pct_cpu`의 prepare를 사용하고 기존 alias 테스트는 GPU→GPU이므로 이 두 테스트만으로 실제 NCE·호스트 backing 쓰기를 검증할 수 없습니다.
3. **기존 실패를 구분.** 1단계의 copy와 import는 각각 11/12였습니다. 양쪽 모두 case 8의 ‘counter report 뒤 fill’에서 기대값 0x82가 남지 않았습니다. 이 항목은 별도 기존 결함으로 기록하고 양방향 순서 모두 계속 검사해야 합니다. [copy 결과](../logs_poolcopy/selftest_copy_154230.txt) · [import 결과](../logs_poolcopy/selftest_import_154208.txt)
4. **게임 성능 기준.** HWAoI 4,800프레임 시점 직전 300프레임에서 frame p50/p95는 36.23/67.87 ms, submit+present p50은 23.38 ms, GPU p50은 9.91 ms였습니다. 별도 샘플의 16,219 faults/s와 80,245 pre-write unprotects/s는 `nvn_dirty` 전체 값입니다. poolcopy 원인별 수치와 지연의 실제 중첩을 측정해야 예상 회복량을 알 수 있습니다. 바요네타 3에서는 이미 얻은 GPU 이득과 원본 프레임 p95가 유지되는지도 같은 장면으로 비교하십시오. [HWAoI 실행](../logs_poolcopy/ab_copy_160639.txt) · [바요네타 3 copy](../logs_poolcopy/ab_copy_160909.txt) · [바요네타 3 import](../logs_poolcopy/ab_import_161134.txt)

검증 범위는 설계식의 독립 모델 실행, 1단계 소스 정적 검토와 보관 로그 대조입니다. 2단계 구현 및 게임 실행 성능은 아직 검증 전입니다.
