# atomic 예외 경로 2안: dispatch 단위 GPU 스테이징 (hybrid3)

2026-11-21 01:00 KST. 상태: **설계만 있음(코드 없음)**. NPARK 승인 뒤 구현, 테스트, 빌드를 한 번에 합니다.
관련 문서: [GPU_MEMORY_HYBRID_DESIGN.md](GPU_MEMORY_HYBRID_DESIGN.md) 7.0-7.2절.

## 0. 왜 바꾸나 (hybrid1/2의 결과)

| 실행 | 전환 | pre/post(120 제출당) | 결과 |
|---|---|---|---|
| hybrid1 (00:25) | 452 청크, 493 MB (256 MB 블록 닫힘) | 4.9 GB, 그중 읽기 전용 SSBO 65% | 4-5 fps |
| hybrid2 (00:40) | 55 청크, 93 MB | 0.93 GB, 거의 전부 atomic 바인딩 | import+ootfadd보다 느림 |

hybrid2 로그(ab_hybrid_004017)의 내용은 이렇습니다.
- 프레임은 45-96 ms인데 GPU는 16-27 ms입니다. CPU가 한도입니다.
- 실행 중 누적으로 replay pool-sync가 17.5초(호출당 3.7 ms), copyback(병합 비교)이 15.8초였습니다.
- mprotect 8.5k/s(57 ms/s), "GPU-written marks decayed"는 840만/120 제출입니다.

원인: atomic 바인딩의 GPU 결과(120 제출당 약 40 MB)를 매 제출마다 CPU가 비교하고 게임 페이지에 병합합니다. 그러면 그 페이지가 다시 dirty가 되어 동기화되고 보호됩니다.

GPU 결과를 CPU 쪽으로 가져오는 과정 자체가 비용입니다. 그러니 그 과정을 없애면 됩니다.

## 1. 원칙

- 게임 메모리는 계속 **import**입니다(원본은 게임 페이지). CPU 쪽 추적, 동기화, 보호, 병합은 하지 않습니다.
- atomic이 도는 동안만 그 바인딩 범위를 **device-local 스테이징 버퍼**로 옮기고, 끝나면 **GPU가** import로 되돌립니다. 모두 GPU 명령이고 CPU는 관여하지 않습니다.
- 출발점은 eden.inc의 기존 `n1x_atomic`입니다(gfx_vk_eden.inc:456-516, exec.c:4581-4589). 이미 이 방식으로 동작합니다.
  - 특정 fragment 셰이더 2개, slot 0, 256 KB 이하에 대해 draw마다 copy-in → draw → copy-out을 합니다.
  - 스테이징 버퍼는 vk_atomic_scratch(제출 슬롯이 소유, 256 KB)입니다.
  - 이것을 셰이더 키가 아닌 셰이더 정보(is_atomic) 기준으로 일반화합니다.

## 2. 데이터 흐름 (compute, 1단계)

용어:
- 창(window): 같은 제출 안에서 연속된 atomic dispatch 묶음
- R: 창이 스테이징하는 import 범위(바인딩 범위의 합집합)
- S: 스테이징 버퍼(device-local)
- P: 스냅샷 버퍼(device-local)

1. **창 열기**: atomic 바인딩(is_written && is_atomic, import 청크)이 있는 dispatch를 기록하기 직전에 엽니다.
   - barrier A: `ALL_COMMANDS/MEMORY_WRITE → TRANSFER/TRANSFER_READ`. 앞선 GPU 쓰기가 R에 반영된 뒤 읽기 위해서입니다.
   - `vkCmdCopyBuffer(import → S, R)`, `vkCmdCopyBuffer(import → P, R)`: 두 번 복사합니다. P는 3-way 병합의 기준값입니다.
   - barrier B: `TRANSFER/TRANSFER_WRITE → COMPUTE_SHADER/SHADER_READ|SHADER_WRITE`.
2. **dispatch**: R에 겹치는 모든 SSBO 바인딩을 S로 리디렉션합니다(buffer=S, offset=a-R.lo). atomic 바인딩만이 아닙니다(3.2).
3. **다음 dispatch도 창 조건을 만족하면** 창을 계속 엽니다(복사 없음).
   - 조건: 그 dispatch가 R과 겹치는 접근이 모두 리디렉션 가능한 SSBO 바인딩뿐입니다.
   - 새 atomic 범위가 R과 겹치거나 인접하면 R을 넓힙니다. 넓힐 때는 추가 구간만 copy-in합니다.
   - dispatch 사이 barrier는 지금처럼 기록합니다(r89 compute 전용 barrier).
4. **창 닫기**(3.4의 조건 중 하나가 오면):
   - barrier C: `COMPUTE_SHADER/SHADER_WRITE → COMPUTE_SHADER/SHADER_READ`
   - **GPU 병합 셰이더**(hy3_merge.comp, 1회 dispatch, 4바이트 워드 단위):
     ```
     m = byte mask where S != P
     if (m == 0) skip
     else import = (import & ~m) | (S & m)
     ```
     바뀐 바이트만 씁니다. 이 셰이더는 import에 대한 일반 load와 store만 하고 atomic은 쓰지 않습니다.
   - barrier D: `COMPUTE_SHADER/SHADER_WRITE → ALL_COMMANDS/MEMORY_READ|MEMORY_WRITE`. INDIRECT_COMMAND_READ, VERTEX_ATTRIBUTE_READ, INDEX_READ, TRANSFER_READ, HOST_READ를 포함합니다.
5. **제출 끝**: 열린 창이 있으면 닫습니다. 창은 제출을 넘지 않습니다.

S와 P는 제출 슬롯이 소유하는 링(scratch)에 둡니다. 슬롯의 펜스가 끝나면 재사용합니다(지금 vk_atomic_scratch와 같은 수명). 크기는 처음 64 MB로 잡고, 모자라면 키웁니다. 키울 수 없으면 그 dispatch는 import로 두고 카운터만 올립니다.

## 3. 위험과 대책

### 3.1 되돌리는 쓰기가 다른 쓰기를 덮는 문제

- 창이 열려 있는 동안 R의 import 바이트를 바꿀 수 있는 것은 두 가지입니다.
  - (가) 같은 제출의 다른 GPU 명령
  - (나) CPU
- (가)는 3.4 규칙으로 막습니다. R에 리디렉션 없이 닿는 명령이 오면 그 전에 창을 닫습니다.
- (나): CPU가 GPU 실행 중에 R의 다른 바이트를 쓸 수 있습니다. 예를 들어 같은 구조체의 다른 필드입니다. 통째 copy-out은 이 바이트를 copy-in 시점 값으로 되돌립니다. 그래서 3-way 병합을 씁니다. GPU가 바꾼 바이트(S≠P)만 import에 쓰고, 나머지는 CPU 값이 남습니다.
- 남는 한계: 같은 4바이트 워드에서 GPU가 바꾼 바이트와 CPU가 같은 순간에 쓴 바이트가 겹치면 병합 셰이더의 read-modify-write 사이에 경합이 생길 수 있습니다. 창 지속 시간은 GPU 실행 시간(마이크로초~밀리초)입니다. 이 경우는 import에서도 GPU와 CPU가 동기화 없이 같은 워드를 쓰는 게임 쪽 경합이라 실제로 의미가 없다고 봅니다. T15처럼 알려진 한계로 문서에 남깁니다.
- 되돌리는 범위 줄이기:
  - 병합 셰이더는 바뀐 바이트만 쓰므로 범위는 정확성과 무관하고 비용에만 영향이 있습니다.
  - write_extent로 범위를 아는 바인딩(hybrid2 nvn_eden_ssbo_reach)은 R을 그 reach까지만 잡습니다.
  - OoT의 atomic 바인딩은 reach를 모릅니다(hybrid2에서 잘린 것 0.4 MB). 그래서 바인딩 크기 전체가 R입니다.

### 3.2 같은 dispatch의 다른 바인딩과 겹침(aliasing)

- 같은 dispatch의 SSBO 바인딩 중 R과 겹치는 것이 있으면 모두 S로 리디렉션합니다.
- 일부만 겹치면 R을 그 바인딩 범위까지 넓힙니다(합집합). 넓히는 만큼 추가 copy-in을 합니다.
- 리디렉션할 수 없는 바인딩이 R과 겹치면 그 dispatch는 스테이징하지 않습니다(import, 카운터). 해당하는 것은 storage texel buffer(vk_tbuf_view), uniform buffer, raw pool table 경유 접근입니다.

### 3.3 graphics(fragment) atomic

- copy는 render pass 안에서 할 수 없습니다.
- 2단계 방식(n1x_atomic과 같음): pass를 끝내고 → copy-in → 같은 pass를 다시 시작하고 draw → pass 끝(또는 3.4 조건)에서 창을 닫습니다.
- pass 분할 비용이 있으므로, 한 pass에서 같은 R을 쓰는 연속 draw는 한 창으로 묶습니다.
- 1단계 구현 전에 카운터로 OoT의 atomic 바인딩이 compute인지 graphics인지 확인합니다. hybrid1 binds 줄은 atomic을 단계 구분 없이 셉니다. 카운터를 단계별로 나눕니다.
- n1x_atomic(키 2개)은 2단계가 들어가면 일반 경로로 대체하고 제거합니다.

### 3.4 창을 닫아야 하는 명령 (결과를 읽는 순서)

닫는 대상:
- R과 겹치는 indirect 인자(dispatch indirect, draw indirect, count buffer)를 읽는 명령
- vertex/index 버퍼
- copy/fill의 src나 dst
- texel buffer
- draw
- R과 겹치는 접근이 리디렉션 불가능한 dispatch
- counter report와 event signal(CPU가 결과를 볼 수 있는 시점)
- 제출 끝

이런 명령이 오면 그 명령을 기록하기 전에 창을 닫습니다(병합과 barrier D).
- 겹침 판정은 기록 시점에 host가 아는 주소로 합니다(vk_buf_at가 찾는 범위).
- 판정이 불확실하면(주소를 모르는 경우) 닫습니다. 정확성이 우선이고 비용은 카운터로 봅니다.

### 3.5 raw pool table(cbuf 0xFF) atomic

- 주소를 GPU만 알기 때문에 R을 정할 수 없습니다. 그래서 스테이징하지 않습니다(import, hybrid1의 H11 규칙과 같음).
- 2단계 후보: 창이 열린 동안 pool table에서 R 구간의 엔트리를 S의 device address로 바꾼 표를 씁니다. 다만 표를 다시 만드는 비용이 있고, 그 표를 쓰는 다른 셰이더와의 순서 문제가 있습니다.
- `hybrid1: shader … runs an atomic through raw global memory` 카운터로 빈도를 본 뒤 정합니다.

### 3.6 그 밖

- copy/hybrid 모드에서 이미 device-local 사본인 청크(p->dl)는 스테이징하지 않습니다. 이미 빠르기 때문입니다.
- 바인딩 크기가 S 상한을 넘으면 그 dispatch는 import로 둡니다(카운터). 쪼개서 스테이징하지 않습니다.
- vbmir 미러: 창의 병합은 GPU 쓰기이므로 sub_gpu_wrote(R)로 기록합니다. 미러 무효화와 g_gw는 지금 규칙 그대로입니다. 단 hybrid2 (b) 규칙(쓰는 바인딩, reach까지)을 적용합니다.

## 4. hybrid1 청크 전환의 정리

- 권장: `NXN1X_POOL_MODE=hybrid`의 뜻을 "import + 버텍스 풀링/미러 + atomic 스테이징"으로 바꿉니다. 청크 전환은 A/B 스위치(`NXN1X_HY_CONVERT=1`)로만 남깁니다.
- H 셀프테스트(전환)는 그 스위치일 때만 돌립니다.
- 스테이징이 OoT, 바요3, HWAoI에서 확인되면 청크 전환 코드와 H 케이스는 _delete로 옮깁니다. 원칙(스위치는 A/B용)에 맞추기 위해서입니다.
- hybrid2 (b)(쓰는 바인딩만, reach까지)는 그대로 둡니다. copy 모드와 HYBRID_CONVERT에도 계속 적용됩니다.
- is_atomic(변환기)과 nvn_eden_ssbo_reach(브리지)는 그대로 재사용합니다.

## 5. 테스트 (먼저 작성, 셀프테스트 S1-S12)

셀프테스트는 셰이더를 직접 돌리지 않습니다. 대신 창 API를 그대로 부르고, 리디렉션된 바인딩의 쓰기를 S에 대한 GPU fill로 흉내 냅니다. 병합 셰이더는 실제로 돕니다.

| # | 내용 | 기대 |
|---|---|---|
| S1 | 창 하나: S에 쓰기 → 제출 완료 뒤 게임 페이지 | GPU 값이 보임 |
| S2 | 창이 실행 중일 때(hold 훅) CPU가 R의 다른 바이트를 씀 | CPU 바이트 유지, GPU 바이트 반영 |
| S3 | 같은 R에 연속 창 2개 | copy-in/out 각 1회(카운터), 두 쓰기 모두 보임 |
| S4 | 창 다음에 R을 src로 하는 copy | copy 전에 창이 닫혀 copy가 GPU 결과를 읽음 |
| S5 | R 안의 indirect 인자를 쓴 뒤 그것을 읽는 indirect dispatch | 결과 인자로 dispatch됨(닫힘과 barrier D) |
| S6 | 같은 dispatch의 두 번째 바인딩이 R 안에 있음 | 리디렉션(offset 정확), 두 번째로 쓴 값이 보임 |
| S7 | 일부만 겹치는 바인딩 | R이 합집합으로 넓어지고 값이 맞음 |
| S8 | raw pool table atomic | 스테이징 안 함, 카운터 +1, 값은 import로 맞음 |
| S9 | 같은 워드에서 GPU가 0번 바이트, CPU가 1번 바이트를 바꿈 | 둘 다 유지(병합 셰이더 바이트 마스크) |
| S10 | copy 모드에서 dl 청크의 atomic | 스테이징 안 함 |
| S11 | import 모드(스위치 끔) | 스테이징 0, 기존 케이스 그대로 |
| S12 | 제출 끝에 열린 창 | 자동으로 닫히고 값이 맞음 |
| (2단계) S13 | fragment atomic draw | pass 분할 후 값이 맞음 |

기존 COPY 48/48, HYBRID(전환 스위치) 60/60은 그대로 통과해야 합니다.

## 6. 기대 수치 (OoT, hybrid2 로그 기준 추정)

- atomic 바인딩: 120 제출당 1,080개, 1.78 GB입니다. 제출이 프레임당 6개이므로 프레임당 54개, 89 MB입니다.
- 묶지 않는 최악의 경우 프레임당 GPU 트래픽은 copy-in 2회(S, P) 178 MB + 병합 읽기 178 MB + 바뀐 바이트 쓰기(약 2 MB)로 약 360 MB입니다. 125-160 GB/s 기준 **약 2.3-2.9 ms/프레임**이고, barrier가 조금 더 붙습니다.
- 같은 R을 쓰는 연속 dispatch를 묶으면 이보다 줄어듭니다. 묶이는 비율은 카운터로 봅니다.
- CPU 쪽: 사본 청크 0개입니다. pool-sync, copyback, mprotect 증가가 없어 import+ootfadd와 같아야 합니다.
- GPU 쪽 이득: atomic이 device-local로 갑니다(membench7 import 대비 48-97배 빠름). import+ootfadd의 무거운 구간 GPU 34-36 ms에서 atomic 몫을 뺀 만큼입니다. 정확한 몫은 미측정입니다. 실행 뒤 pass 시간(DM_VK_PASS_TIMES)으로 확인합니다.
- 판정 기준: OoT 무거운 구간에서 import+ootfadd(22-25 fps, 프레임 35-45 ms)보다 프레임 시간이 짧아야 채택합니다. 아니면 기본은 import+ootfadd로 둡니다.
- 다음 일로 따로 남긴 것: full barrier 262개/프레임(약 6 ms) 축소.

## 7. 로그 (구현 시)

- `hybrid3 atomic staging, last 120 submissions`:
  - 창 수, 창당 dispatch 수(묶임)
  - copy-in MB, 병합 MB, 바뀐 바이트 MB
  - 닫힌 이유별 수(indirect, copy, draw, 리디렉션 불가, 제출 끝)
  - 스테이징 못 한 수(크기 초과, 리디렉션 불가, raw)
  - 단계별(compute/graphics) atomic 바인딩 수

## 8. 2단계 계획: 창 묶기 + 한도 16 MB (11-21 08:15, 설계만, 승인 대기)

### 8.1 1단계 결과 (ab_tsdiag_080904)

- bdf51d78은 그룹당 3.1 µs에서 0.4 µs가 됐습니다(약 7.7배). 해상도가 953x536으로 달라서 그룹당 시간으로 비교했습니다.
- dispatch 합계는 3.65 ms, span은 18.1 ms입니다.
- 스테이징한 dispatch는 프레임당 약 23개, 버퍼 41개, 약 24 MB입니다. 4 MB 초과로 못 한 것이 프레임당 약 7.6개 남았습니다.
- 남은 gap: dispatch → dispatch 4.64 ms(100개)가 대부분입니다. 스테이징 dispatch마다 복사 barrier, 병합 dispatch(측정 창 밖이라 gap에 들어감), 병합 barrier가 붙습니다.

### 8.2 규칙

- **창**: 제출(명령 버퍼) 안에서 스테이징된 범위 목록 R1..Rn과 그 S, P를 유지합니다. 다음 dispatch의 바인딩이 열린 Ri 안에 있으면 복사 없이 S로 바인딩합니다. 새 atomic 범위는 이 dispatch 앞에 copy-in을 추가하고 창에 넣습니다.
- **닫기**(병합 + barrier, 모든 Ri를 한 번에): s63 래퍼(cbar2가 쓰는 GPU 작업 기록 지점) 중 dispatch가 아닌 GPU 작업이 기록되기 직전입니다.
  - 대상: draw, pass 시작과 끝, copy, fill, update, clear, query begin, 명령 버퍼 end
  - mesh draw는 vk_bar_flush 자리에서 닫습니다.
  - dispatch는 prep에서 "창 호환"으로 표시된 경우에만 창을 유지합니다. 호환 조건: R과 겹치는 바인딩이 모두 S로 리디렉션 가능, indirect 인자가 R 밖, texel buffer와 raw pool table이 R과 겹치지 않음. 아니면 그 dispatch를 기록하기 전에 닫습니다.
  - timestamp와 query reset은 닫지 않습니다. 측정용이고 R에 닿지 않습니다.
  - 창 안에서 스테이징 자체가 기록하는 copy, dispatch, barrier는 재진입 플래그로 제외합니다.
- **counter report / event**: CPU가 결과를 보는 시점은 제출 완료 뒤(post op)이므로 제출 끝 닫기로 충분합니다. 단 counter report가 GPU 명령(timestamp 이후 copy 등)을 기록하면 그 명령의 래퍼에서 닫힙니다.
- **한도**: 바인딩 16 MB까지입니다. 창 하나의 S+P 총량은 64 MB까지이고, 넘으면 닫고 새로 엽니다.
- **스위치**: `DM_VK_HY3=0`이면 끄고, `DM_VK_HY3=1`이면 1단계(dispatch마다 닫기)입니다. 기본은 2입니다.

### 8.3 테스트 (먼저 작성, `#if defined(HY3_PHASE2)` 아래 gfx_vk_eden.inc, 2단계 구현 때 켬)

- 공통 방식:
  - import 청크 D(import, hybrid 셀프테스트)에서 GPU 경로로 확인합니다.
  - "dispatch"는 hy3 테스트 훅이 대신합니다. 바인딩, copy-in, S에 fill로 쓰기를 하고 닫지 않습니다.
  - 닫힘은 실제 래퍼 경로(mtl_sub_copy, mtl_sub_fill, mtl_sub_commit 등)로 일으킵니다.
- 케이스:

| # | 내용 | 기대 |
|---|---|---|
| S3 | 같은 R에 연속 dispatch 2개(각각 다른 16바이트에 씀) | copy-in 1회, 병합 1회(카운터), 두 쓰기 모두 게임 페이지에 |
| S4 | 창 다음에 R을 src로 하는 mtl_sub_copy | copy 직전에 닫힘(병합 1회), copy 대상이 GPU 값을 읽음 |
| S5 | 창 다음에 R 안의 indirect 인자를 읽는 dispatch | 그 dispatch 전에 닫힘, 인자가 병합된 값 |
| S12 | 창이 열린 채 mtl_sub_commit | 명령 버퍼 end 전에 닫힘, 값이 맞음 |
| S13 | 창 다음에 draw(pass 시작) | pass 시작 전에 닫힘 |
| S14 | 창 다음 dispatch가 R 위 texel buffer를 바인딩 | 닫힌 뒤 그 dispatch는 R에 대해 스테이징 안 함 |
| S15 | 창 다음에 counter report | 제출 끝에서 닫힘, 보고값과 R 값이 모두 맞음 |
| S16 | 16 MB 바인딩 | 스테이징됨(1단계는 4 MB 초과면 안 함) |
| S17 | 창 총량 64 MB 초과 | 닫고 새 창 |

### 8.4 다음 실행의 판정 기준 (해상도 차이 때문에 절대 ms 대신)

- 그룹당 시간:
  - bdf51d78은 0.4 µs/그룹 이하를 유지합니다.
  - 4ef1f343, 39e60874, a0a84773(지금 4 MB 초과로 스테이징 안 됨)은 그룹당 시간이 줄어야 합니다. 같은 이유로 bdf51d78처럼 몇 배 줄 것으로 예상합니다.
- 전환별 gap: dispatch → dispatch gap의 개당 평균(지금 4.64 ms/100 = 46 µs)이 줄어야 합니다. 목표는 약 25 µs 이하입니다.
- stage1 줄의 "windows / dispatches a window"(새 카운터)로 묶임 정도를 봅니다.
- 화면 깨짐 확인은 필수입니다. 깨지면 `DM_VK_HY3=1`(1단계)로 같은 장소를 봅니다.

## 9. stage2s 결과 (stage2s1, ab_tsdiag_082824, 08:28 KST, OoT import, TS_FIX)

- 셀프테스트: HYBRID 93/93, COPY 73/73 PASS. 오류는 기존 RED 1줄(SetComputeWorkloadPriority)뿐입니다.
- 창 묶기:
  - 창 8,192개에 스테이징된 dispatch 195,155개(창당 약 24개)
  - 복사한 버퍼 45,538개(stage1은 235,809개)
  - 16 MB 초과로 못 한 것 0(stage1에서는 4 MB 초과가 프레임당 7.6개)
  - 취소 0, 메모리 부족 0
  - 닫힌 이유: 창을 못 쓰는 dispatch 앞 4,097, 그 밖 GPU 작업 앞 4,095
- 시간:
  - dispatch 합계 2.65 ms(159개). stage1은 3.65 ms(116개)였습니다.
  - 4ef1f343, 39e60874, a0a84773, bdf51d78은 모두 상위 8개(0.38 ms 이상) 밖으로 빠졌습니다.
  - 가장 무거운 pass는 53d38847 0.83 ms입니다(이미지 작업, atomic 없음).
- gap: dispatch → dispatch 2.79 ms / 147.7개 = **개당 18.9 µs**입니다(stage1은 46 µs, 목표 25 µs 이하 달성).
- span 21.73 ms = pass 13.57 + **part 경계 4.70** + gap 3.45입니다. 경계가 1.48 ms(stage1)에서 늘었습니다.
- fps 27-34, longframe 프레임 37-47 ms, GPU 25-35 ms
- 다음으로 큰 몫:
  - part 경계 4.7 ms/프레임(6.9회). 명령 버퍼 part 사이에 GPU가 다음 part 제출을 기다리는 시간이고, CPU 기록과 제출 쪽입니다.
  - 그 다음은 pass 자체(13.6 ms)입니다.
