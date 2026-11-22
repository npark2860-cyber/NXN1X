# 검수 요청: GPU 메모리 캐시 2단계 (직렬화 제거, 4 KB dirty 비교, volatile 페이지)

검수 대상은 [GPU_MEMORY_CACHE_STAGE2_DESIGN.md](GPU_MEMORY_CACHE_STAGE2_DESIGN.md)이고, 구현 전 설계 검수입니다.

## 배경

1단계 copy 모드의 정확성은 합성 테스트 12항목 중 11항목 통과로 확인했습니다. 남은 1건(카운터가 나중에 쓰는 문제)은 import에서도 같습니다. NCE 크래시는 tebfix2로 고쳤습니다.

HWAoI 실측은 22~24 fps입니다(import는 60). GPU는 9.9 ms로 같아서, 비용은 전부 CPU 쪽입니다.
- 앞 제출 대기: 8~10 ms/프레임
- 동기화 비교: 6~7 ms/프레임
- 쓰기 예외: 1.6만 건/s, 호스트 사전 해제 8만 건/s

## 검수 중점

1. **기준본 대체**: shadow("제출 시작 때 사본 내용") 대신 "에폭별 첫 쓰기 스냅숏 `S[e](P)`"를 씁니다. 이 기준으로 업로드(바뀐 바이트만)와 병합(3-way)이 진행 중 제출이 여러 개여도 정확한지 확인해 주세요. 설계 2.1·2.2의 판정 규칙, 특히 에폭 N 이후와 N+1 이후 쓰기가 섞이는 경우를 봐 주세요.
2. **스냅숏 누락(S1)**: 첫 쓰기 예외를 거치지 않는 CPU 쓰기 경로를 찾아 주세요. 후보는 보호되지 않은 페이지, alias VA, prepare 없는 호스트 쓰기, NCE 외 스레드입니다. volatile을 "동기화 시점 스냅숏"으로 처리하는 방식이 이를 메우는지도 판단해 주세요.
3. **R9 순서**: 업로드 staging → 소비자, 마지막 쓰기 → readback → host, 완료 → 병합 → `g_done` 순서가 직렬화 제거 뒤에도 유지되는지. 남는 대기는 링이 가득 찼을 때뿐인데, 그것으로 충분한지.
4. **부분 제출 실패(S4)**: 진행 중 제출이 있을 때 전체 재동기화의 순서가 맞는지.
5. **volatile 정책**: 3회 연속 쓰이면 volatile, 8회 쓰이지 않으면 복귀하는 정책의 정확성과 비용.
6. **검증 계획(7절)**: 교차 쓰기 합성 테스트 범위가 충분한지.

## 참고

- 1단계 검수: `design\GPU_MEMORY_CACHE_REVIEW.md`
- 1단계 구현: `host\dm\windows\gfx_vk_core.c`의 poolcopy 블록, `pc_readback`, `pools_copy_back`의 dl 분기, `vk_poolcopy_selftest`
- 실측 로그: `logs_poolcopy\`, HWAoI copy 실행(16:01)은 `%APPDATA%\NXN1X\log\nxn1x_log.txt`
