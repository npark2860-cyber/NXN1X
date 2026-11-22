# 검수 요청: GPU 메모리 캐시 설계 (N1X 근본 수정)

대상 문서: `design\GPU_MEMORY_CACHE_DESIGN.md`. 구현 전 설계 검수입니다.

## 배경 요약

N1X에서는 GPU가 import한 게스트 메모리를 다룰 때 여러 경로가 느립니다.
- 고정 기능 정점·인덱스 fetch: 20~250배
- atomic: 25배
- 반복 읽기: 36배
- SSBO 쓰기: 1.8배

드라이버 할당 메모리의 성능은 연산에 따라 달랐습니다. N1X type 2/3과 device-local의 추가 측정은 [검수 결과](GPU_MEMORY_CACHE_REVIEW.md)에 있습니다.

HWAoI는 정점 미러로 16에서 60 fps가 됐습니다. Bayonetta 3은 import 메모리에 쓰는 깊이 패스 하나가 GPU 시간 40~57 ms를 차지합니다.

제안: NVIDIA에서는 모든 풀을 기존 "copied pool" 경로(드라이버가 할당한 사본)로 운용합니다.
- CPU 쓰기는 nvn_dirty 기반으로 4 KB 증분 업로드합니다.
- GPU 쓰기는 기존 범위 추적과 완료 스레드를 확장해 되돌리는 설계입니다. 쓰기 범위 누락, GPU 전송 완료, CPU 관찰 순서와 alias 일관성을 검증해야 합니다.

## 검수 중점

1. **사본 위치는 (A) device-local로 결정됐습니다**(NPARK, 메모리 용량은 제약 아님). 남은 질문은 이것입니다.
   - GPU 복사 업로드와 되돌림의 barrier 순서
   - `g_done`보다 앞서 복사가 끝나는지(R9)
   - import 원본을 복사 원본·대상으로 계속 쓰는 것이 맞는지
2. **동기화 정확성**: "CPU가 읽기 전 되돌림"을 `g_done`(copy-back 후 공개)과 `nvnSyncWait`/`nvnQueueFinish` → `mtl_wait`로 보장하는 구조(`gfx_vk_core.c:3488`, `hle_nvn.c:1078-1138`)에 빈틈이 있는지.
   - 펜스 없이 폴링하는 게임(R2)
   - 이벤트/카운터 쓰기
   - query 결과 복사
   - s56 렌더 타깃 착지
3. **업로드 누락 위험 R3**: 바인딩 범위 지연 업로드를 하려면 모든 바인딩 지점에 표시가 필요합니다. 빠진 경로가 없는지 봐 주세요.
   - 바인딩 지점: vertex/index, SSBO, UBO, texel, indirect, xfb, copy/fill, raw global BDA
   - 대안은 "dirty 페이지 전체 업로드"이고 더 단순합니다.
4. **되돌림 양 R1**: `gw[]`가 바인딩 범위 단위라 프레임당 수십 MB 비교가 생길 수 있습니다. 실제로 쓴 페이지만 되돌리는 방법이 필요한지.
5. **Eden 이식 범위**: BufferCache 클래스는 그대로 재사용하지 않고 메커니즘만 이식한다는 판단(2절 사실 3)이 맞는지. 이식 후보는 GPU-modified 범위 집합, 비동기 다운로드, 새로운 쓰기 유지 규칙입니다.
6. **정점 미러와 풀링**: 사본 모드에서 자동으로 꺼지는 구조가 맞는지, bridge와 s48 chunk(R4)를 어떻게 표현할지.
7. **fault 비용(R6, R8)**. 메모리 사용량은 검수 대상이 아닙니다.
   - 128 GB 기기라 NPARK가 제약이 아니라고 판단했습니다.
   - 예산 한도나 부분 사본 같은 절약 장치는 두지 않고, GPU가 쓰는 풀은 전부 사본을 만듭니다.
   - 참고값: 바요3 1,684 MB(실측), HWAoI 약 5 GB(기록).
8. **단계 구성(6절)과 검증 방법(7절)**이 충분한지.

## 참고 근거

- 측정: `membench2\results.txt`, `membench4\results.txt`, `membench5\results.txt`, `nvidia_report\README.md`
- Bayo3 패스 측정: 2026-11-20 14:29~14:35 실행 로그(`%APPDATA%\NXN1X\log\nxn1x_log.txt.old.txt`)의 `passes by GPU time`과 `s67` 줄
- 코드 위치: 설계서 2~3절의 파일:줄
