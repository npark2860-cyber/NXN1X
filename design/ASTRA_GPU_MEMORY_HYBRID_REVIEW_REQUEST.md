# 검수 요청: GPU 메모리 혼합(backing 범위별 import / copy)

2026-11-20. 대상: [GPU_MEMORY_HYBRID_DESIGN.md](GPU_MEMORY_HYBRID_DESIGN.md). 짧게 봐 주시면 됩니다.

## 배경 (한 단락)

NXN1X는 Switch 게임 메모리 pool을 Vulkan에서 두 방식으로 씁니다.
- import: VK_EXT_external_memory_host로 게임 페이지를 직접 바인딩합니다.
- copy: device-local 사본에 제출마다 CPU 변경을 올리고, GPU 변경은 pre/post 사본을 비교해 병합합니다(stage2c).

N1X 드라이버에서는 import 메모리의 고정 기능 vertex/index fetch(100-250배), atomic(25배), 반복 읽기(36배)가 느립니다. 그래서 copy가 GPU 쪽을 빠르게 합니다(바요3 depth pass 40-57 ms → 0.6 ms). 하지만 copy에는 CPU 쓰기 추적 비용이 있습니다. HWAoI는 이동 중 fault가 20-35k/s, mprotect가 30-50k/s입니다. 그래서 GPU가 느리게 쓰는 범위만 copy로 바꾸려 합니다.

## 봐 주셨으면 하는 점

1. **단위**: pool이 아니라 backing 범위(256 MB 정렬 공유 사본, 또는 private 청크)로 정한 것이 맞는가? alias가 사본을 공유하므로 pool 단위로는 한 페이지가 두 모드에 동시에 보일 수 있다고 봤습니다.
2. **전환**: 첫 "느린 쓰기" 바인딩을 본 다음 제출 시작에 전체 drain → 사본 생성 → 전체 동기화 → 보호로 바꿉니다(4절).
   - 그 사이 바이트가 두 모드에 동시에 보이는 순간이 있는가?
   - 특히 이미 기록된 descriptor나 캐시된 바인딩에 import 버퍼가 남는 경우를 걱정하고 있습니다.
3. **단방향**: 그 실행 동안 copy로 고정합니다. 되돌림을 넣어야 할 사례가 있는가?
4. **분류 기준**:
   - copy로 바꾸는 것: shader store, atomic, (측정 결과에 따라) attachment 쓰기
   - import에 남기는 것: transfer dst, 읽기 전용
   - 빠진 GPU 쓰기 종류가 있는가? 예: transform feedback, query 결과 복사, indirect 인수 쓰기
5. **전제 검증 방법**: 1.1절의 카운터(범위별 GPU 쓰기 종류, fault의 범위 분류)와 판단 기준(fault의 60% 이상이 GPU 쓰기 없음/transfer만 범위)이 타당한가?
6. **드라이버 재확인**: membench 재측정 결과로 드라이버 버전 표를 갱신하고, 빠른 드라이버에서는 copy와 vbmir를 자동으로 끄는 방식(5절)에 문제가 없는가?
7. **테스트 H1-H11**에 빠진 경계가 있는가?
8. **raw global memory(3.1절)**:
   - OoT에서 raw 바인딩 때문에 copy 경로가 모든 청크를 통째로 pre/post 복사해(120번 제출당 723 MB) 3D가 4 fps였습니다.
   - 권장안: raw 바인딩만으로는 바꾸지 않습니다(import 유지). 전역 atomic을 쓰는 raw 셰이더의 write log에 나온 범위만 copy로 바꿉니다.
   - raw atomic을 셰이더 정보로 판별하는 것이 충분한가? raw store의 import 손해(약 1.8배)를 받아들이는 것이 맞는가?

## 현재 상태

- 전제(1.1절)는 **미검증**입니다. 일부 카운터(fault난 페이지의 GPU 기록 여부)는 20:35 빌드에 있고, 아직 실행하지 않았습니다.
- 2절의 attachment와 transfer dst 성능은 **미측정**입니다.
- 코드는 아직 없습니다.
