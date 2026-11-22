# 검수 요청: 정점 속성 풀링 설계 (NXN1X, N1X)

요청일: 2026-11-20 · 검수 대상: `design\VERTEX_PULLING_DESIGN.md` · 코드 변경은 아직 없음

## 배경 한 단락
N1X(NVIDIA RTX Spark N1X, 드라이버 621.24, Windows on ARM)에서 `VK_EXT_external_memory_host`로 import한 메모리를 고정 기능 정점/인덱스 입력으로 읽으면 100~250배 느리다. 셰이더 읽기는 같은 메모리에서도 정상 속도다. 지금은 정점/인덱스를 device-local 미러로 복사해서(vbmir) HWAoI가 16 → 60fps가 됐지만, 미러는 CPU 쓰기 감지를 위해 페이지 쓰기 보호를 쓰고 폴트가 늘었다. 설계는 정점 속성을 셰이더가 import 메모리에서 직접 읽고(풀링), 인덱스만 미러에 두는 것이다.

## 측정 (nxn1x\membench3\results.txt, GPU µs, 39×1399 / 183×1399)
| 방식 | µs |
|---|---|
| 고정 기능, device-local (기준) | 13.9 / 38.9 |
| 고정 기능, import | 1,057.0 / 3,667.6 |
| 속성+인덱스 풀링(SSBO / BDA / 텍셀), import | 16.8 / 42.0 · 16.7 / 40.7 · 17.4 / 43.3 |
| 속성 풀링(import) + 인덱스는 device-local IB (선택안) | 15.0 / 34.8 |
풀링 결과는 고정 기능 대비 출력 값의 약 8%가 최대 1ulp 다름(unorm 변환식).

## 읽을 파일
1. 설계: `design\VERTEX_PULLING_DESIGN.md`
2. 벤치 코드: `membench3\membench3.c`, `membench3\ff.vert`, `membench3\pull.vert`
3. 현재 경로(설계 2절의 file:line):
   - `host\dm\windows\gfx_vk_eden.inc` 103-108, 893-906
   - `host\dm\windows\gfx_vk_exec.c` 2161-2190(vertex_format), vbm_bind / draw_idx
   - `source\src\video_core\nvn\nvn_pipeline_key.cpp` 490-591
   - `source\src\video_core\renderer_vulkan\vk_pipeline_cache.cpp` 291-298
   - `source\src\video_core\renderer_vulkan\vk_graphics_pipeline.cpp` 823-876, 1117-1124
   - `source\src\shader_recompiler\backend\spirv\spirv_emit_context.cpp` 2107-2153
   - `source\src\shader_recompiler\backend\spirv\emit_spirv_context_get_set.cpp` 309-344
   - `source\src\shader_recompiler\backend\spirv\emit_spirv.h` 18-33(푸시 상수 레이아웃)
   - `source\src\video_core\nvn\nvn_shader_bridge.cpp` 1011-1034

## 확인해 주실 것
1. **의미 동등성**: 풀링 결과가 지금의 고정 기능 입력과 같은가.
   - 인스턴스 divisor 공식(`gl_BaseInstance + (gl_InstanceIndex - gl_BaseInstance) / d`)이 Maxwell/NVN의 인스턴스 주파수 의미와 맞는가.
   - `gl_VertexIndex`가 indexed/비인덱스/간접 드로우에서 VAF가 쓰는 정점 번호와 같은가(vertexOffset, firstVertex).
   - 빠진 성분 기본값, BGRA, 11_11_10, 10_10_10_2, scaled, 정수 입력에 float 형식(r75 규칙).
2. **비트 정확도**: 3.5절의 정확 반올림 나눗셈(fma 보정)이 모든 8/10/16비트 unorm/snorm에서 올바르게 반올림된 결과를 주는가. 하드웨어가 올바른 반올림이 아닐 가능성과 전수 비교 계획이 충분한가.
3. **범위 밖 읽기**: robustBufferAccess가 꺼진 상태에서 0 반환으로 고정하는 게 맞는가.
4. **호스트 쪽**: 드로우마다 페치 테이블(스크래치)과 푸시 상수 8바이트로 넘기는 방식의 수명/동기화 문제(스크래치 재사용, 같은 제출 내 여러 드로우).
5. **파이프라인 키**: VIDS가 꺼져 있어 속성 형식이 이미 키에 있으므로 파이프라인 수가 늘지 않는다는 판단이 맞는가. EDS1의 동적 stride를 테이블로 넘기는 것.
6. **빠뜨린 경우**: VertexA+B 결합, 메시 셰이더, 트랜스폼 피드백, 쿼드, 프리미티브 재시작, 지오메트리/테셀레이션, 스트림이 풀 밖인 경우.
7. **미러와의 공존**: 풀링 드로우에서 정점은 미러를 안 쓰고 인덱스만 미러를 쓰는 구성에서 동기화 누락이 없는가.
8. **검증 계획과 단계**가 충분한가, 더 작은 첫 단계가 있는가.

## 열린 질문(설계 7절)
- VertexA/VertexB 결합 셰이더에서 generic 입력의 위치
- `support_scaled_attributes` 값
- 간접 인덱스 드로우의 인덱스 미러
- 푸시 상수 범위의 단계 플래그

검수 수준: 최대(이전 검수와 같은 수준).
