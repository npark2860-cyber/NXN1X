# 검수 요청: 버텍스 풀링 B단계 (NXN1X_VTX_PULL)

기준 문서: `design\VERTEX_PULLING_DESIGN.md`. 그 문서의 "검수 반영" 절이 우선합니다.
원본 백업: `_backup\vtxpull\`. 경로는 원래 위치와 같고, 새 파일 `vk_vertex_pulling.h`만 백업이 없습니다.
emit_spirv.h 백업은 수정 전 파일을 복사해 둔 것이 아니라, 지금 파일에서 vtxpull 4줄을 지워 다시 만든 것입니다.

## 무엇을 하는가

N1X에서는 import 메모리(VK_EXT_external_memory_host)에서 고정 기능 정점 fetch가 약 100~250배 느립니다. 셰이더 load는 정상 속도입니다.

B단계는 Eden 경로에서만 동작합니다.
- 조건을 만족하는 direct draw는 VS가 fetch table을 읽습니다. fetch table은 스트림별 BDA, stride, size 항목입니다.
- VS는 이 table로 zero-copy 풀의 정점을 직접 읽습니다.
- 그 외의 draw는 기존 vbmir 미러 경로를 그대로 탑니다.
- 인덱스는 계속 미러에서 읽습니다.
- 기본값은 OFF입니다. `NXN1X_VTX_PULL=1`(ctl.txt의 `env` 줄)을 주면 다음 present부터 적용됩니다.

## 변경 파일과 내용

| 파일 | 내용 |
|---|---|
| `shader_recompiler/runtime_info.h` | `VertexPullKind`, `VertexPullAttrib`, `RuntimeInfo::{vertex_pulling, vertex_pull_attribs[32], vertex_pull_divisor[32]}` |
| `backend/spirv/emit_spirv.h` | `VERTEX_PULL_TABLE_OFFSET` = 48 (RescalingLayout 32 + RenderAreaLayout 16), `VERTEX_PULL_PUSH_SIZE` = 56 |
| `backend/spirv/spirv_emit_context.{h,cpp}` | VertexB에서 pulling일 때:<br>- Int64, PhysicalStorageBuffer64 capability, 타입 정의<br>- table 주소를 기존 push constant 블록(rescaling 또는 render area)에 offset 48 멤버로 붙임. 둘 다 없으면 별도 블록(엔트리포인트당 블록 1개)<br>- VertexIndex, InstanceIndex, BaseInstance 입력 정의<br>- pull하는 attribute는 Input 변수를 만들지 않음 |
| `backend/spirv/emit_spirv_context_get_set.cpp` | `EmitPulledAttribute`:<br>- index 계산: VertexIndex, 또는 base + (inst − base) / divisor<br>- 64비트 rel = idx·stride + offset<br>- **load 전에** `rel + bytes ≤ size`를 검사하고, 범위 밖이면 rel 0을 읽은 뒤 결과를 0으로 select<br>- unorm/snorm은 exact FMA 식, half와 ufloat는 비트 연산(NaN payload 보존), 10-10-10-2와 11-11-10<br>- `EmitGetAttribute`는 enabled attrib이면 이 함수를 먼저 탐 |
| `renderer_vulkan/fixed_pipeline_state.h` | `raw1` bit 22 `vertex_pulling`. 키와 디스크 캐시 레시피에 들어감 |
| `renderer_vulkan/vk_vertex_pulling.h` (새 파일) | `DescribeVertexPullFormat`: VkFormat을 디코더 레이아웃으로 변환(BGRA, 64비트 등은 거부)<br>`FillVertexPulling`: 키 비트, `!dynamic_vertex_input`, `!loads_indexed_attributes`, BDA와 Int64 지원, 모든 attribute 포맷이 서술 가능한지 확인 |
| `renderer_vulkan/vk_pipeline_cache.cpp` | VertexB의 `MakeRuntimeInfo`에서 `FillVertexPulling` 호출 |
| `renderer_vulkan/vk_graphics_pipeline.{h,cpp}` | 생성자에서 같은 헬퍼로 `pulls_vertices`를 정함(mesh 파이프라인 제외). pulling이면 vertex input state를 비움. `PullsVertices()` 추가 |
| `renderer_vulkan/pipeline_helper.h` | 모든 graphics 레이아웃의 push range를 48에서 56으로 늘림(레이아웃 사이 push constant 호환 유지). compute는 그대로 |
| `nvn/nvn_shader_bridge.{h,cpp}` | `nvn_eden_draw_state::vertex_pull`을 키에 Assign<br>`nvn_eden_pipeline_info::vertex_pull` = `PullsVertices()`<br>`nvn_eden_cmd_vertex_table`: ALL_GRAPHICS, offset 48, 8바이트 push |
| `host/dm/windows/gfx_vk.h` | `vkshader_t::{vpull_inputs, vpull_known}`: VS가 load하는 generic 입력(첫 Eden 파이프라인에서 학습) |
| `host/dm/windows/gfx_vk_exec.c` | `g_vpull_cfg`(x_read, `NXN1X_VTX_PULL`). `g_vpull_direct`: draw와 draw_idx에서 `!ind && !mdi_cnt && prim != 7`, mesh는 0 |
| `host/dm/windows/gfx_vk_eden.inc` | `vpull_decide`(eden_graphics 첫머리), present 단위 적용과 약 600 present(약 10초)마다 요약 로그, s45_ctx의 `vpull`, EKF, eden_draw에서 table 작성과 push. vertex buffer 바인드와 vbm은 건너뜀 |

## 판정 규칙 (호스트, draw마다)

pull하는 조건(모두 만족해야 함):
1. 스위치가 이번 present에 ON이고, direct draw이며 mesh가 아님.
2. VS 입력을 이미 알고 있음. 그 VS의 첫 파이프라인 draw는 fallback.
3. VS가 load하는 위치의 attribute 스트림마다:
   - address와 size가 있고, size ≤ 4 GB
   - `vk_pool_of`로 찾은 chunk가 `zero_copy`이고 `!bridge`이며 `gpu`(BDA)가 있음
   - `[a, a+n)`이 chunk 안에 통째로 들어감
   - `vk_sub_gw_overlaps(a, n) == 0`: 같은 submit 안에서 GPU가 먼저 쓴 범위가 아님

그 외에는 기존 경로를 탑니다. 키에 비트가 들어가므로 pull 파이프라인이 따로 생깁니다. 파이프라인이 거부하면(포맷 등) `pi.vertex_pull == 0`이 되고, 그 draw는 vertex buffer로 그립니다(PIPE fallback으로 집계).

## 위험 지점 (중점 검수 요청)

1. **이전 submit에서 GPU가 쓴 정점의 가시성.**
   - 같은 submit 안에서 쓴 범위는 fallback합니다.
   - 이전 submit의 쓰기는 기존 import 직독 경로와 SSBO 경로처럼 submit 경계의 동기화에 기댑니다.
   - VS 단계의 SHADER_READ 가시성이 보장되는지 확인해 주세요. 대상은 submit 시작부의 전역 barrier(`vk_img_barrier_all`)와 `vk_dev_access`입니다.
2. **Push constant.**
   - graphics 레이아웃 크기를 48에서 56으로 바꿨습니다.
   - VS의 블록 하나에 rescaling과 fetch_table 멤버가 함께 들어가거나, render area와 fetch_table이 함께 들어갑니다.
   - 멤버 offset은 증가 순서입니다.
   - s45가 `nvn_eden_cmd_state`를 생략하는 경우에도 table은 draw마다 다시 push합니다.
3. **Unaligned 16비트 성분이 워드 경계를 넘을 때.**
   - `PullBits`가 다음 워드를 읽습니다. 버퍼 끝이 chunk 끝과 정확히 같으면 4바이트 밖을 읽을 수 있습니다.
   - Vulkan 포맷 정렬 규칙상 정상 입력에서는 이런 경우가 생기지 않는다고 판단했습니다.
4. **Divisor 의미.**
   - `binding_divisors[s] == 0`은 per-vertex, 그 외는 per-instance입니다. `MakePipeline`과 같습니다.
   - `enabled_divisors`는 보지 않습니다.
5. **`vpull_decide`가 한 draw에 여러 번 호출됨.**
   - 호출 경로는 prepare_draw → s64 → draw_uses_only_uniforms → eden_graphics입니다.
   - 판정은 상태의 순수 함수이고, table은 마지막(eden_draw의) 호출 값을 씁니다.
6. **Fetch table 메모리.**
   - `g_eden_uring`(host-coherent, BDA)에서 draw마다 256바이트를 할당합니다.
   - 실패하면 그 draw를 건너뛰고 RED 로그를 한 번 남깁니다. 이 경우는 fallback할 수 없습니다(파이프라인이 이미 pull로 정해짐).
7. **디스크 캐시.**
   - bit 22가 0인 기존 레시피는 영향이 없습니다.
   - bit 22가 1인 레시피는 지원하지 않는 장치에서 `FillVertexPulling == false`가 되어 일반 파이프라인으로 만들어집니다.
8. **NaN payload.** half와 B10G11R11의 NaN payload는 고정 기능 결과와 비트가 다를 수 있습니다(A단계 결과: half 2,046건, B10G11R11 184건 불일치). 비트 연산으로 payload를 보존하지만, 드라이버 쪽 동작이 정규화합니다.

## 로그

- `vtxpull: vertex pulling requested on/off (NXN1X_VTX_PULL, live: applied at the next present)`
- `vtxpull: vertex pulling on/off from present N`
- `vtxpull: per present over N presents: X draws pulled, fallbacks to vertex buffers; <사유> <회수/present> ...`
- 미러 복사 MB/present는 기존 `vbmir:` 줄에 나옵니다.
