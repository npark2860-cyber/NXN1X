# 정점 속성 풀링(vertex attribute pulling) 설계 — N1X / NXN1X

작성: 2026-11-20 · 개정: 2026-11-20 검수 2건 반영(맨 끝 "검수 반영" 절이 본문보다 우선) · 상태: 설계안(코드 변경 없음) · 대상: Eden 경로(lane B) 1단계, NVN 경로 2단계

## 0. 한 줄 요약

정점 셰이더가 정점 속성을 고정 기능 정점 입력(VAF) 대신 **게임 풀의 import 버퍼에서 버퍼 디바이스 주소(BDA)로 직접 읽게** 한다. 인덱스는 지금의 device-local 미러(vbmir)를 그대로 쓴다(membench3 g안). 정점 데이터의 복사와 쓰기 보호, 그로 인한 폴트가 없어진다.

## 1. 근거

### 1.1 원인
N1X의 NVIDIA 드라이버(621.24)는 `VK_EXT_external_memory_host`로 import한 메모리를 **고정 기능 정점/인덱스 읽기**로 쓸 때 100~250배 느리다. 셰이더 읽기(SSBO, BDA, 텍셀 버퍼, UBO)와 복사는 같은 메모리에서도 정상 속도다. 상세: `N1X 메모리 문제 정리` 문서, `nvidia_report\README.md`.

### 1.2 membench3 (nxn1x\membench3, 같은 풀 드로우, GPU µs, 15회 중앙값)
정점 stride 24(float3, half2, unorm8x4, unorm 10-10-10-2) + 인스턴스 float4, 16비트 인덱스.

| 방식 | 39×1399 | 183×1399 |
|---|---:|---:|
| a 고정 기능, device-local (기준) | 13.9 | 38.9 |
| b 고정 기능, import (미러 이전 현재 방식) | 1,057.0 | 3,667.6 |
| c 속성+인덱스 풀링, SSBO, import | 16.8 | 42.0 |
| d 속성+인덱스 풀링, BDA, import | 16.7 | 40.7 |
| e 속성+인덱스 풀링, 텍셀 버퍼, import | 17.4 | 43.3 |
| f 속성+인덱스 풀링, SSBO, device-local | 14.9 | 40.1 |
| **g 속성 풀링(BDA 대신 SSBO), import + 인덱스는 device-local IB로 DrawIndexed** | **15.0** | **34.8** |

정확성: 정점 셰이더 출력(위치 + varying)을 인스턴스×정점별로 기록해 a와 비트 비교. b는 동일. 풀링 arm은 705,096개 중 57,359개가 **최대 1ulp** 차이(셰이더의 unorm 변환 `c/255`가 하드웨어 변환과 마지막 비트에서 다름, 3.5절에서 해결).

### 1.3 현재 미러(vbmir)의 한계
- 정점 데이터를 device-local로 복사하려고 CPU 쓰기 감지(페이지 쓰기 보호)를 쓴다. 폴트가 늘면서 fast-fail 0xd 크래시와 겹쳤다(별도 수정 tebfix 진행 중).
- 미러 메모리 1.3~1.7GB, 매 프레임 복사 수 MB, 동적 정점은 휘발성으로 매 제출 복사.
- 풀링은 이 모두를 정점 쪽에서 없앤다. 인덱스는 양이 작아(HWAoI 프레임당 2~3MB, 대부분 정적) 미러 유지 비용이 작다.

## 2. 현재 정점 입력 경로 (코드 위치)

### 2.1 Eden 경로(HWAoI가 쓰는 lane B)
1. 호스트 상태 → Eden 키: `host\dm\windows\gfx_vk_eden.inc:103-108`. 게임이 설정한 속성마다 `{location, stream, offset, vk_format}`, 스트림마다 `{stride, divisor}`를 `nvn_eden_draw_state`에 넣는다. 형식 변환은 `gfx_vk_exec.c:2161 vertex_format()`(Maxwell 형식 → VkFormat, r75 정수 입력 규칙 포함).
2. 파이프라인 키: `source\src\video_core\nvn\nvn_pipeline_key.cpp:490 FillNvnGraphicsKey`. 속성 형식/오프셋/버퍼(`:576`), divisor(`:571`), stride(`:591`, 단 EDS1에서는 키에서 빠지고 동적).
3. 셰이더 런타임 정보: `source\src\video_core\renderer_vulkan\vk_pipeline_cache.cpp:291-298`. 정점 단계에서 `info.generic_input_types[index]`를 키의 속성 타입으로 채운다(`runtime_info.h:95`).
4. SPIR-V 입력 정의: `source\src\shader_recompiler\backend\spirv\spirv_emit_context.cpp:2107-2153`(`DefineInputs`). generic마다 `in_attr{N}` Input 변수(타입은 `GetAttributeType :236`, 로드 방식은 `GetAttributeInfo`).
5. 속성 읽기: `source\src\shader_recompiler\backend\spirv\emit_spirv_context_get_set.cpp:309 EmitGetAttribute`. generic이면 `OpLoad` 후 Bitcast/SToF/UToF. 비활성 속성은 `(0,0,0,1)` 상수(`:317`).
6. 파이프라인 정점 입력 상태: `source\src\video_core\renderer_vulkan\vk_graphics_pipeline.cpp:823-876`(바인딩, divisor, 속성). `:1117-1124` 동적 정점 입력(VIDS)은 **N1X에서 꺼져 있다**(로그: `Disabling broken VK_EXT_vertex_input_dynamic_state`, `vertex_input_dynamic=false`). 따라서 속성 형식/오프셋은 파이프라인 정적 상태이고, stride는 EDS1 동적 상태(`s21` 로그: dyn 0x5b).
7. 정점 버퍼 바인딩: `gfx_vk_eden.inc:893-906`. 스트림마다 `vk_buf_at()`(풀 청크의 import 버퍼 + 오프셋), vbmir가 켜져 있으면 `vbm_bind()`가 미러로 바꾼다. 실제 바인딩은 `source\src\video_core\nvn\nvn_shader_bridge.cpp:1011 nvn_eden_cmd_vertex_buffers`(EDS1이면 `BindVertexBuffers2EXT`로 stride 전달).
8. 인덱스: `gfx_vk_exec.c draw_idx`(`vk_buf_at` → `vbm_bind` → `bc_index`).

### 2.2 NVN 경로(호스트 자체 번역기)
- 정점 입력: `gfx_vk_exec.c` prepare_draw의 `k.attr/k.bind`(`vertex_format`, `S.vattr`, `S.vstream`), 바인딩 루프(같은 파일 vertex buffers 루프).
- 셰이더: `gfx_vk_shader.c`(dmsh 출력). 2단계 대상.

## 3. 새 경로 설계 (Eden 경로)

### 3.1 개요
- 파이프라인에 **정점 입력 바인딩/속성을 넣지 않는다**(정점 입력 상태 비움).
- 정점 셰이더(VertexB)는 generic 속성을 **정점 페치 테이블**에서 읽는다.
- 정점 페치 테이블(드로우마다, 스크래치 버퍼): 스트림 16개 × `{u64 address, u32 stride, u32 size}` = 256바이트. `address`는 풀 청크 import 버퍼의 BDA + 오프셋(`vkGetBufferDeviceAddress`), `size`는 바인딩 범위(`S.vbuf[s].size`).
- 테이블 주소(u64)는 **푸시 상수**로 넘긴다. Eden의 푸시 상수 사용: `RescalingLayout`(textures 16B + images 8B + down_factor) 또는 `RenderAreaLayout`(16B) — `emit_spirv.h:23-33`. 끝에 8바이트 추가, 128바이트 한도 안.
- 속성마다의 형식, 오프셋, 스트림, divisor는 **이미 파이프라인 키에 있으므로** 셰이더 코드에 상수로 굽는다(`RuntimeInfo` 확장). **파이프라인 수 증가 없음**(풀링 on/off 한 비트만 키에 추가).

### 3.2 런타임 정보 확장
`runtime_info.h`의 `RuntimeInfo`에 정점 단계용으로 추가:
- `bool vertex_pulling`
- `std::array<VertexPullAttrib, 32>`: `{bool enabled; u8 stream; u16 offset; Maxwell::VertexAttribute::Size size; Type type; bool bgra; u32 divisor}` — 값은 `FixedPipelineState::attributes`/`binding_divisors`에서 채움(`vk_pipeline_cache.cpp:291` 자리).

### 3.3 셰이더 생성(SPIR-V)
- `DefineInputs`(`spirv_emit_context.cpp:2107`): `vertex_pulling`이고 VertexB면 generic Input 변수를 정의하지 않고, `PhysicalStorageBuffer` 포인터 타입(`uint` 배열, 4바이트 정렬)과 푸시 상수의 테이블 주소 멤버를 정의한다. 능력: `PhysicalStorageBufferAddresses`(디바이스는 이미 BDA 사용 중: `g_imp.bda`).
- `EmitGetAttribute`(`emit_spirv_context_get_set.cpp:309`): generic이면 `PullAttribute(index, element)`를 호출해 같은 정점의 같은 속성은 한 번만 디코딩하도록 캐시(셰이더 진입부에서 속성별 vec4를 미리 계산해 Private 변수에 저장하는 방식이 단순).
- 인덱스 계산:
  - 정점 속도: `vi = gl_VertexIndex`(Vulkan에서 indexed 드로우의 vertexOffset 포함, 비인덱스 드로우의 firstVertex 포함 → 지금 VAF와 같은 의미).
  - 인스턴스 속도(divisor d>0): `ii = gl_BaseInstance + (gl_InstanceIndex - gl_BaseInstance) / d`(`VK_EXT_vertex_attribute_divisor` 규칙과 동일). d=0이면서 인스턴스 스트림이 아닌 경우는 정점 속도.
  - 바이트 주소: `addr = table[stream].address + idx * table[stream].stride + offset`.
- 범위 밖: `idx * stride + offset + bytes > size`이면 해당 속성을 0으로 읽는다(지금 디바이스는 robustBufferAccess를 끈다: `gfx_vk_core.c` `en.robustBufferAccess = 0`. 현재 VAF의 범위 밖 결과는 정의되지 않음 → 0 반환은 더 안전하고, 비교 검증에서 차이가 나면 기록).
- 주소 0(스트림이 풀 밖, 현재 `vk_zero_buffer()`로 바인딩되는 경우): 0으로 읽음 → 이후 기본값 규칙.

### 3.4 형식 디코딩 (모든 Maxwell 정점 형식)
- 크기: 32_32_32_32, 32_32_32, 32_32, 32, 16_16_16_16, 16_16_16, 16_16, 16, 8_8_8_8, 8_8_8, 8_8, 8, 10_10_10_2, 11_11_10.
- 타입: UnsignedNorm, SignedNorm, UnsignedScaled, SignedScaled, UnsignedInt, SignedInt, Float(32: float, 16: half, 11_11_10: 부동소수 패킹).
- BGRA 스위즐 비트: x와 z 교환.
- 읽기: 바이트 주소를 4바이트 워드 2개(필요 시 3개)로 읽어 시프트로 추출(오프셋/stride가 4의 배수가 아닌 경우 포함). BDA 로드는 4바이트 정렬 `uint` 단위로만 한다.
- 빠진 성분: `(0, 0, 0, 1)`, 정수 타입은 `1`도 정수 1(Maxwell/Vulkan 동일).
- 정수 입력에 float 형식(r75 규칙, `gfx_vk_exec.c:2169-2176`): 변환된 float의 비트를 정수 입력으로 넘긴다. 현재와 같은 결과가 나오도록 그대로 재현.
- scaled: `support_scaled_attributes` 여부와 관계없이 셰이더에서 `float(int)` 변환(지금 `GetAttributeType :244-248`의 분기와 같은 결과).

### 3.5 비트 정확도 (membench3의 1ulp 차이 해결)
- 원인: unorm/snorm 변환을 `c / (2^b-1)`로 셰이더에서 하면 SPIR-V `OpFDiv`는 정확한 반올림을 보장하지 않는다(허용 오차 2.5ulp).
- 방법: 정확히 반올림된 나눗셈을 만든다. `q = float(c) * r` (r = 1/(2^b-1)를 float로), `e = fma(-q, D, float(c))`, `q = fma(e, r, q)`(D = 2^b-1). b ≤ 16이면 정수 c와 D가 float로 정확하므로 결과는 올바르게 반올림된 c/D가 된다. snorm은 `max(c / D, -1.0)`(D = 2^(b-1)-1).
- 하드웨어가 올바르게 반올림하지 않는 경우에 대비해 **전수 비교**로 확인한다: 8비트(256값), 10비트(1024값), 16비트(65536값) unorm/snorm, half 전체(65536), 11/10비트 float 전체를 고정 기능 입력과 풀링 입력으로 각각 정점 셰이더에 넣고 출력 비트를 비교하는 벤치(membench4). 차이가 있으면 하드웨어 결과 표(LUT)를 만들어 셰이더 상수 배열로 쓴다(8/10비트는 LUT가 작음).

### 3.6 드로우 시점(호스트)
`gfx_vk_eden.inc:893-906` 자리에서:
- 풀링 파이프라인이면 정점 버퍼를 바인딩하지 않는다. 대신 사용하는 스트림마다 `vk_buf_at()`으로 import 버퍼/오프셋을 얻고, `vkGetBufferDeviceAddress`(버퍼별 캐시: `vkpool_t`에 이미 `gpu` 필드가 있음, `gfx_vk.h` vkpool_t)로 주소를 만든다.
- 테이블을 `vk_scratch()`에 쓰고 그 BDA를 푸시 상수로 넘긴다(같은 내용이면 이전 테이블 재사용).
- vbmir는 이 드로우의 정점 스트림에 대해 호출하지 않는다(인덱스만).
- 풀 메모리가 아니면(주소 0) 테이블의 address=0.

### 3.7 다른 기능과의 관계
- 트랜스폼 피드백: 정점 셰이더 출력 기록이라 영향 없음(입력만 바뀜). s30b 경로 유지.
- 기하/테셀레이션: VertexB 입력만 바뀌므로 그대로.
- 쿼드: 호스트가 쿼드 인덱스로 변환(`quad_indices`) → `gl_VertexIndex` 의미가 같아 그대로.
- 프리미티브 재시작: 인덱스 경로 그대로(미러 + 고정 기능) → 영향 없음.
- 간접 드로우(인덱스): firstIndex/vertexOffset/firstInstance는 Vulkan이 `gl_VertexIndex`/`gl_InstanceIndex`/`gl_BaseInstance`에 반영 → 그대로. 단 간접 드로우는 인덱스 미러를 안 쓰고 import 인덱스를 읽는다(현재와 같음, 별도 과제).
- 정점 캐시: 고정 기능의 재사용은 유지된다(인덱스는 고정 기능이므로 post-transform 캐시 그대로). g안이 c안보다 빠른 이유.

### 3.8 폴백과 스위치
- 스위치: `NXN1X_VTX_PULL`(기본 1, `0`이면 끔, ctl.txt로 실시간 A/B — 파이프라인 키 비트가 바뀌므로 다음 드로우부터 새 파이프라인).
- 풀링을 쓰지 않는 경우(지금처럼 고정 기능 + vbmir):
  - 디바이스에 BDA가 없을 때
  - NVN 경로(1단계에서는)
  - 메시 셰이더 파이프라인(정점 입력 없음)
  - 정점 셰이더가 아닌 첫 단계(VertexA+B 결합 등 특수 케이스: 확인 필요)
- vbmir는 그대로 두되 풀링 드로우에서는 인덱스만 미러링한다. 결과적으로 미러 크기와 복사량이 줄어든다.

## 4. 검증 계획

1. **membench4(형식 전수)**: 3.5절. 모든 형식에서 고정 기능 대비 비트 동일(또는 LUT로 동일)해야 다음 단계.
2. **게임 내 비교 캡처**(진단, 환경변수로만): 지정한 정점 셰이더 키의 첫 N개 드로우에서 정점 셰이더 출력을 SSBO에 기록하는 변형을 두 번(고정 기능 파이프라인, 풀링 파이프라인) 그려 비교, 차이 개수/최대 ulp를 로그에 남김.
3. **장면**: HWAoI 풀밭, 전투, 마을, 컷신 영상(11:11 크래시 지점), 메뉴; 베요네타 3(스위치1, Eden 경로 확인), 동키콩(정점 많은 장면).
4. **성능 기준**: HWAoI 풀밭 GPU 시간 ≤ 현재 미러(p50 10ms), 화면 동일. `vbmir` 로그의 정점 미러 크기/복사량이 인덱스분만 남는지(프레임당 2~3MB 이하), 쓰기 보호 폴트 수 감소.
5. **안정성**: 크래시 링 invalid-SP 0건 유지, 10분 이상 플레이.

## 5. 위험

| 위험 | 대응 |
|---|---|
| 셰이더 변환 버그로 형식 하나가 틀림(화면 깨짐) | 형식 전수 벤치 + 게임 내 비교 캡처, 스위치로 즉시 A/B |
| 비트 차이(1ulp)로 깊이 같음 비교(EQUAL 깊이 테스트) 패스에서 깜빡임 | 3.5절 정확 반올림/LUT, 깊이 프리패스와 컬러 패스가 같은 셰이더 경로를 쓰는지 확인 |
| 푸시 상수 충돌(재스케일링/렌더 영역과 겹침) | 레이아웃 끝에 8바이트 추가, 파이프라인 레이아웃 생성부 확인 |
| 범위 밖 읽기 결과가 지금과 다름 | 0 반환으로 고정, 비교 캡처에서 차이 기록 |
| 셰이더 캐시/디스크 캐시 무효화 | 키 비트 추가로 자연 분리, 캐시 버전 올림 |
| 셰이더가 길어져 정점 단계가 느려짐 | 같은 정점 속성은 한 번만 디코딩, 벤치상 영향 작음(c/d/g) |

## 6. 단계

1. **0단계(완료)**: membench3로 성능/개념 검증.
2. **1단계**: membench4(형식 전수 비트 비교) → 변환식 확정.
3. **2단계**: Eden 경로 구현(RuntimeInfo 확장, DefineInputs/EmitGetAttribute, 파이프라인 정점 입력 비움, 호스트 테이블/푸시 상수), 스위치 기본 끔으로 빌드 → 게임 비교 캡처.
4. **3단계**: 기본 켬, vbmir는 인덱스 전용 동작 확인, 장면 테스트.
5. **4단계**: NVN 경로(gfx_vk_shader.c) 같은 방식으로.

## 7. 열린 질문

- Eden의 VertexA/VertexB 결합 셰이더(두 단계 정점 셰이더)에서 generic 입력이 어느 단계에 있는지.
- `support_scaled_attributes`가 N1X에서 참인지(로그 확인).
- 간접 인덱스 드로우의 인덱스도 미러로 옮길지(개수를 GPU가 정함: 별도 설계).
- 파이프라인 레이아웃의 푸시 상수 범위를 정점 단계에도 열어야 하는지(현재 단계 플래그 확인).

## 검수 반영 (2026-11-20, VERTEX_PULLING_REVIEW.md = R1, ASTRA_VERTEX_PULLING_REVIEW.md = R2)

이 절이 본문과 다르면 이 절을 따른다. 각 항목은 실제 코드로 확인한 결과다.

### 판정 요약
두 검수 모두 "방향 타당, 현 설계 그대로 기본 적용은 불가"다. 지적은 대부분 받아들인다. 거부한 항목은 없고, 일부는 코드로 사실을 고쳐 반영했다.

### 받아들인 항목과 바뀐 설계

| ID | 내용 | 확인 | 바뀐 설계 |
|---|---|---|---|
| R1-P1 주소 / R2-F02 | 1바이트 조회 폴백(`gfx_vk_eden.inc:899` `vk_buf_at(addr, 1, ...)`)으로는 뒤쪽 바이트의 BDA 연속성이 보장되지 않음. 16바이트 형식은 정렬 4워드, 비정렬 최대 5워드 | 맞음 | 테이블에는 **실제 VkBuffer와 오프셋으로 증명한 연속 범위**만 싣는다(`vk_buf_at(addr, size)` 전체 성공 시만). 실패하거나 size=0이면 그 드로우 전체를 기존 경로(FF + vbmir)로 보낸다. 워드 수는 `((addr&3)+bytes+3)/4`, 검사는 64비트 산술, `rel <= size && bytes <= size - rel`, 정렬 워드 범위 `[wordStart, wordStart+4*wordCount)`까지. **검사 후 분기해서 로드**(select로 가리지 않음). 음수 vertexOffset 포함 |
| R1-P1 입력 / R2-F05 | 내장 변수 선언이 게스트 IR 사용 여부에 묶임(`spirv_emit_context.cpp:2062` 부근), 동적 속성 번호는 `EmitGetAttributeIndexed`(`emit_spirv_context_get_set.cpp:447`), BDA 초기화가 global memory 사용에 묶임(`:1027`) | 맞음 | 풀링 VS는 VertexIndex·InstanceIndex·BaseInstance와 PhysicalStorageBuffer64를 **무조건** 선언. `loads_indexed_attributes`가 있는 셰이더는 1단계에서 폴백 |
| R1-P1 동기화 / R2-F01 | 소비 단계가 VERTEX_INPUT/VERTEX_ATTRIBUTE_READ에서 VERTEX_SHADER/SHADER_STORAGE_READ로 바뀜 | 맞음 | 같은 제출에서 GPU가 쓴 범위(`vk_sub_gw_overlaps`)를 읽는 풀링 드로우 앞에는 생산자 → VERTEX_SHADER/SHADER_READ 배리어를 넣는다(실제 겹침이 있을 때만). 인덱스는 기존 INDEX_INPUT/INDEX_READ 유지. import 버퍼 사용 기록·참조 유지(기존 `vk_buf_at` 경로의 사용 표시)는 바인딩을 생략해도 그대로 남긴다 |
| R2-F03 | 테이블 수명 | 맞음 | 테이블은 `vk_scratch()`(BDA 사용 비트 지원, `gfx_vk_core.c:2707`)의 제출별 메모리에 쓰고, 제출 펜스 완료로 회수되는 기존 수명을 따른다. 같은 제출 안에서 내용이 바뀌면 새 공간, 명령 버퍼 분할 뒤에는 다시 푸시 |
| R1-P1 비교 / R2-F04 | 게임 드로우 두 번 실행은 부작용과 경합 | 맞음 | 4.2절 "게임 내 이중 그리기 비교"를 **폐기**. 대신 (a) 순수 디코딩 셰이더로 입력 스냅샷 비교, (b) 부작용 없는 VS만 격리 재생, (c) 기록은 invocation별 독립 슬롯/아토믹 claim, (d) 정확도용과 성능용 빌드 분리 |
| R1-P1 벤치 / R2-F11 | g안은 SSBO + 고정 IB이고 설계는 BDA + 테이블 + 보정 디코더 + 고정 IB. 인스턴스 버퍼가 device-local이고 값이 모두 1.0 | 맞음 | 1단계 벤치(membench4)에 **production과 같은 조합**(BDA 테이블, 런타임 stride, import 인스턴스 스트림, 위치별로 다른 인스턴스 값) arm을 추가 |
| R1 푸시 상수 | Eden 그래픽 푸시 상수 범위는 `pipeline_helper.h:249-258`에서 ALL_GRAPHICS, 크기 48바이트(RescalingLayout 32 + RenderAreaLayout 16) | 맞음(코드 확인) | 테이블 주소(u64)는 **offset 48, 범위 56바이트**. stageFlags는 기존 ALL_GRAPHICS 그대로, 푸시도 같은 마스크. 3.1절의 "끝에 8바이트"를 이것으로 고정. (R2의 offset 32 제안은 문서 수치만 본 것이라 코드 기준인 R1을 따름) |
| R2-F06 / R1 형식 | 기준은 raw Maxwell 형식이 아니라 Eden이 최종 선택한 VkFormat과 IR 로드 방식 | 맞음 | 디코더는 `vertex_format(fmt,0,0)`(`gfx_vk_exec.c:2161`)이 준 **VkFormat**과 `GetAttributeInfo`의 Bitcast/SToF/UToF를 재현한다. Eden 경로는 r75 정수 재해석을 쓰지 않으므로(want_int=0) 3.4절의 r75 문장은 Eden 1단계에서 삭제. BGRA는 VkFormat에 이미 반영된 것만, 한 번만. 지원 형식은 `vertex_format()`의 유효 매핑 allowlist |
| R1·R2-F07 | divisor=0 의미 | 맞음(코드 확인: `nvn_pipeline_key.cpp:570-572`, 인스턴싱 꺼짐이면 divisor 0 = 정점 속도) | inputRate는 "인스턴싱 켜짐 여부"로 보존. 인스턴스 속도이면서 divisor 0인 상태는 현재 매핑에서 생성되지 않음(위 코드 근거) — 생기면 폴백 |
| R2-F08 / R1 수치 | 보정식은 CPU 전수에서 0 불일치(두 검수 모두 확인). GPU의 GLSL fma가 fused인지, 반올림 모드는 보장 안 됨 | 맞음 | 이 기기는 `VK_KHR_shader_fma`(shaderFmaFloat32=true)와 `shaderRoundingModeRTEFloat32=true`를 지원(vulkaninfo 확인). 디코더는 **OpFmaKHR + RTE 실행 모드**를 쓰고, 정점 셰이더의 다른 연산에 미치는 영향을 막기 위해 RTE는 디코더 함수에만 적용 가능한지 확인, 안 되면 비교 결과로 판단. 검증은 세 기준값(정확 유리수 CPU, N1X VAF 직후, 풀링 디코더 직후) 분리 기록, 개별 성분 단위. membench3의 57,359개 차이는 합성된 출력이라 원인 분리 불가 — 재측정 |
| R2-F09 / R1 키 | "파이프라인 수 증가 없음"은 단일 모드에서만. 셰이더 캐시 키·디스크 레시피·빠른 경로(`gfx_vk_eden.inc:218`)도 모드를 구분해야 | 맞음 | 풀링 비트를 replay key, Eden GraphicsPipelineCacheKey, 셰이더 재사용 키, 디스크 레시피에 모두 반영. 실시간 전환은 상태 버전을 올려 빠른 경로를 무효화. 형식을 셰이더에 굽기 때문에 셰이더 모듈 수는 늘 수 있음(측정 항목) |
| R2-F10 | zero-buffer/주소 0은 "(0,0,0,1)"이 아니라 0 원시 데이터를 형식대로 디코딩 | 맞음 | 3.3절 수정: 의도적 zero-buffer는 원시 0을 디코딩 후 빠진 성분만 기본값. 해석 못 하는 유효 스트림은 드로우 전체 폴백. 범위 밖 0은 별도 정책(FF와 동등성 주장 안 함) |
| R2-F10 미러 전환 | 풀링 중 vbm_bind를 안 불러도 미러/보호는 남음, OFF 복귀 시 미러가 오래됐을 수 있음 | 맞음 | vbmir는 GPU/CPU 쓰기 감지를 계속 하므로(`vk_vbm_gpu_wrote`, nvn_dirty) 오래된 블록은 다음 사용 때 다시 복사된다. 다만 보호를 건 적 없는 페이지는 valid=0이라 첫 사용 때 복사. 전환은 제출 경계에서만 적용 |
| R1 VertexA+B | 결합 결과도 Stage::VertexB(`translate_program.cpp:485`), NVN 브리지는 VertexB부터 구성(`nvn_shader_bridge.cpp:529`) | 맞음 | stage로 구분하지 않고 **결합 여부 플래그**로 제외 |
| R1 scaled | `support_scaled_attributes`는 프로필상 true(`vk_pipeline_cache.cpp:641`) | 맞음 | 열린 질문에서 삭제. scaled는 float 입력 경로로 디코딩 |
| R1 robustness | 실제 VkDevice는 Eden/NXN1X가 인계한 것(`nvn_bridge_device.cpp:49`) | 맞음 | 실제 enabled feature를 로그로 기록하고 그 기준으로 OOB 정책 결정 |
| R1 미러 비용 | 미러는 청크 전체 크기라 IB만 남아도 할당이 유지될 수 있음 | 맞음 | 측정을 정점 복사/인덱스 복사/보호 페이지·폴트/청크 할당으로 나눠 기록 |
| R2-F12 문구 | 효과·기본값 문구 과장, g/a는 1.08배(39×1399)와 0.90배(183×1399), b/a는 76배·94배 | 맞음 | 스위치 기본값은 **검증 단계 0, 통과한 범위만 1**. "항상 빠름", "캐시 때문" 문구 삭제. 100~250배는 membench(1차) 조건, 76~94배는 membench3 조건으로 분리 표기 |
| R2 tebfix | 풀링으로 폴트가 줄어 강종이 안 나는 것을 NCE 수정의 증거로 쓰지 않음 | 맞음 | tebfix 판정은 크래시 링의 invalid-SP 0건으로만 |

### 바뀐 단계 (R2 12절 채택)
1. **A: production 디코더 벤치(membench4)** — BDA + 테이블 + 보정 디코더(OpFmaKHR/RTE) + 고정 device-local IB, 이미 측정한 5개 VkFormat부터. 세 기준값 비교, 성능은 계측 없는 빌드로.
2. **B: 호스트 통합(기본 0)** — 일반 Eden VertexB, 직접 indexed draw, 한 청크 안에 증명된 범위, allowlist 형식만. 나머지 폴백. spirv-val, validation/동기화 레이어, 분할 명령 버퍼, 같은 제출의 GPU write → VS read 포함.
3. **C: 게임 비교** — 입력 스냅샷 + 격리 VS 비교, 이어서 동일 저장 지점 A/B(GPU p50/p95, 프레임 p50/p95, 복사량, 폴트, 폴백 횟수와 이유).
4. **D: 확대** — 형식, 비인덱스/간접, GS/TES/XFB, VertexA+B, NVN 경로.

### 결정이 필요한 것(NPARK)
- 단계 A(벤치)부터 시작해도 되는지.
- 실시간 A/B 전환은 제출 경계 적용으로 충분한지(프레임 단위가 더 안전).

## A단계 결과 (2026-11-20, membench4\results.txt)

- 형식 전수 비교(CPU 기준 / N1X 고정 기능 / 풀링 디코더, 성분별 비트): R8G8B8A8 UNORM·SNORM, R16G16B16A16 UNORM·SNORM, A2B10G10R10 UNORM, R32G32B32A32 SFLOAT 모두 **0 불일치**. 고정 기능은 모든 형식에서 CPU 기준과 일치.
- 남은 차이는 NaN payload뿐: half 2,046개, B10G11R11 184개. 하드웨어는 payload를 보존하고 디코더(unpackHalf2x16, 표준 NaN)는 바꿔서 생김 → B단계 디코더는 비트 연산으로 payload를 보존하는 변환을 쓴다.
- 정규화는 GLSL fma(GLSL.std.450 Fma) + precise로 이 GPU에서 전수 일치. OpFmaKHR/RTE 실행 모드 없이도 일치했지만, B단계에서 SPIR-V를 다시 검증한다.
- 풀 드로우(GPU µs, 9회 교차 중앙값 / 최소, 39×1399 · 183×1399):

| 방식 | 39×1399 | 183×1399 |
|---|---|---|
| 고정 기능, device-local (현재 미러) | 12.1 / 5.0 | 29.7 / 27.8 |
| 고정 기능, import | 202.8 / 193.2 | 3,633.5 / 3,456.2 |
| 풀링(BDA 테이블, 런타임 stride, import 인스턴스 스트림) + device-local IB | 13.8 / 13.5 | 29.1 / 27.9 |

결론: production 조합이 현재 미러와 같은 속도이고, NaN을 뺀 모든 값이 하드웨어와 비트 단위로 같다. B단계로 진행할 근거가 된다.
