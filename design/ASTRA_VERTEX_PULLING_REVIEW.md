# NXN1X 정점 속성 풀링 설계 검수

검수일: 2026-11-20 KST  
대상: `VERTEX_PULLING_DESIGN.md`, `ASTRA_REVIEW_REQUEST.md`  
판정: **설계 방향은 타당. 현재 안은 구현 전 필수 보완이 필요하며, 기본 활성화 단계는 아님.**

## 1. 검수 범위와 핵심 판단

첨부된 두 문서 전체와 Khronos 공식 규격을 대조했다. 정규화 보정식은 별도 호스트 프로그램으로 전수 계산했다. 원문에 열거된 프로젝트 소스, `membench3.c`, `ff.vert`, `pull.vert`, 원시 GPU 결과는 첨부되지 않아 직접 검토하지 못했다. 따라서 아래의 “누락”은 설계 문서의 누락을 의미하며, 기존 코드에 해당 처리가 없다고 단정하지 않는다.

**유지할 방향:** 정점 속성은 셰이더에서 읽고 인덱스는 고정 기능·device-local 미러를 사용하는 혼합 방식. 기존 미러 경로를 폴백으로 유지한다.

**먼저 수정할 부분:** 실제 BDA 접근 범위, 소비 단계 변경에 따른 동기화, scratch 테이블 수명, 형식 변환 계약, 게임 드로우를 두 번 실행하는 비교 방법.

**긍정적으로 확인된 부분:** `VertexIndex` 및 divisor>0의 Vulkan 주소 계산 방향은 맞고, 제안한 정규화 보정식은 RN 연산을 전제로 한 호스트 전수 검사에서 262,138건 모두 기준값과 일치했다.

### 우선순위 정의

- **P0:** 잘못된 GPU 메모리 접근, 수명·동기화 오류, 검증 자체의 상태 오염으로 이어지는 설계 조건. 게임 경로 구현에 함께 반영해야 한다.
- **P1:** 의미 동등성·캐시·성능 판단을 확정하기 위해 필요한 조건. 해당 조합을 지원하기 전에 해결하거나 명시적으로 폴백한다.
- **P2:** 범위를 과장하거나 서로 충돌하는 문구. 개발 완료 기준을 잘못 잡지 않도록 수정한다.

## 2. 핵심 지적 목록

| ID | 우선순위 | 대상 절 | 지적 | 조치 |
|---|---|---|---|---|
| F01 | P0 | 3.3, 3.6, 3.7 | VAF 읽기에서 VS의 BDA 읽기로 바뀌는 동기화 계약이 없음 | VS shader-storage read와 기존 IB read를 각각 등록 |
| F02 | P0 | 3.1, 3.3, 3.4 | 논리 범위 검사만 있고 실제 u32 load·청크 경계·산술 폭 보장이 없음 | 실제 VkBuffer 범위, overflow, 정렬 워드 범위를 함께 검사 |
| F03 | P0 | 3.1, 3.6 | scratch 내용의 불변 기간과 BDA 자원 수명이 없음 | GPU 완료 기준 회수 및 원본 버퍼 참조 유지 |
| F04 | P0 | 4.2 | 게임 드로우를 두 번 실행하면 부작용과 중복 기록 경쟁이 발생 가능 | 순수 디코딩 벤치와 격리 재생으로 변경 |
| F05 | P1 | 3.1~3.3, 7 | 푸시 상수·SPIR-V ABI가 “8바이트 추가”로만 기술됨 | offset·stride·stageFlags·capability를 수치로 고정 |
| F06 | P1 | 3.2~3.4 | raw Maxwell 형식과 기존 mapped VkFormat/r75 후처리의 관계가 불완전 | 기존 입력 결과까지 재현하는 디코딩 계약 작성 |
| F07 | P1 | 3.2, 3.3 | inputRate 및 instance-rate divisor=0 처리 미정 | 상태를 보존하거나 발생 불가 근거를 명시 |
| F08 | P1 | 3.5, 4.1 | 보정식의 GPU 반올림 보장과 VAF 비트 일치를 동일시 | 연산 전제 및 3종 기준값 비교를 명시 |
| F09 | P1 | 3.1, 3.8, 5 | pipeline key만으로 shader cache·variant 증가까지 해결된다고 판단 | shader 캐시 키·직렬화·동적 상태를 별도 검토 |
| F10 | P1 | 3.3, 3.6~3.8 | 미지원 스트림의 zero 처리, 미러 ON/OFF 상태 전이가 불완전 | 명시적 폴백과 미러 재동기화·보호 소유권 처리 |
| F11 | P1 | 1.2, 3.7, 4 | 선택한 BDA+고정 IB 조합의 직접 측정 및 특수 경로 검증이 없음 | production과 같은 벤치 arm 추가, 초기 지원 범위 제한 |
| F12 | P2 | 0, 1.1, 3.7~3.8, 4~6 | 기본값·효과·완료 판단에 과도한 단정이 있음 | 단계별 기본값과 실측 범위를 일치시킴 |

## 3. F01 — 소비 단계 변경에 맞는 동기화

기존 정점 입력은 `VERTEX_ATTRIBUTE_INPUT / VERTEX_ATTRIBUTE_READ`, 풀링은 `VERTEX_SHADER / SHADER_STORAGE_READ`에서 소비한다. physical storage buffer 읽기는 후자에 포함된다. 기존 정점 입력 배리어만으로 새 읽기의 가시성을 보장할 수 없다. [1]

| 데이터 흐름 | 필요한 처리 |
|---|---|
| GPU compute/transfer/transform-feedback 쓰기 → 정점 풀링 | 실제 생산자의 write → `VERTEX_SHADER`와 `SHADER_STORAGE_READ` |
| 인덱스 미러 복사 → indexed draw | 기존 `INDEX_INPUT`와 `INDEX_READ` 유지 |
| 같은 원본을 FF와 풀링이 함께 사용 | 실제 소비 단계·접근 마스크를 모두 포함 |
| CPU 쓰기 → 원본 직접 읽기 | 제출 전 쓰기의 순서와 플랫폼 캐시 일관성 보장 |
| GPU 읽기 → CPU 덮어쓰기·풀 해제 | GPU 사용 완료 및 자원 수명 보장 |

여기서 추가할 것은 필요한 의존성이다. 모든 드로우 앞에 전역 대기나 `ALL_COMMANDS` 배리어를 무조건 넣는 설계로 바꾸지 않는다.

`vbm_bind()`와 `vkCmdBindVertexBuffers*()`를 생략해도 다음은 유지해야 한다.

- 실제 import 버퍼의 GPU 사용 기록과 참조 유지.
- GPU가 생산한 범위와 alias 관계 추적.
- 게임 동기화 명령의 호스트 대응.
- CPU가 GPU 소비 전에 원본을 덮어쓰지 않도록 하는 기존 보장.

페이지 보호는 미러의 변경 감지 수단이다. 이를 제거해도 CPU/GPU 사이의 순서 보장이 없어지지 않는다. import 원래 host pointer의 캐시 규칙은 `vkMapMemory()`가 반환한 포인터의 규칙과 구분해야 한다. [2]

## 4. F02 — 읽기 폭, 주소 산술, 범위 검사

### 4.1 “32비트 워드 2~3개”는 16바이트 형식을 처리하지 못함

3.4절의 `32_32_32_32`는 정렬된 상태에서도 4워드가 필요하다. 설계가 지원하려는 비정렬 읽기까지 포함하면 최대 5워드다. 12바이트 형식도 비정렬이면 4워드가 필요하다.

```text
wordStart = addr & ~uint64(3)
wordCount = ((addr & 3) + attributeBytes + 3) / 4
```

형식 전체를 읽는 구현이면 이 계산을 사용한다. 성분별로 1~2워드씩 읽는 구현이라면 그 방식과 중복 load 제거를 명시해야 한다. 현재 문구처럼 속성 전체를 2~3워드로 읽는다고 작성하면 안 된다.

실제 load는 약속한 정렬을 충족해야 한다. `PhysicalStorageBuffer`의 `Aligned 4`는 컴파일러에 주는 보장이므로 주소를 실제로 4바이트 경계에 맞춰야 한다. 정렬된 경우에 32비트 값의 `<<32`·`>>32`를 생성하지 않도록 추출식을 분기한다. [3]

### 4.2 세 가지 범위를 구분

1. 게임이 지정한 논리 스트림 범위.
2. 기존 Vulkan 경로가 실제로 바인딩한 범위.
3. BDA를 얻은 해당 `VkBuffer`가 제공하는 유효 주소 범위.

`S.vbuf[s].size`만으로 세 범위가 같다고 가정할 수 없다. 속성 바이트가 논리 범위 안에 있어도, 정렬된 u32 load가 실제 버퍼 끝을 넘어갈 수 있다. BDA 참조에는 해당 버퍼의 범위가 적용된다. [4][5]

청크 기반 풀에서는 스트림이 다음 청크로 넘어갈 때 BDA도 연속이라고 가정하지 않는다. 전체 접근이 한 청크에 들어가는 경로, 명시적 청크 주소 해석, 기존 경로 폴백 중 하나를 정한다. 단순히 길이를 잘라 나머지 정점을 0으로 만드는 처리는 기존 유효 지오메트리를 바꿀 수 있다.

### 4.3 산술과 검사 순서

- 인덱스를 검증한 뒤 **곱셈 전에** 64비트로 확장한다.
- `idx * stride`, attribute offset, base address 추가의 overflow를 방지한다.
- 범위 검사는 `rel <= size && bytes <= size - rel`처럼 덧셈 overflow를 피한다.
- 논리 속성 범위뿐 아니라 실제 `[wordStart, wordStart + 4*wordCount)`를 보장한다.
- 부족한 끝부분은 지원되는 좁은 load, 실제로 확보된 padding, 또는 폴백으로 처리한다.
- `u32 size`를 유지하려면 모든 지원 범위가 4GiB 미만임을 명시하고 host에서 검사한다. `VK_WHOLE_SIZE`를 그대로 줄여 저장하지 않는다.
- 유효성 검사는 **load 전에 분기**한다. 잘못된 주소를 먼저 읽고 `select(valid, value, 0)`를 적용해도 잘못된 접근은 이미 발생한다.

FF 비교 기준 역시 해당 VkFormat의 정렬 규칙을 지켜야 한다. 기존 경로에서 유효하지 않은 오정렬 입력은 FF 비트 동등성의 정답으로 사용하지 않는다. [6]

## 5. F03·F05 — scratch 수명과 ABI

### 5.1 테이블 수명

푸시 상수에 저장되는 것은 주소다. 그 주소가 가리키는 256바이트 테이블까지 명령 버퍼에 복사되는 것은 아니다.

테이블은 명령 기록 이후 마지막 GPU 사용 완료까지 불변이어야 한다. scratch 슬롯은 fence/timeline 완료값으로 회수한다. 같은 슬롯을 다음 드로우나 다음 submit 기록 과정에서 덮어쓰면 앞 드로우가 뒤 드로우의 테이블을 읽을 수 있다.

“같은 내용이면 재사용”은 기존 슬롯의 생존 여부와 버퍼 세대가 확인될 때만 허용한다. 원본 버퍼와 scratch 버퍼 모두 GPU 사용 완료까지 유지하며, 풀 재생성 시 캐시한 BDA를 무효화한다. 이는 Vulkan 자원의 비동기 사용과 수명 규칙에서 요구되는 조건이다. [7]

### 5.2 테이블 ABI

현재 16바이트 항목을 유지한다면 다음을 host와 SPIR-V에서 일치시킨다.

| 필드 | 바이트 offset | 크기 |
|---|---:|---:|
| address | 0 | 8 |
| stride | 8 | 4 |
| size | 12 | 4 |
| 항목 배열 stride | 16 | — |
| 16개 전체 | — | 256 |

형식과 주소 정렬, 실제 버퍼 범위 보장을 위해 항목을 확장한다면 테이블 크기도 함께 다시 계산한다. 256바이트 자체는 필수 조건이 아니다.

### 5.3 푸시 상수 ABI

기존 Rescaling 데이터가 문서대로 28바이트이고 u64 멤버를 추가한다면, 주소 필드는 8바이트 정렬을 충족하는 **offset 32**, 끝은 **40바이트**로 두는 것이 명확하다. `28 + 8 = 36`만으로 처리하지 않는다. 다른 표현을 선택하면 해당 표현의 정렬을 별도로 명시한다.

RenderArea 경로와도 주소 offset을 통일하고 `offsetof/sizeof`, SPIR-V `Offset/ArrayStride`, `VkPushConstantRange`, `vkCmdPushConstants`의 offset·size·stageFlags를 맞춘다. 실제 소비 단계인 `VERTEX_BIT`가 포함돼야 하며, 겹치는 range의 단계 플래그 규칙과 기존 layout 호환성도 지켜야 한다. [8]

### 5.4 BDA 활성화 조건

`g_imp.bda`만 확인하는 것으로 끝내지 않는다. import 버퍼와 scratch 버퍼 양쪽에서 다음이 성립해야 한다. [9]

- `bufferDeviceAddress` 활성화.
- `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`와 필요한 allocation flag.
- `PhysicalStorageBufferAddresses` capability와 `PhysicalStorageBuffer64` addressing model.
- 필요한 SPIR-V 버전/extension 및 load의 `Aligned` 피연산자.
- u64 산술을 생성한다면 `shaderInt64` 지원·활성화. BDA 자체가 shaderInt64를 보장하지는 않는다.

## 6. F06·F07 — 정점 의미와 형식 계약

### 6.1 인덱스 공식

Vulkan의 `VertexIndex`는 indexed draw의 인덱스 값에 `vertexOffset`이 반영된 값이며, non-indexed draw에서는 `firstVertex`부터 시작한다. 이 값에 offset을 다시 더하지 않는다. `firstIndex`는 인덱스 버퍼에서 읽을 위치를 선택하며 최종 정점 번호에 추가하는 값이 아니다. [10]

divisor>0의 제안식은 Vulkan 주소 계산과 일치한다. 전체 규칙은 다음과 같다. [6][11]

```text
inputRate == VERTEX:
    idx = VertexIndex
inputRate == INSTANCE && divisor == 0:
    idx = BaseInstance
inputRate == INSTANCE && divisor > 0:
    idx = BaseInstance + (InstanceIndex - BaseInstance) / divisor
```

현재 구조에는 inputRate가 없다. 이를 보존하거나, 현재 NVN/Maxwell→Eden 매핑에서는 instance-rate divisor=0이 생성되지 않는다는 소스 근거를 남긴다. 현재 게임에 그 상태가 실제 존재한다고 단정하는 지적은 아니다.

FF 비교·폴백에서는 divisor 관련 지원 조건을 만족해야 한다. KHR/core의 `supportsNonZeroFirstInstance` 등도 비교 대상 경로의 조건으로 확인한다. 순수 풀링 계산 자체가 고정 기능 divisor 지원을 요구하는 것은 아니다. [12]

signed `vertexOffset`과 32비트 정점 번호의 해석도 명시한다. 음수 offset을 무조건 거부하지 말고, offset 적용 후 유효한 주소가 되는 정상 사례를 포함한다. 32비트 상위 비트와 64비트 확장 정책은 기존 경로와 맞춘다.

### 6.2 디코더의 기준은 기존 mapped VkFormat과 후처리

원문의 기존 경로는 다음 순서다.

```text
Maxwell 형식
  → vertex_format()/r75
  → 실제 VkFormat에 따른 추출·변환
  → GetAttributeType / Bitcast / SToF / UToF
  → 기존 IR이 받는 값
```

새 구현은 이 마지막 결과를 재현해야 한다. raw Maxwell `Size/Type/bgra`만 해석해도 같은 결과가 된다는 근거는 현재 없다.

속성별로 다음을 확정한다.

- 실제 mapped VkFormat와 유효한 guest 형식 조합.
- 성분 폭·순서·부호·기본 성분.
- Vulkan 입력의 수치 타입과 기존 IR 결과 타입.
- r75 변환과 scaled 우회가 적용되는 위치.
- BGRA가 VkFormat에 이미 반영됐는지 여부.

BGRA는 한 번만 적용한다. integer 성분 `1`을 채운 후 Bitcast하는 것과 float `1.0`을 직접 채우는 것은 다른 비트다. `support_scaled_attributes`와 무관하게 결과가 같다는 주장은 실제 분기를 대조한 뒤 확정한다.

지원 형식은 크기×타입의 모든 조합으로 선언하지 말고 현재 `vertex_format()`의 유효 매핑을 기준으로 한다. 11_11_10도 실제 mapped VkFormat의 unsigned float 배치를 따른다. [13]

generic 속성 읽기의 진입점은 `EmitGetAttribute` 한 함수에 한정하지 않고, 동적 속성 인덱싱 등 다른 IR/emitter 경로가 존재하는지 확인한다. 이 항목은 소스 미제공으로 존재 여부가 미확인이다.

## 7. F08 — 정규화 보정식의 실제 검증 결과

### 7.1 호스트 전수 계산

검수 과정에서 아래 식을 직접 검사했다.

```text
r = RN_binary32(1 / D)
q = RN_binary32(float(c) * r)
e = fmaf(-q, D, float(c))
result = fmaf(e, r, q)
SNORM: result = max(result, -1.0)
```

기준값은 double 나눗셈이 아니라 정수 나눗셈·나머지로 유리수 `c/D`를 binary32 round-to-nearest, ties-to-even으로 변환해 만들었다. 실험은 x86_64 Linux, Python 3.12.14, glibc 2.41의 `fmaf`를 사용했다.

| 범위 | 검사 수 | 기준값과 불일치 |
|---|---:|---:|
| UNORM, b=1~16 전체 | 131,070 | 0 |
| SNORM, b=2~16 전체 | 131,068 | 0 |
| **합계** | **262,138** | **0** |

RN binary32 곱셈·덧셈으로 두 FMA를 분해한 경우도 같은 범위에서 불일치가 없었다. 따라서 이 도메인에서 “FMA를 분해하면 반드시 틀린다”고 결론 내리지 않는다.

**보정식은 유지할 근거가 있다. 이 결과는 대상 N1X GPU에서 실행한 결과가 아니다.**

### 7.2 문서에서 빠진 GPU 연산 전제

`GLSL.std.450 Fma`와 정확히 반올림되는 fused FMA를 동일시하면 안 된다. Khronos는 기존 FMA primitive가 fused임을 보장하지 않는다고 설명하며, 정확한 fused 연산을 위한 `VK_KHR_shader_fma` / `OpFmaKHR`를 별도로 정의한다. 해당 장치의 `shaderFmaFloat32` 지원·활성화 여부는 미확인이다. [14][15]

또한 정확히 반올림되는 연산도 RTE/RTZ 설정이 없으면 반올림 방향이 구현 정의다. 다음은 정확 유리수로 RTZ를 모사한 반례이며, NVIDIA에서 관찰한 결과는 아니다. [16]

```text
UNORM8: c=1, D=255
r = RTE(1/255)       = 0x3b808081
보정식의 RTZ 결과   = 0x3b808080
정확한 RTE 기준값   = 0x3b808081
```

따라서 “정수 c와 D가 정확하므로 GPU 결과도 RN으로 보장된다”는 논증은 불충분하다. reciprocal의 비트, 사용 opcode, 반올림 방향, 재배열 제한을 고정해야 한다. 디코더를 위해 전역 FP 실행 모드를 바꿀 경우 기존 게임 셰이더의 다른 연산에 미치는 영향도 함께 확인한다.

이 확장이 없다고 보정식을 폐기할 필요는 없다. 그 경우 해당 드라이버의 production SPIR-V 전수 검사로 검증된 경로라고 한정한다.

### 7.3 세 가지 기준값을 구분

membench4는 다음을 각각 기록한다.

1. 정확 유리수 기반 CPU 기준값.
2. N1X 고정 기능 입력 직후의 비트.
3. 실제 production 풀링 디코더 직후의 비트.

수학적 RN 일치와 기존 VAF 비트 일치는 별도 검사다. 일반적인 정규화·FP 변환 규정만으로 모든 결과 비트가 항상 같은 RN 표현이라고 가정하지 않는다. [7]

검증 범위에는 다음을 추가한다.

- packed 10_10_10_2의 **2비트 alpha**, 부호 확장과 SNORM 최솟값 포화.
- half 및 11/10비트 unsigned float의 0, subnormal, Inf, NaN 처리.
- 정수·scaled 형식의 경계값과 32비트 값의 `2^24` 부근.
- 빠진 성분, BGRA, offset·stride, 한 워드 또는 청크 경계에 걸치는 입력.

NaN의 의미가 같다는 것과 payload 비트까지 같다는 것은 구분한다. 합법적인 기존 FF 입력을 기준으로만 비교한다.

LUT는 차이가 확인된 형식에 한정한다. binary32 결과를 저장하면 8비트 표는 1KiB, 10비트 표는 4KiB, 16비트 표는 256KiB다. 16비트 표를 무조건 셰이더 상수 배열에 넣지 않는다. 장치·드라이버가 바뀌면 VAF 기준표의 유효성도 재검토한다.

### 7.4 3.5절 대체 문구

> 제안한 보정식은 고정된 binary32 reciprocal과 RN 연산을 사용한 호스트 전수 검사에서 UNORM b=1~16 및 SNORM b=2~16의 262,138개 입력 모두 정확 유리수 기반 binary32 RTE 기준값과 일치했다. 이는 대상 GPU의 opcode·반올림·최적화 동작 또는 고정 기능 입력과의 비트 일치를 보장하지 않는다. production SPIR-V의 연산 계약을 명시하고 대상 장치의 고정 기능 입력 직후 비트와 전수 비교해 형식별 디코더를 확정한다.

## 8. F04 — 비교 캡처는 게임 상태를 두 번 변경하면 안 됨

4.2절처럼 같은 게임 드로우를 FF와 풀링으로 연속 실행하면 SSBO/image 쓰기, atomic, transform feedback, query, depth/stencil, blending 효과가 두 번 발생할 수 있다. 두 번째 실행이 같은 상태에서 시작한다는 보장이 없다.

또한 Vulkan은 같은 정점이 여러 번 등장할 때 VS 결과 재사용을 허용하지만 실행 횟수를 고정하지 않는다. 따라서 `[instance, vertex]` 위치에 비원자적으로 기록하는 캡처는 중복 invocation의 write/write 경쟁이 생길 수 있다. 같은 값이라고 해서 경쟁이 안전해지는 것은 아니다. [17][4]

검증은 다음 순서로 변경한다.

1. 입력 스냅샷을 순수 속성 디코딩 셰이더로 비교한다.
2. 전체 VS 비교는 부작용 없는 셰이더로 제한하거나 writable 리소스·TF·query 등을 격리한 재생으로 수행한다.
3. 중복 기록은 atomic slot claim 또는 invocation별 독립 기록으로 처리한다. 대응 키에는 draw·instance·vertex와 필요한 view를 포함한다.
4. 계측용 SSBO 저장을 제거한 빌드에서 성능을 측정한다.

`rasterizerDiscard`만으로 shader storage와 transform feedback 부작용이 제거된다고 전제하지 않는다.

## 9. F09 — 파이프라인·셰이더 캐시

“VIDS가 꺼졌으므로 파이프라인 수 증가 없음”은 단일 모드에서 같은 상태 집합을 비교할 때만 성립할 수 있다. ON/OFF와 폴백이 공존하면 variant가 생긴다. 기존 하나의 shader module을 여러 정점 형식이 공유했다면, 형식을 코드에 굽는 풀링에서는 shader module 수도 증가할 수 있다.

다음을 설계에 추가한다.

- 풀링 코드에 영향을 주는 형식·offset·stream·inputRate/divisor·enable·변환 규칙을 shader 캐시의 equality/hash에도 반영.
- graphics pipeline key, shader 재사용 키, disk cache 직렬화를 각각 확인.
- 실제 Vulkan vertex-input state를 비우기 전에 원래 메타데이터를 보존.
- EDS1 stride는 테이블의 동적 값으로 유지하고 변경 시 테이블 재사용 판정 갱신.
- BDA 주소나 scratch 위치를 셰이더·파이프라인 키에 넣지 않음.
- 모드 전환 시 새 variant의 컴파일 비용을 정상 GPU 성능과 분리 기록.

캐시 버전 증가는 필요할 수 있지만, 누락된 runtime key를 대신하지 않는다.

## 10. F10 — 폴백과 기존 미러 상태

### 10.1 0 입력으로 묶으면 안 되는 상태

| 상태 | 처리 |
|---|---|
| 명시적으로 비활성인 속성 | 기존 emitter의 상수 규칙 유지 |
| 기존 경로의 의도적인 zero-buffer | 해당 형식으로 0 원시 데이터를 디코딩하고 빠진 성분을 채움 |
| 유효하지만 현재 풀링이 주소를 해석하지 못하는 스트림 | 드로우 전체를 기존 FF+미러로 폴백 |
| 범위 밖 입력 | 명시한 안전 정책으로 처리하고 정상 범위 동등성 검사와 분리 |

원시 0의 RGBA 형식은 alpha도 0이지만, RGB 형식은 빠진 alpha가 기본값 1이다. “주소 0이면 항상 `(0,0,0,1)`”로 구현하지 않는다. [5]

robustBufferAccess가 꺼진 기존 경로의 범위 밖 결과는 안정적인 비교 기준이 아니다. 풀링에서 0 반환을 선택하는 것은 명시적 정책으로 둘 수 있지만, 기존 범위 밖 결과와의 비트 동등성을 주장하지 않는다.

### 10.2 미러 ON/OFF

풀링에서 `vbm_bind`를 호출하지 않는 것만으로 기존 미러 메모리와 페이지 보호가 사라지지 않는다. 이미 기록된 GPU 작업이 사용하는 미러는 완료 후 회수한다.

풀링 중 보호를 해제했다면 OFF 복귀 시 미러가 오래된 데이터일 수 있다. 전체 재동기화 또는 세대/변경 이력에 따른 갱신을 수행한다. VB와 IB가 같은 4KB 페이지를 공유하면 남아 있는 IB의 보호 소유권도 고려한다.

실시간 A/B는 프레임·패스 그룹 경계에서 적용하는 것을 권장한다. 깊이 프리패스와 컬러 패스 사이에서 입력 변환 모드가 섞이지 않도록 한다. 다음 드로우에서 키만 바꾸는 것으로 모드 전환 처리가 끝나지는 않는다.

## 11. F11·F12 — 성능 근거와 특수 경로

### 11.1 설계와 동일한 조합을 한 번 더 측정

원문 g안은 **SSBO 속성 풀링 + 고정 IB**다. 설계는 **BDA + fetch table + 보정 디코더 + 고정 IB**다. d안은 인덱스까지 BDA로 풀링하므로 최종 설계와 다르다.

기존 벤치에 production과 같은 조합을 추가하면 된다. 별도의 대규모 게임 벤치를 먼저 늘릴 필요는 없다. g안 대비 테이블 간접 읽기와 보정 디코더 비용을 함께 확인한다.

g/a의 기록된 시간비는 39×1399에서 약 **1.079배**, 183×1399에서 약 **0.895배**다. 첫 조건은 기준보다 약 7.9% 느리고 두 번째는 약 10.5% 빠르다. 따라서 “항상 기준보다 빠르다”는 결론은 아니다.

b/a는 각각 약 **76.0배**, **94.3배**다. 1.1절의 100~250배는 다른 벤치 조건의 수치라면 그 출처와 조건을 분리한다. 서로 다른 벤치의 수치를 한 범위로 일반화하지 않는다.

고정 인덱스 유지로 정점 결과 재사용 기회가 유지되는 방향은 타당하다. 그러나 Vulkan이 동일한 캐시 재사용률을 보장하는 것은 아니므로, “g가 빠른 이유는 캐시 때문”이라는 확정 설명은 제거한다. [17]

### 11.2 특수 경로의 초기 처리

| 경로 | 초기 처리 |
|---|---|
| 일반 Eden VertexB | 아래 최소 지원 범위에서 적용 |
| VertexA+B 결합 | 외부 generic 입력과 A→B 중간 값의 관계가 확인될 때까지 풀링 제외 |
| NVN 자체 번역기 | 기존 경로 유지, 후속 단계 |
| Mesh shader | 기존 mesh 경로 유지; VAF 풀링 대상 아님 |
| GS/TES/XFB | 대표 경로의 입력·출력 및 동기화 검증 후 지원 |
| quad/restart | 기존 인덱스·topology 처리를 유지하되 경계 사례 검증 |
| 간접 indexed draw | import IB가 남는 예외를 명시하고 별도 측정·정책 결정 |

VertexA+B를 “첫 단계가 정점 셰이더가 아닌 경우”라고 표현하면 부정확하다. 결합된 결과가 하나의 Vulkan Vertex 실행 모델일 수 있으므로, 정확한 결합 모드로 제외해야 한다.

간접 indexed draw는 문서상 인덱스 미러를 사용하지 않는다. 따라서 “풀링에서는 항상 인덱스만 미러”라는 요약에 예외가 필요하다. GPU가 횟수를 정한다는 이유만으로 인덱스 미러가 불가능한 것은 아니지만, 안전한 전체 바인딩 범위나 GPU 생산·복사 순서를 별도로 설계해야 한다.

### 11.3 효과와 완료 기준 문구

- “정점 복사·쓰기 보호·폴트가 없어진다” → “지원되는 풀링 드로우에서 정점 미러 때문에 필요한 복사·보호를 제거한다.”
- “셰이더 읽기와 복사는 정상” → “명시한 벤치 조건에서 큰 성능 저하가 관찰되지 않았다.”
- 스위치 기본값 → “개발·검증 단계 0, 통과한 지원 범위에 한해 기본 1.”
- 60fps → GPU timestamp, CPU 제출 비용, p50/p95와 함께 평가. 계측·컴파일 시간을 분리.
- invalid-SP 0 및 10분 플레이 → 관찰 결과로 기록. `tebfix`의 정확성은 별도 검증.

풀링으로 보호 예외가 줄어 강종이 재현되지 않는 것만으로 NCE 스택 전환 결함이 수정됐다고 판정하지 않는다.

## 12. 더 작은 첫 구현과 완료 조건

### 단계 A — 단일 production 디코더 검증

기존 벤치에 BDA+테이블+보정 디코더+고정 device-local IB를 추가한다. 코드 경로는 게임에서 사용할 emitter와 재사용한다. 처음에는 이미 측정한 float3, half2, unorm8x4, packed 10-10-10-2, instance float4에 해당하는 실제 VkFormat만 대상으로 한다.

CPU 기준, VAF 입력 직후, 풀링 입력 직후의 세 값을 비교하고, production 조합의 GPU 시간을 측정한다. 1ulp 차이의 원인이 어느 단계인지 분리한다.

### 단계 B — 호스트 상태·수명 통합

일반 Eden VertexB·직접 indexed draw·한 import 청크 안에 들어가는 검증된 형식만 활성화한다. 나머지는 기존 경로로 폴백한다. 같은 제출의 여러 테이블, 여러 프레임 진행 중 재사용, GPU 생산 정점, CPU 갱신, 버퍼 세대 교체, ON→OFF 재동기화를 작은 재현 장면에서 확인한다.

이 단계에서 F01~F03, F05의 조건과 SPIR-V validation을 통과시킨다. correctness 검증과 성능 측정은 분리한다.

### 단계 C — 게임 비교

지원된 경로에서 입력 스냅샷과 격리된 VS 결과를 비교한다. 이후 실제 게임 장면으로 화면과 GPU/CPU 비용을 확인한다. fallback 횟수와 이유, 실제 import IB 사용량도 함께 기록한다.

### 단계 D — 지원 확대

추가 형식, 비인덱스·간접 경로, GS/TES/XFB, VertexA+B, NVN 경로를 확인된 항목부터 확장한다. “모든 형식을 먼저 끝내야 첫 게임 검증을 시작한다”는 순서 대신 검증된 형식 allowlist와 폴백을 사용한다.

## 13. 검수 요청 8개 항목에 대한 답

| 요청 | 판정 |
|---|---|
| 1. 의미 동등성 | VertexIndex와 d>0 공식은 맞음. inputRate/d=0, mapped VkFormat/r75/scaled 계약 보완 필요 |
| 2. 비트 정확도 | RN 호스트 전수 검사 통과. GPU 연산 계약과 VAF 비트 일치는 별도 검증 |
| 3. 범위 밖 0 반환 | 명시적 정책으로 가능. 실제 load 이전 검사가 필수이며 FF의 undefined 결과와 동등성 주장 불가 |
| 4. host scratch/푸시 상수 | 불변 기간·GPU 완료 회수·정확한 ABI·stageFlags 보완 필요 |
| 5. 파이프라인 키 | stride 동적 테이블 방향은 타당. shader cache 키와 variant 증가 판단은 수정 필요 |
| 6. 빠진 경우 | VertexA+B 명시적 제외, 간접 IB 예외, 형식 경계·특수값·부작용 검증 추가 |
| 7. 미러 공존 | VS storage read 동기화, IB 동기화, 자원 수명, ON/OFF 재동기화·공유 페이지 보호 필요 |
| 8. 단계·검증 | 전체 형식 구현 전에 동일 production 벤치와 작은 지원 범위를 먼저 검증 |

열린 질문의 `support_scaled_attributes` 값과 VertexA+B 내부 결합 방식은 첨부 문서만으로 확정할 수 없다. 푸시 상수의 정점 단계 접근 허용은 필요한 조건이며, 현재 코드가 이미 충족하는지는 미확인이다.

## 14. 최종 판정

**정점 속성 풀링을 진행할 기술적 근거는 충분하다. 현재 설계를 그대로 기본 적용하는 것은 적절하지 않다.**

첫 구현은 기존 미러를 보존한 상태에서 범위가 명확한 BDA 풀링 경로로 제한한다. 메모리·동기화·ABI 조건을 먼저 고정하고, 실제 production 디코더를 기존 벤치에 넣어 확인한 뒤 지원 범위를 넓힌다.

## 출처

프로젝트 상태·측정 수치는 사용자 제공 `VERTEX_PULLING_DESIGN.md`와 `ASTRA_REVIEW_REQUEST.md`에 근거한다. 두 문서에 적힌 소스 위치는 코드 검토 완료의 증거로 취급하지 않았다.

- [1] [Khronos — VkAccessFlagBits2](https://docs.vulkan.org/refpages/latest/refpages/source/VkAccessFlagBits2.html)
- [2] [Khronos — Memory Allocation](https://docs.vulkan.org/spec/latest/chapters/memory.html)
- [3] [Khronos — Buffer Device Address Alignment](https://docs.vulkan.org/guide/latest/buffer_device_address_alignment.html)
- [4] [Khronos — Memory Model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)
- [5] [Khronos — vkCmdBindVertexBuffers2](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindVertexBuffers2.html)
- [6] [Khronos — Fixed-Function Vertex Processing](https://docs.vulkan.org/spec/latest/chapters/fxvertex.html)
- [7] [Khronos — Fundamentals](https://docs.vulkan.org/spec/latest/chapters/fundamentals.html)
- [8] [Khronos — Push Constants](https://docs.vulkan.org/guide/latest/push_constants.html)
- [9] [Khronos — Buffer Device Address](https://docs.vulkan.org/guide/latest/buffer_device_address.html)
- [10] [Khronos — VertexIndex](https://docs.vulkan.org/refpages/latest/refpages/source/VertexIndex.html)
- [11] [Khronos — BaseInstance](https://docs.vulkan.org/refpages/latest/refpages/source/BaseInstance.html)
- [12] [Khronos — VkPhysicalDeviceVertexAttributeDivisorProperties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceVertexAttributeDivisorProperties.html)
- [13] [Khronos — Formats](https://docs.vulkan.org/spec/latest/chapters/formats.html)
- [14] [Khronos — VK_KHR_shader_fma](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_fma.html)
- [15] [Khronos — VkPhysicalDeviceShaderFmaFeaturesKHR](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceShaderFmaFeaturesKHR.html)
- [16] [Khronos — Vulkan Environment for SPIR-V](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
- [17] [Khronos — Shaders / Vertex Shader Execution](https://docs.vulkan.org/spec/latest/chapters/shaders.html)
