# GPU 메모리 캐시 수정본 검수 — device-local A안

- 검수일: 2026-11-20
- 기준: `GPU_MEMORY_CACHE_DESIGN(1).md`, `ASTRA_GPU_MEMORY_CACHE_REVIEW_REQUEST(1).md`
- 판정: **A안의 방향은 타당하다. 아래 정확성 계약을 보완한 뒤 구현한다.**
- 범위: 첨부 설계와 변경분을 검토했다. 문서가 가리키는 프로젝트 소스, 벤치 원본, 실행 로그는 첨부되지 않았으므로 실제 구현까지 확인한 결과는 아니다.
- 확정 조건: device-local A안을 사용한다. B안 성능 검증, 메모리 용량·예산·LRU·부분 사본을 통한 절약은 이번 검수 대상에서 제외한다. DK 제외와 동일 주소 픽셀 쓰기 벤치 취소도 유지한다.

## 1. 핵심 결론

**GPU의 일반적인 버퍼 읽기·쓰기 대상을 device-local 사본으로 통일하는 방향은, 정점 미러나 정점 풀링보다 이번 문제의 범위를 제대로 다룬다.** CPU의 NCE 주소 공간을 바꾸지 않고 GPU 저장소를 분리하는 선택도 일관된다.

그러나 다음 세 가지는 구현 전에 수정해야 한다.

1. **되돌림의 기본 경로를 별도 readback staging으로 고정한다.** device-local에서 게스트 원본으로 먼저 GPU 복사한 뒤 `snap_merge`하면, CPU 변경분을 병합 전에 덮어쓸 수 있다.
2. **4 KB dirty 감지와 최신 데이터의 소유권을 구분한다.** CPU와 GPU가 같은 페이지의 서로 다른 부분을 수정했을 때, 페이지 전체 업로드·다운로드가 한쪽 변경을 지우지 않아야 한다.
3. **모든 별칭과 접근 경로가 같은 device-local 저장소를 사용하도록 한다.** SSBO만 local로 바꾸고 raw BDA, bridge, 게임의 copy/fill이 원본이나 별도 사본을 보면 정확성이 깨진다.

첫 단계는 **A 사본 + 모든 dirty 페이지를 대상으로 하는 증분 업로드 + 제출별 readback staging + 기존 완료 경로의 정확성 보강**으로 잡는 것이 좋다. 바인딩 범위 지연 업로드와 GPU dirty bitmap은 실제 비용을 확인한 뒤 추가한다.

여기서 “모든 dirty 페이지 업로드”는 매번 풀 전체를 복사한다는 뜻이 아니다. 또한 사용자가 정한 전체 풀 사본 정책을 바꾸는 제안도 아니다.

## 2. 수정본에서 해결된 것과 남은 것

| 항목 | 판정 |
|---|---|
| GPU 사본 위치 | A=device-local로 확정. B의 미측정 성능 전제 R7은 종료 |
| 기존 B안 설명의 CPU 직접 접근 구조 | A안에서는 제출별 GPU 다운로드와 staging 수명 문제로 바뀜 |
| R9: 다운로드 완료 뒤 `g_done` 공개 | 필요한 조건을 추가했지만 barrier, snapshot, 병합, 공개 순서를 더 구체화해야 함 |
| `local → import 원본 → snap_merge` | **새 P0 문제. 일반 되돌림 경로로 사용하면 안 됨** |
| 4 KB/volatile 페이지, CPU·GPU 부분 갱신 | 설계 명세의 정확성 조건 누락 유지 |
| alias·bridge·raw BDA, 이벤트·query·s56 | 설계 명세의 정확성 조건 누락 유지 |
| 단계와 설명 | §6에는 아직 “1단계 B / 2단계 A”, §4.5에는 CPU memcpy 업로드가 남음 |

이 문서의 P0는 **데이터 손실·오래된 값 관측·교착을 막기 위해 구현 전에 계약을 정해야 하는 항목**이다. 실제 코드에 해당 결함이 존재한다고 확정한 표시는 아니다.

## 3. 저장소의 역할부터 분리해야 한다

수정본 §4.1은 “import하지 않음”, §4.2는 “import를 복사 원본·대상으로 사용”이라고 적는다. 구현자가 다른 경로를 선택하지 않도록 역할을 다음과 같이 정리한다.

| 역할 | 저장소와 규칙 |
|---|---|
| NCE가 직접 접근하는 게임 메모리 | 기존 게스트 원본 G |
| 게임 GPU 명령의 버퍼 접근 | 해당 주소 구간의 유일한 device-local 사본 D |
| 캐시 업로드의 원본 | 안전 조건을 충족한 G의 import 버퍼, 또는 안정된 upload staging U |
| 캐시 다운로드의 목적지 | 제출에 귀속된 별도 readback staging R |
| CPU가 결과를 관측할 저장소 | R의 결과를 검증·병합한 뒤의 G |

**게임의 NVN copy/fill도 D를 사용해야 한다.** import에서도 복사가 빠르다는 이유로 게임 copy/fill을 G에 남기면, GPU의 최신값이 D에 있는 상황에서 오래된 G를 읽거나 잘못된 저장소를 갱신하게 된다.

import를 남겨도 되는 것은 **캐시 자체의 제한된 전송 endpoint**이다. 렌더링·compute·간접 명령·게임 copy/fill의 일반 바인딩으로 다시 선택돼서는 안 된다.

`p->zero_copy` 같은 단일 상태가 “전송용 import가 존재함”과 “게임 GPU 접근이 import를 사용함”을 동시에 뜻하면 안 된다. 최소한 다음 역할이 구별돼야 한다.

- 실제 GPU 바인딩 버퍼와 BDA.
- 업로드용 import 원본의 존재.
- 제출별 readback 버퍼와 수명.

구체적인 구조체 이름은 기존 코드를 보고 정하면 된다. 새 플래그를 많이 만드는 것이 목적이 아니라, 저장소 역할의 혼용을 막는 것이 목적이다.

## 4. P0 — 게스트 원본에 직접 GPU 복사한 뒤 병합하면 늦다

### 문제 위치

수정본 §4.2:

> 사본 → import 원본(또는 staging) GPU 복사를 제출 끝에 기록하고, 완료 스레드의 pools_copy_back이 snap_merge 규칙으로 게스트 페이지를 맞춘다.

### 성립해야 할 정확성

같은 페이지의 서로 다른 구간을 CPU와 GPU가 갱신하는 경우에도 두 결과를 보존해야 한다. 비중첩 접근이 독립적으로 유효한 경우를 캐시 구현이 새 경합으로 바꾸면 안 된다. [S1]

설계의 누락을 보여주는 순서는 다음과 같다. 실제 게임에서 이 순서를 관측했다는 뜻은 아니다.

1. 원본 G와 사본 D의 초기값이 `(a=0, b=0)`이다.
2. GPU는 D의 `a=1`을 기록하고, CPU는 G의 다른 필드에 `b=2`를 기록한다.
3. 넓은 페이지 copy-back으로 D를 G에 복사하면 G가 `(a=1, b=0)`이 된다.
4. 이 뒤의 `snap_merge`는 이미 사라진 CPU 값 `b=2`를 현재 G에서 읽을 수 없다.

두 원래 쓰기는 서로 겹치지 않는다. **전체 페이지 복사가 덮어쓰기 문제를 추가한 것**이다. 복사 뒤 barrier를 넣거나 `g_done`을 늦게 공개해도 지워진 값을 복구하지는 못한다.

### 수정 계약

일반 경로는 다음 순서로 고정한다.

1. D에서 해당 제출에 귀속된 R로 GPU snapshot 복사.
2. GPU 복사 완료와 CPU 가시성 확보.
3. R의 GPU 세대, G의 CPU 세대, 풀의 allocation generation 확인.
4. 유효한 결과를 G에 병합.
5. 그 완료 범위의 `g_done` 공개.
6. CPU 병합까지 끝난 R만 재사용.

R은 GPU 복사 완료부터 CPU 병합 종료까지 내용이 바뀌지 않아야 한다. D의 다음 GPU 쓰기도 snapshot 복사가 D를 읽는 작업보다 앞질러 실행되면 안 된다.

`D → G import` 직접 복사는 실제 복사 범위 전체에서 충돌하는 CPU 접근이 없고, CPU 최신값을 덮지 않으며, 이전 세대가 새 세대를 덮지 않는다는 조건을 입증한 별도 경로에 한해 가능하다. **현재 문서의 “snap_merge 유지”는 그 입증이 아니다.**

CPU 페이지 보호나 `nvn_dirty`를 GPU 복사의 병합 장치로 취급해서는 안 된다. imported payload의 CPU/GPU 접근 동기화 책임은 애플리케이션에 있다. [S2]

## 5. P0 — A안의 barrier와 완료 순서

### 5.1 업로드

GPU 복사 명령을 기록했다고 원본이 즉시 snapshot으로 고정되는 것은 아니다. 다음 계약이 필요하다.

1. CPU 쓰기가 업로드 원본에 반영돼 있고 GPU copy에 보인다.
2. GPU copy가 원본 범위를 읽는 동안 충돌하는 CPU 쓰기·해제·재매핑이 없다.
3. D의 이전 접근과 이번 업로드의 덮어쓰기가 올바르게 순서화된다.
4. 업로드의 transfer write가 이후 실제 소비자에게 보인다.

제출 전에 완료된 host write는 필요한 캐시 처리를 마친 뒤 queue submission의 host→device 보장을 사용할 수 있다. 따라서 항상 별도의 HOST barrier를 추가해야 한다는 뜻은 아니다. 제출 뒤 원본을 다시 쓰는 경합은 이 보장으로 해결되지 않는다. [S3]

| 경계 | 필요한 범위 |
|---|---|
| 이전 D 사용 → D 업로드 | 실제 겹치는 이전 GPU 읽기·쓰기와 transfer write의 순서 |
| 업로드 copy → 정점 읽기 | `COPY / TRANSFER_WRITE` → `VERTEX_ATTRIBUTE_INPUT / VERTEX_ATTRIBUTE_READ` |
| 업로드 copy → 인덱스 읽기 | `COPY / TRANSFER_WRITE` → `INDEX_INPUT / INDEX_READ` |
| 업로드 copy → 셰이더 | 실제 shader stage의 uniform·sampled·storage read/write/atomic에 맞는 의존성 |
| 업로드 copy → indirect | `COPY / TRANSFER_WRITE` → `DRAW_INDIRECT / INDIRECT_COMMAND_READ` |
| 업로드 copy → 후속 copy/fill | 해당 transfer read/write에 맞는 의존성 |

표의 stage/access는 Vulkan synchronization2의 이름을 줄여 적었다. 실제 코드에서는 사용한 기능과 큐에 맞는 비트를 사용한다. 정점 입력용 barrier만 재사용하면 SSBO·indirect·raw BDA 경로를 모두 처리한 것이 아니다. [S3][S4]

### 5.2 다운로드

| 순서 | 요구 사항 |
|---|---|
| 1 | 해당 범위의 실제 GPU 생산자 쓰기 |
| 2 | 생산자의 stage/write access → `COPY / TRANSFER_READ` |
| 3 | D → R GPU 복사 |
| 4 | `COPY / TRANSFER_WRITE` → `HOST / HOST_READ` |
| 5 | 이 복사까지 포괄하는 timeline/fence signal과 host wait |
| 6 | R이 non-coherent mapped memory이면 필요한 범위를 invalidate |
| 7 | R과 G의 세대를 검증하고 CPU 병합 |
| 8 | `g_done` release 공개, 성공한 대기 경로에서 acquire 관측 |

Khronos는 GPU 쓰기, host-read 의존성, 완료 대기, 필요시 invalidate를 통한 가시성 확보를 구분한다. 같은 제출에 명령을 넣었다는 설명만으로 이 조건이 대체되지는 않는다. [S3][S5][S6]

**Bayo3의 문제 쓰기는 깊이 패스 안의 fragment SSBO store다.** 해당 버퍼의 생산자는 `FRAGMENT_SHADER / SHADER_STORAGE_WRITE`를 포함해야 한다. 깊이 attachment용 `EARLY/LATE_FRAGMENT_TESTS / DEPTH_STENCIL_ATTACHMENT_WRITE`만으로는 이 SSBO 쓰기를 표현하지 못한다. 이는 첨부 문서의 셰이더 설명을 전제로 한 barrier 판단이다. [S4]

다른 생산자는 compute, transform feedback, transfer 등 실제 쓰기 경로로 분류한다. Query 결과의 버퍼 복사는 transfer write로 동기화해야 한다. [S7]

첫 구현에서는 기존 큐 안에서 업로드·실행·다운로드를 순서화하는 편이 범위가 작다. 별도 transfer queue를 쓰려면 semaphore와 필요한 queue-family ownership transfer까지 추가한다. 복사 명령은 허용되는 render-pass 범위에서 기록한다. [S3][S8]

### 5.3 `g_done`의 정확한 의미

`g_done=N`은 GPU 실행만 끝났다는 값이 아니라 **N의 관측 대상 결과가 G에 반영됐다는 완료값**이어야 한다.

- `nvnSyncWait`, `nvnQueueFinish`, 성공한 완료 조회가 같은 의미의 토큰을 사용해야 한다.
- timeout/error를 완료 성공으로 취급하면 안 된다.
- 병합을 병렬화해 완료 순서가 바뀐다면, 단일 `g_done`은 필요한 하위 작업까지 끝난 연속 완료 구간을 나타내야 한다.
- 호스트 쓰기와 완료 공개는 release/acquire 또는 동등한 동기화 관계가 필요하다. 단순 코드 순서나 `volatile`만을 근거로 삼으면 안 된다. [S6]
- R의 재사용 기준은 GPU 완료뿐 아니라 **CPU 병합 완료**까지 포함한다.
- 풀 해제·주소 재사용 뒤 옛 다운로드가 새 풀에 반영되지 않도록 allocation generation을 확인한다.

이 조건을 기존 코드가 이미 만족한다면 그대로 재사용하면 된다. 설계서의 파일·줄 번호만으로 만족 여부를 확인할 수는 없다.

## 6. P0 — import를 업로드 원본으로 유지할 조건

**import를 전송 원본으로 유지하는 선택 자체는 타당하다.** 첨부 표에서는 import 복사가 빠른 조건을 보고했고, 문제의 일반 GPU 접근은 D로 옮긴다. 다만 전송 원본의 정확성 조건은 별도다.

### 원본 안정성과 수명

- GPU copy가 실제로 읽는 전체 범위를 기준으로 CPU 충돌을 막는다.
- 4 KB로 확대한 복사 범위에서 CPU가 다른 필드를 갱신하는 경우도 포함한다.
- 보호 fault가 나자마자 페이지를 writable로 풀어 CPU 쓰기를 허용한다면 원본 안정성을 보장한 것이 아니다.
- volatile 상태도 이 조건을 면제하지 않는다.
- import object가 살아 있는 동안 원본 메모리를 유효하게 유지하고, 진행 중인 복사가 끝나기 전에 해제·재매핑하지 않는다. [S2]

원본 안정성을 보장할 수 없는 경우에는 U에 안정된 snapshot을 확보해 복사한다. U를 만드는 CPU 복사 과정 역시 원본의 동시 변경을 다뤄야 한다.

### 원래 host pointer와 Vulkan mapped pointer의 구분

외부 host pointer를 import했다고 해서 그 원래 포인터가 Vulkan의 host-mapped pointer로 취급되는 것은 아니다.

- 원래 게스트 포인터를 통한 접근의 host cache 처리는 플랫폼 동기화 계약으로 확인해야 한다.
- `vkFlushMappedMemoryRanges`와 `vkInvalidateMappedMemoryRanges`는 Vulkan에서 매핑한 포인터를 통한 접근에 적용하는 규칙이다.
- 원래 게스트 포인터에 대한 캐시 가시성을 “Vulkan flush를 호출했으므로 해결”이라고 단정하면 안 된다. [S9]

실제로 선택한 import memory type과 플랫폼의 CPU/GPU 일관성 조건을 연결해서 적어야 한다. NCE가 계속 접근할 원본을 host-visible이 아닌 memory type으로 import하는 경우에는 별도의 명세상 제한이 있으므로, 기존 할당 코드가 어떤 타입을 쓰는지 확인 대상이다. [S2]

이 조건을 증명하지 못하는 경로는 driver-owned mapped U를 사용해 Vulkan의 명시된 flush 규칙으로 처리한다. R도 driver-owned mapped staging으로 두면 읽는 포인터와 invalidate 대상이 명확해진다. [S5][S10]

## 7. P0 — 4 KB dirty와 volatile만으로 부분 갱신을 보존할 수 없다

### 7.1 페이지 전체 업로드의 전제

4 KB는 변경 감지 단위다. 그 페이지의 모든 바이트가 CPU에서 최신이라는 뜻이 아니다.

CPU가 G의 한 필드를 바꿨고 GPU가 D의 다른 필드를 바꿨다면, G 전체 페이지를 D로 올려서는 안 된다. 먼저 필요한 GPU 결과를 안전하게 합쳐 페이지 전체가 최신이 되게 하거나, 정확한 범위와 세대에 따라 각각의 갱신을 반영해야 한다.

기존 `mtl_wait(dependency)`가 이를 해결한다면, **어떤 페이지의 어떤 GPU 작업과 copy-back 완료까지 기다리는지**를 명시해야 한다. 단순히 앞에 wait가 있다는 설명으로는 부족하다.

초기 volatile 적용 범위는 **CPU가 갱신하고 GPU는 읽기만 하는 페이지**로 제한하는 것이 명확하다. GPU 쓰기 대상으로 바뀌는 페이지는 소유권을 다시 정한 뒤 정책을 전환한다. 이는 풀 사본 크기를 줄이는 정책이 아니라 같은 페이지의 갱신을 보존하는 규칙이다.

### 7.2 dirty 해제와 재보호

“페이지를 올린 뒤 다시 보호”라는 문장에는 CPU 쓰기 누락을 막는 절차가 빠져 있다.

- 복사에 사용한 CPU generation을 기록한다.
- 복사 중 또는 직후에 발생한 새 CPU 쓰기는 dirty로 남긴다.
- 관측한 세대가 변했는데도 무조건 clean으로 바꾸지 않는다.
- GPU 업로드의 명령 기록 시점과 실제 원본 읽기 완료 시점을 구분한다.
- 내부 copy-back 쓰기와 게스트 CPU 쓰기를 구분한다.
- 내부 병합을 위해 보호를 잠시 풀 때, 다른 게스트 스레드의 쓰기나 불완전한 값 읽기를 허용하지 않도록 한다.

세대 스탬프가 존재한다는 사실보다 **수집·보호·snapshot·clear의 원자성과 순서**가 중요하다.

### 7.3 `snap_merge`의 보장 범위

`snap_merge` 본문이 없으므로 안전하다고도, 잘못됐다고도 확정할 수 없다. 다만 다음을 검증해야 한다.

- 값이 snapshot과 같다고 해서 CPU가 쓰지 않았다고 판정하지 않는다. 같은 값 재쓰기나 A→B→A 이력은 값 비교만으로 복원되지 않는다.
- 실제 CPU/GPU 접근이 서로 다른 바이트인데 32비트 단어 단위로 한쪽을 선택하면 갱신을 잃지 않는지 확인한다.
- 이전 다운로드가 이후의 합법적인 CPU 쓰기를 덮지 않는다.
- 이전 다운로드 커밋으로 더 새로운 GPU dirty 세대를 지우지 않는다.

이는 정의되지 않은 동일 주소 동시 쓰기에 정답을 만들라는 요구가 아니다. **지원해야 할 순서와 비중첩 접근을 구현의 병합 단위 때문에 손상시키지 말라는 요구**다. [S1]

정확한 쓰기 범위를 모른다면 페이지의 소유권을 넘기기 전에 최신 내용을 회수하는 등 정보 부족을 해결할 규칙이 필요하다. 단순 page compare만으로 모든 경우를 해결한다고 적어서는 안 된다.

### 7.4 첫 초기화

새 D의 첫 GPU 읽기 전에 기존 G 내용이 반영돼야 한다. 보호 등록 이후 새로 dirty가 된 페이지만 올리는 구현으로는 이미 존재하던 초기 데이터를 놓칠 수 있다.

SSBO atomic은 기존 값을 읽는다. writable SSBO라는 이유만으로 전체 덮어쓰기가 보장되지 않는다. 업로드를 생략할 수 있는 범위는 정확한 full overwrite가 증명된 경우로 한정한다.

## 8. P0 — alias·bridge·BDA와 쓰기 범위의 완전성

### 8.1 같은 메모리의 GPU 저장소는 하나여야 한다

같은 게스트 메모리를 포함하는 두 pool/view가 별도 D를 가지면, 한쪽 SSBO 쓰기를 다른 쪽 BDA가 보지 못한다. copy-back을 나중에 수행해도 같은 제출 안의 GPU→GPU 관측을 복원할 수 없다. atomic도 서로 다른 사본에 분산되면 같은 메모리에 대한 연산이 아니다.

필요한 계약은 다음과 같다.

- 게스트 주소와 실제 메모리 별칭 관계를 기준으로 canonical backing과 allocation generation을 정한다.
- `pool_rank`, 중첩 풀, s48 range, bridge는 그 저장소의 view 또는 명시적으로 동기화되는 표현이어야 한다.
- 버퍼·descriptor·BDA가 모두 같은 주소를 같은 최신 저장소로 해석한다.
- backing 확장·재할당 시 GPU 최신 데이터를 보존한다.
- 이전 제출이 사용하는 버퍼·descriptor·BDA는 해당 사용이 끝날 때까지 살아 있어야 한다.

별도 사본 두 개를 만들고 barrier만 넣는 것으로는 데이터가 서로 전달되지 않는다. 또한 일반 non-sparse `VkBuffer`의 backing은 하나의 메모리 객체에 연속으로 바인딩돼야 하므로, 서로 다른 chunk를 BDA 숫자 계산만으로 연속 버퍼처럼 취급할 수 없다. [S1][S8]

**R4는 모든 풀을 사본화하기 전에 표현을 확정해야 하는 항목이다.**

### 8.2 접근 경로 점검표

아래는 구현에서 대조할 경로다. 첨부 문장만으로 모두 연결됐다고 판정하지 않았다.

| 경로 | 업로드·읽기 측면 | 쓰기·되돌림 측면 |
|---|---|---|
| vertex/index | 실제 바인딩 범위와 D 선택 | GPU가 생성한 같은 범위의 최신값 유지 |
| UBO/texel/SSBO | stage·범위·RMW 구분 | storage write와 atomic의 잠재 쓰기 범위 |
| indirect/count/conditional | 명령·count·predicate 버퍼까지 추적 | GPU 생성 결과가 같은 D에 있어야 함 |
| XFB | counter를 읽는 경로 포함 | 출력과 counter 쓰기 포함 |
| 게임 copy/fill/update | source와 기존값이 필요한 destination | 실제 destination 및 부분 갱신 |
| query/event/counter | CPU/GPU 관측 경로 구분 | 데이터와 완료 표시의 범위·순서 |
| 이미지↔버퍼, s56 | 최신 이미지/버퍼 저장소 연결 | 변환·복사까지 포함한 완료와 dirty |
| raw global BDA | 주소 테이블도 D를 가리킴 | 임의 쓰기가 도달할 잠재 범위를 누락하지 않음 |

raw BDA는 업로드의 전체 dirty fallback만 있으면 끝나는 것이 아니다. 쓰기 대상의 정확한 범위를 모르면 접근 가능한 풀 집합에 대해 보수적인 GPU dirty 범위를 기록해야 한다.

`gw[]`가 넘칠 때 범위를 합쳐 확대하는 것은 누락보다 안전하다. 항목을 버리면 안 된다. 구간 트리나 bitmap 도입 자체는 필수 조건이 아니며, **누락 없는 범위 보존**이 필수 조건이다.

## 9. P0 — CPU 관측 경로는 fence 두 개로 끝나지 않는다

“펜스 없이 읽으면 Switch에서도 경합이라 정답이 없다”는 문장은 삭제해야 한다. fence API 호출 여부만으로 해당 접근의 동기화 계약을 판정할 수 없다.

이벤트, counter, query availability, 완료 조회가 동기화 역할을 할 수 있다. 반대로 raw-memory polling이 모두 합법적이라는 뜻도 아니다. NVN의 해당 기능·풀 속성·접근 방법을 확인해야 한다. Vulkan에도 별도 fence를 전제로 하지 않는 query 결과 조회가 존재하지만, 그것이 NVN raw polling의 의미를 대신 정해 주지는 않는다. [S11]

| 관측 경로 | 반드시 확인할 조건 |
|---|---|
| SyncWait / QueueFinish 성공 | 해당 토큰의 결과 반영이 성공 반환보다 앞섬 |
| 이벤트·counter·query availability | 결과와 완료 표시의 세대가 맞고, 완료만 먼저 보이지 않음 |
| CPU가 기록한 이벤트를 GPU가 읽음 | G의 기록이 D의 관측에 전달됨 |
| s56 렌더 타깃 착지 | 마지막 image→buffer·변환·복사까지 완료 토큰에 포함 |
| 직접 CPU load | 어떤 기존 동기화가 G의 최신값을 보장하는지 명시 |
| GPU/CPU가 중간 신호를 주고받음 | 제출 끝 copy-back만 기다려도 진행 가능한지 확인 |

GPU가 중간 신호를 내고 CPU의 응답을 기다리는 경로가 있다면, 제출 종료까지 신호를 G에 전달하지 않는 방식은 진행을 막을 수 있다. 해당 경로의 존재는 코드에서 확인해야 하며, 존재한다면 중간 관측 지점의 처리가 필요하다.

### NCE read fault를 사용할 경우

read fault는 단순 성능 최적화가 아니다. 필요한 정상 CPU 관측 경로를 다른 방법으로 보장하지 못한다면 정확성 기능이다.

- NCE의 native load가 Eden의 `OnCPURead`를 자동으로 호출하지는 않는다.
- Windows의 읽기 차단 보호는 쓰기도 함께 다룬다. `PAGE_GUARD`는 일회성 경보이므로 지속적인 읽기 추적의 대체물이 아니다. [S12]
- `nvn_dirty`, GPU-dirty 읽기 보호, volatile 상태를 한 보호 정책으로 조정해야 한다.
- 완료 스레드 자체의 복사·비교가 fault를 내거나 자기 `g_done`을 기다리는 재진입을 막아야 한다.
- VEH에서 필요한 락을 가진 채 완료를 기다리면 완료 스레드가 같은 락을 필요로 하는 교착이 생길 수 있다. Microsoft도 VEH 안에서 동기화 객체 획득이나 메모리 할당을 하는 호출을 피하도록 안내한다. [S13]
- `tebfix` 적용 여부와 이 새 보호·교착 계약의 정확성은 별개다.

현재 관측 경로가 기존 API 처리로 모두 보장된다면 read fault를 당장 도입할 이유는 없다. 필요한 경로가 빠진 채 전체 copy를 기본으로 켜는 것은 피해야 한다.

## 10. Eden 재사용과 vbmir/pulling 처리

### Eden BufferCache

문서가 설명한 주소 체계, Maxwell 레지스터 의존, 보호 소유자, 스케줄러 차이를 전제로 하면 **클래스를 그대로 연결하기보다 메커니즘을 선택해 옮기는 판단이 타당하다.** “어떤 리팩터링을 해도 클래스 재사용은 불가능하다”는 뜻으로 확장할 필요는 없다.

다음은 정확성 계약으로 1단계에 반영해야 한다.

- GPU-modified 범위와 제출별 snapshot의 관계.
- 이미 다운로드 중인 세대와 새 GPU 쓰기 세대의 구분.
- 새 CPU 쓰기를 오래된 다운로드가 덮지 않는 규칙.
- 풀 해제·재사용 후 오래된 다운로드의 커밋 방지.

기존 copied-pool 구현이 이를 제공한다면 재사용하면 된다. 비동기 다운로드의 처리량 최적화는 뒤로 미룰 수 있지만, 세대·수명·최신값 보존까지 최적화로 분류해서 미루면 안 된다.

### vbmir / vertex pulling

게임 GPU 접근이 D로 통일되면 별도 정점 미러와 import 회피용 정점 풀링은 필요하지 않다.

다만 “`zero_copy` 조건이 있으므로 자동으로 꺼진다”는 설명 대신 아래를 확인해야 한다.

- 실제 VB/IB, descriptor, raw BDA가 D를 가리킴.
- 전송용 import가 남아 있어도 vbmir/pulling 대상으로 오인하지 않음.
- import 사유 fallback이 게임 실행 버퍼에 남지 않음.
- import 모드로 별도 실행하면 기존 경로가 정상 복원됨.

이 확인에는 셰이더 재작성이나 풀링 기능 확장이 필요하지 않다.

## 11. 권장 구현 단계 — A안으로 갱신

### 0단계: 정확성 계약 대조

기존 코드에서 다음만 먼저 연결한다.

- `vk_buf_at`·BDA·bridge의 실제 저장소 선택.
- CPU dirty 수집·재보호·clear 순서.
- `gw[]`의 누락 없는 범위 기록.
- 완료 토큰과 SyncWait/QueueFinish/event/query/s56의 관계.
- `snap_merge`가 보장하는 세대와 부분 갱신 범위.

기존 코드가 이미 제공하는 기능을 다시 만들 필요는 없다. 없는 계약을 “기존 경로라 정확하다”는 문장으로 대체하지 않으면 된다.

### 1단계: 명시적 copy 모드의 A 기준 구현

- GPU가 사용하는 풀의 D를 device-local로 만든다.
- `copy` 실행 옵션으로 먼저 비교하고, 자동 기본 전환은 검증 뒤 수행한다.
- **모든 dirty 페이지를 대상으로 증분 업로드**한다. 바인딩별 지연 사용 수집은 아직 넣지 않는다.
- import upload source는 안정성·가시성·수명 조건이 충족된 범위에서 사용하고, 필요한 경우 U를 사용한다.
- D→R snapshot, 완료 대기, 필요시 invalidate, 세대 검증 병합, `g_done` 공개를 구현한다.
- alias·bridge·BDA와 CPU 관측의 필수 경로를 함께 연결한다.
- GPU도 쓰는 페이지에 기존 volatile 정책을 무조건 적용하지 않는다.

이 단계에서 완료 스레드의 copy-back 입력은 mapped B가 아니라 **R과 해당 제출의 메타데이터**로 바뀐다. `pools_copy_back`을 그대로 호출한다는 문장만으로는 구현 변경 범위를 설명하지 못한다.

### 2단계: 실제 게임 A/B

HWAoI와 Bayonetta 3의 같은 장면·해상도·빌드 조건에서 import와 copy를 비교한다. 문제 패스, 전체 GPU 시간, CPU 동기화 비용, 결과 정확성을 함께 확인한다.

### 3단계: 관측된 병목만 최적화

필요한 경우에만 다음을 추가한다.

- 바인딩 범위 지연 업로드.
- 다운로드 범위 세분화.
- 더 많은 제출을 겹치는 비동기 처리.
- GPU dirty bitmap.

정확성 조건을 충족하고 실제 비용을 확인한 뒤 N1X 자동 정책을 copy로 바꾼다. 취소한 동일 주소 픽셀 벤치나 DK 실행을 다시 요구할 필요는 없다.

## 12. R1과 검증: 정확한 쓰기 페이지 추적은 처음부터 필수는 아니다

보수적인 `gw[]` 범위가 정확하다면, 그 범위 전체를 R로 복사하고 안정된 snapshot을 비교하는 방식으로 먼저 시작할 수 있다.

단, page compare는 G에 실제로 쓰는 양을 줄일 뿐이다. D→R 복사량과 비교를 위해 읽는 양을 자동으로 줄이지는 않는다.

30 MB를 프레임마다 한 번 검사하면 60 fps에서 검사 대상 크기만 1.8 GB/s다. 실제 메모리 트래픽은 비교 배열, snapshot, cache 동작에 따라 달라진다. `gw[]`가 제출 단위라면 같은 범위를 프레임 안에서 여러 번 처리하는지도 집계해야 한다.

### 필요한 로그

10초 주기 집계는 유지하되 다음 항목을 나눠 기록한다.

- 제출 수/프레임.
- 업로드 bytes와 GPU copy 시간, CPU 준비·대기 시간.
- D→R 다운로드 bytes.
- CPU 병합에서 검사한 bytes, 실제 G에 쓴 bytes, 병합 시간.
- GPU 완료부터 `g_done` 공개까지 지연.
- fault/s와 fault 처리 시간, volatile 페이지 수.
- 전체 프레임 p50/p95, 전체 GPU 시간, 문제 패스 GPU 시간.

업로드·다운로드 GPU 시간을 원래 패스와 따로 측정해야 “패스만 빨라지고 프레임 비용은 다른 곳으로 이동한 경우”를 구분할 수 있다.

### 고정보량 정확성 점검

다음은 성능 마이크로벤치 요구가 아니라, 캐시가 손상시키면 안 되는 상태 전이의 검증 항목이다.

| 점검 | 합격 조건 |
|---|---|
| 같은 페이지의 CPU·GPU 비중첩 갱신 | 두 변경 모두 보존 |
| snapshot 뒤 새 CPU 쓰기 | 이전 다운로드가 새 쓰기를 덮지 않음 |
| 연속 제출의 같은 범위 GPU 쓰기 | 이전 커밋이 새 GPU dirty를 지우지 않음 |
| 중첩 풀·bridge·raw BDA 교차 사용 | 동일 주소의 최신 결과와 atomic 저장소가 일치 |
| query/event/counter/s56 완료 관측 | 완료가 보일 때 해당 결과도 올바른 세대 |
| 풀 해제 후 주소 재사용 | 이전 제출이 새 풀에 쓰거나 병합하지 않음 |
| 업로드·병합 중 CPU fault | lost write, 재진입, 교착 없이 정해진 소유권 전환 |
| import source 또는 staging 재사용 | 실제 GPU/CPU 소비가 끝나기 전 내용을 바꾸지 않음 |

기존 게임 실행이나 작은 캐시 상태 검증에서 필요한 항목을 확인하면 된다. 정상 화면 몇 분만으로 위 조건이 모두 증명됐다고 처리하지 않는다.

### 게임별 판정

- **HWAoI:** 60 fps 상한뿐 아니라 GPU 시간·프레임 p95·업로드/병합 비용과 화면을 확인한다.
- **Bayonetta 3:** 같은 깊이 패스의 시간과 오브젝트 가시성 결과를 확인한다. “40~57 ms → 수 ms”는 목표이며 보장값이 아니다.
- 다른 색상 패스와 draw 수를 비교한 것만으로 해당 SSBO store의 비용이 확정되지는 않는다. 실제 같은 작업의 import/copy 비교를 근거로 삼는다.
- 전체 풀을 전환하는 A/B는 메모리 정책의 효과를 검증한다. 단일 store나 드라이버 내부 원인만 분리해 증명하는 실험은 아니다.

## 13. 문서에 바로 반영할 정정

| 위치 | 수정 |
|---|---|
| §0 “모든 풀을 사본으로 운용해야 한다” | 이번 N1X 정책으로 채택한다고 표현. 유일한 가능한 해결책이라는 단정은 제거 |
| §1 “반복 읽기: 캐시 안 됨” | “반복 읽기 벤치에서 35.9배 차이”로 한정. 내부 캐시 정책 확정과 구분 |
| §1 “셰이더 load 정상” | 정상으로 측정된 접근 패턴으로 한정. 반복 읽기 결과와 구분 |
| §1 드라이버 할당이면 모두 정상 | exportable/재import 경로와 일반 할당을 분리. 관측 범위를 넘어 일반화하지 않음 |
| §1 고정 기능 “20~250배” | 같은 행의 202.8/12.1은 약 16.8배, 3633/29.7은 약 122.3배다. 범위의 근거 측정과 예시 수치를 구분 |
| §1 “게스트 주소 배치 경로가 없다” | 조사한 현재 경로의 제약으로 한정. map_memory_placed 미지원만으로 모든 재설계를 불가능하다고 단정하지 않음. [S14] |
| §4.1 import 금지 | 일반 GPU 실행은 D, import는 허용된 전송용 endpoint로 역할 구분 |
| §4.2 직접 import 되돌림 | 기본 D→R→CPU 병합→g_done으로 변경 |
| §4.3 올린 뒤 다시 보호 | 세대·원본 안정성·재보호·clear 계약 추가 |
| §4.4 기다리면 항상 정답 | 가시성·완료 토큰·병합·release/acquire 조건을 함께 명시 |
| §4.4 fence 없는 읽기 일괄 경합 취급 | 실제 NVN 관측 계약에 따라 분류 |
| §4.5 CPU memcpy 업로드 | A의 GPU copy 업로드와 필요한 staging CPU 준비로 갱신 |
| §4.5 실제 쓴 페이지만 되돌려야 함 | 보수적 범위로 정확성 확보 후 비용이 큰 경우 세분화 |
| §6 “1단계 B / 2단계 A” | 처음부터 A로 갱신 |
| §0의 “9절”, §3의 메모리 타입 “5절”, vbmir “9절” | 실제 관련 절인 §5, §4.2, §5로 교정 |
| §4.1 Turnip도 import 기본이라는 표현 | §2의 “Turnip은 external_memory_host가 없어 copied pool” 설명과 맞춰 기존 장치별 경로 유지로 정정 |

`VirtualProtect 약 1 ms/페이지`는 코드 주석이나 특정 관측의 값으로 표시한다. 모든 fault의 고정 비용으로 사용하지 않는다.

### 반영용 결정문

> N1X에서 게임 GPU 명령의 버퍼 접근은 device-local 사본으로 통일한다. 기존 게스트 주소 공간은 유지한다. import 버퍼는 안전 조건이 확인된 업로드 전송 원본으로 제한해 사용할 수 있다. 다운로드는 제출별 readback staging에 기록하고, 실제 생산자→copy→host 의존성, 완료 대기, 필요한 invalidate, 세대 보존 병합을 마친 뒤 g_done을 공개한다. 4 KB dirty 감지는 데이터 소유권과 구분하며, 겹치는 풀·bridge·BDA는 같은 최신 GPU 저장소를 사용한다. 1단계는 모든 dirty 페이지의 증분 업로드로 시작하고, 바인딩 범위 지연 업로드와 쓰기 범위 세분화는 관측된 비용에 따라 추가한다.

## 14. 검수 요청 8개 항목에 대한 답

| 요청 | 답 |
|---|---|
| 1. A의 복사·barrier·import 유지 | A 유지. import upload source는 조건부 가능. 일반 readback은 별도 staging. §4–6 |
| 2. g_done·polling·event·query·s56 | copy-back 뒤 공개만으로 충분하지 않음. 토큰·가시성·관측 계약 필요. §5, §9 |
| 3. 업로드 누락 R3 | 1차는 모든 dirty 페이지 대상. 지연 업로드는 전체 접근 지점 대조 뒤. §8, §11 |
| 4. 되돌림 양 R1 | 보수적 범위로 시작 가능. 실제 복사·검사·병합 비용을 분리해 최적화 결정. §12 |
| 5. Eden 이식 | 메커니즘 선별 재사용 타당. 최신 쓰기 보존과 세대는 1단계 정확성 계약. §10 |
| 6. vbmir/pulling/bridge | 별도 미러·풀링 불필요. 실제 D 바인딩 확인. bridge는 적용 전 해결. §8, §10 |
| 7. fault 비용 R6/R8 | volatile 범위 제한과 단일 보호 정책, lost-write·재진입·교착 계약 필요. §7, §9 |
| 8. 단계·검증 | A로 단계 갱신, 먼저 정확한 기준 경로, 실제 게임 비용과 상태 전이 검증. §11–12 |

## 15. 근거

장치 수치와 프로젝트 함수 설명의 출처는 첨부된 수정본 두 문서다. 아래 공식 자료는 Vulkan·Windows의 일반 규칙을 뒷받침하며, 프로젝트의 특정 구현이나 측정값을 대신 검증하지 않는다.

- [S1] Vulkan Resource Creation: Memory Aliasing / Resource Memory Overlap / Resource Memory Association.
- [S2] VkImportMemoryHostPointerInfoEXT: payload 공유, 동기화·수명 책임, non-host-visible import의 host 접근 제한.
- [S3] Vulkan Guide: Synchronization Examples, upload와 CPU readback.
- [S4] VkAccessFlagBits2: shader storage·transfer·host·vertex/index 접근 구분.
- [S5] vkInvalidateMappedMemoryRanges: host 가시성, 완료 대기, non-coherent 범위.
- [S6] Microsoft: Synchronization and Multiprocessor Issues.
- [S7] vkCmdCopyQueryPoolResults: 결과·availability와 transfer-write 동기화.
- [S8] vkCmdCopyBuffer: 복사 범위·usage·backing·명령 사용 조건.
- [S9] Vulkan Memory Allocation: imported 원본 host pointer와 Vulkan mapped pointer 구분.
- [S10] vkFlushMappedMemoryRanges: host write 가시성과 non-coherent 범위.
- [S11] vkGetQueryPoolResults: 결과 및 availability 조회.
- [S12] Microsoft Memory Protection Constants.
- [S13] Microsoft PVECTORED_EXCEPTION_HANDLER.
- [S14] VkPhysicalDeviceMapMemoryPlacedFeaturesEXT: 지정 주소 매핑 기능의 의미.

[S1]: https://docs.vulkan.org/spec/latest/chapters/resources.html
[S2]: https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryHostPointerInfoEXT.html
[S3]: https://docs.vulkan.org/guide/latest/synchronization_examples.html
[S4]: https://docs.vulkan.org/refpages/latest/refpages/source/VkAccessFlagBits2.html
[S5]: https://docs.vulkan.org/refpages/latest/refpages/source/vkInvalidateMappedMemoryRanges.html
[S6]: https://learn.microsoft.com/en-us/windows/win32/sync/synchronization-and-multiprocessor-issues
[S7]: https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyQueryPoolResults.html
[S8]: https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyBuffer.html
[S9]: https://docs.vulkan.org/spec/latest/chapters/memory.html
[S10]: https://docs.vulkan.org/refpages/latest/refpages/source/vkFlushMappedMemoryRanges.html
[S11]: https://docs.vulkan.org/refpages/latest/refpages/source/vkGetQueryPoolResults.html
[S12]: https://learn.microsoft.com/en-us/windows/win32/memory/memory-protection-constants
[S13]: https://learn.microsoft.com/en-us/windows/win32/api/winnt/nc-winnt-pvectored_exception_handler
[S14]: https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMapMemoryPlacedFeaturesEXT.html

