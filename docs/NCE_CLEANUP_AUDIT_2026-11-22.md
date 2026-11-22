# NCE 백엔드 클린업 사전 점검 보고서

2026-11-22 · NXN1X 로컬 소스 스냅샷 · 수정 전 기준

**첫 정리 대상으로 권하는 범위는 패처의 미사용 private 오버로드 9개다.** 호출부와 Clang의 오버로드 선택 결과가 일치한다. 비Windows 구현의 파일 내부 상수 4개도 미사용 근거가 분명하지만, 해당 플랫폼을 빌드할 때 별도로 다루는 편이 검증하기 쉽다. Windows 실행 경로에서 확인한 중복 중에는 명령어 인코딩과 스레드 중단에 관여하는 코드가 있다. 이들은 첫 변경에서 얻는 이점보다 검증 비용이 크다.

## 1. 미사용 코드 후보와 판단 근거

아래에서 파일명만 표시한 위치는 `source/src/core/arm/nce/` 기준이다. 줄 번호는 점검일의 소스를 가리킨다.

| 대상 | 확인한 증거 | 판단 |
|---|---|---|
| 기본 코드 생성기를 쓰는 private 래퍼 9개 · `patcher.h:111–121` | 아래 표의 짧은 시그니처를 선택하는 호출이나 주소 참조를 찾지 못했다. Windows 진단 끔·켬 모두 `patcher.cpp`의 Clang AST에서 9개에 `used` 표시가 없었다. 같은 블록의 실제 사용 래퍼 2개에는 표시가 있다. | **첫 삭제 후보.** 명시적으로 코드 생성기를 받는 구현과 호출부는 유지한다. |
| 파일 내부 상수 4개 · `interpreter_visitor.cpp:15–17`, `arm_nce.cpp:406` | `CACHE_LINE_SIZE`, `PREFETCH_STRIDE`, `SIMD_PREFETCH_THRESHOLD`, `CACHE_PAGE_SIZE`는 해당 번역 단위에서 각각 정의만 있다. 모두 상수 초기화이며, 마지막 상수 뒤의 `ClearInstructionCache()`는 고정된 배리어 명령을 사용한다. | **삭제 근거 높음.** Android/Linux ARM64 대상이므로 해당 구성에서 컴파일할 수 있을 때 정리한다. |
| 패치 캐시 키 타입과 해시 · `patcher.h:22–38` | `PatchCacheKey`는 타입 정의, 기본 비교 연산자, `std::hash` 특수화 안에서만 발견됐다. 현재 소스에서 이 키로 캐시를 선언하거나 조회하는 곳을 찾지 못했다. | **별도 삭제 후보.** 헤더에 공개된 두 정의를 한 쌍으로 검토한다. 바로 위의 `ModuleID`는 아래 setter의 타입이다. |
| 쓰기만 하는 패처의 모듈 ID · `patcher.h:66–68,87` | `SetModuleID`는 실제 호출된다. NSO 로더 두 곳과 동적 NRO 경로 한 곳에서 호출하지만, 저장된 `Patcher::module_id`를 읽는 코드는 찾지 못했다. 위치는 `source/src/core/loader/nso.cpp:660,670`, `source/src/core/hle/service/ro/ro_nce_windows.cpp:117`이다. | **보류.** 제거하면 공개 메서드와 객체 레이아웃, 로더 호출부가 함께 바뀐다. RO 서비스 자체는 모듈 ID로 중복 로딩을 확인한다(`ro.cpp:267`). |
| 중단 전달용 스택 경계 조회 2개 · `windows_cross_thread_break.h:789–790` | `DeliveryHostStackLow/High()`의 호출을 찾지 못했다. 두 값의 스냅샷은 같은 헤더 665–666행에서 여전히 기록한다. 인접한 `DeliveryTeb()`는 `arm_nce_windows.cpp:754,855`에서 실제로 읽는다. | **접근자만 미사용 후보. 동기화 변경은 보류.** 원본 경계는 `IsHostStackPointer()`와 `windows_nce_transition.cpp:179,195`가 소비한다. 두 접근자의 검색 결과를 근거로 스택 경계나 스냅샷 전체를 삭제하면 안 된다. |

### 패처에서 정확히 어느 오버로드가 대상인가

| 후보 선언 | 호출되는 구현을 확인한 위치 |
|---|---|
| `LockContext()`, `UnlockContext()`, `UnlockContextFromGuestContext()` | `patcher.cpp:941,1677,1770,1800`에서 모두 생성기 `cg`를 넘긴다. 마지막 래퍼는 Windows 전용이다. |
| `WriteSvcTrampoline(dest, svc)` | `patcher.cpp:1080,1082`에서 생성기와 해당 생성기의 저장·복원 레이블을 명시한다. |
| `WriteMrsHandler(dest, reg, sysreg)`, `WriteMsrHandler(dest, reg)` | `patcher.cpp:1097,1099,1166,1168`에서 생성기 `c_pre` 또는 `c`까지 넘긴다. |
| `WriteCtrEl0Handler(dest, reg)`, `WriteCntfrqHandler(dest, reg)`, `WriteCntpctHandler(dest, reg)` | `patcher.cpp:1111,1113,1142,1144,1154,1156`에서 생성기까지 넘긴다. |

같은 블록의 **`WriteSaveContext()`와 `WriteLoadContext()`는 유지 대상**이다. 패처 생성자가 각각 `patcher.cpp:258,264`에서 호출한다. 인접한 `c_pre` 오버로드도 별도 버퍼에 코드를 생성한다. 블록 전체를 지우거나 두 버퍼를 하나로 취급하면 실제 생성 경로가 바뀐다.

## 2. 중복 구현과 통합 판단

| 중복 | 위치와 의미 | 판단 및 검증 기준 |
|---|---|---|
| 로그 샘플링 조건 | `nce_write_step.h:86`, `x18_host_fpcr.h:76`의 `Sample(u64)`는 첫 4회와 이후 2의 거듭제곱에서 기록하는 같은 식이다. 입력 0도 참이다. | **통합 검토 가능.** 조건식만 공유할 수 있다. 두 모듈의 `Get()`이 반환하는 카운터와 활성화 조건은 독립적이다. 입력 0, 1, 4, 5, 8에서 결과와 기존 로그 빈도를 비교한다. |
| 단조 시계의 나노초 변환 | `windows_cross_thread_break.h:247`의 `BreakRescue::SteadyNs()`와 `windows_nce_stats.h:99`의 `Stats::SteadyNanos()`는 시계, 변환, 반환 타입이 같다. 전자는 중단 대기를 측정하고 후자는 통계를 집계한다. | **우선순위 낮음.** 통합 시 원시 시각 함수의 의미를 유지해야 한다. 통계가 꺼지면 0을 반환하는 `Stats::Now()`를 중단 대기 경로에 쓰면 동작이 달라진다. |
| AArch64 명령의 5비트 레지스터 필드 치환 | `patcher.cpp:75`, `windows_x18_lse.h:39`, `instructions.h:235` 세 곳이 같은 마스크와 이동 연산을 쓴다. 앞의 두 함수는 본문이 같고 세 번째는 입력 매개변수 이름이 다르다. 각각 비트필드, LSE 원자 명령, exclusive 명령에서 실제 사용한다. | **실행 경로이므로 보류.** 통합한다면 순수 비트 치환만 대상으로 삼고, 기존 `static_assert`와 함께 각 경로가 생성하는 명령어 바이트를 전후 비교한다. exclusive의 `AsOrdered()` 처리와 각 호출자가 결정하는 피연산자 위치를 보존한다. |
| CASP 명령 판별식 | `x18_classifier.cpp:52`, `windows_x18_exclusive.h:31–33`, `windows_x18_lse.h:54–55`가 같은 마스크와 패턴을 검사한다. exclusive 경로에서는 이 조건에 맞는 명령을 **제외**하고, LSE 경로에서는 이를 **선택**한다. | **분기 통합 보류.** 판별식만 공유할 여지는 있다. 양성·음성 명령에 대해 분류 결과와 최종 패치 경로를 모두 비교해야 한다. |

## 3. 삭제 시 기능이 바뀌는 유사 코드

- **TLS 슬롯.** `current_nce_context.h:46,77`의 `TebSlotIndex()`와 `S55SlotIndex()`는 본문이 같아도 각 함수의 지역 정적 변수가 서로 다른 `TlsAlloc()` 결과를 보관한다. 전자는 NCE 실행 파라미터, 후자는 NVN 호출 상태용이다. `source/src/yuzu/main.cpp:102`가 두 슬롯을 일찍 예약한다. 하나의 정적 슬롯으로 합치면 두 값이 충돌한다.
- **어셈블리와 간접 호출.** Windows의 `RunThread`는 핸들러 유무에 따라 두 어셈블리 진입점 중 하나를 호출한다(`arm_nce_windows.cpp:2301–2306`). 예외 핸들러는 2099행에서 Windows에 등록하고, 중단 변환과 커밋 함수는 3042–3043행에서 함수 포인터로 전달한다. `windows_nce_transition.cpp:40–50`의 오프셋 검사는 어셈블리와 C++ 컨텍스트 레이아웃의 계약이다.
- **진단 호출.** `EpochGuard::Enter/Exit`는 `source/src/core/hle/kernel/physical_core.cpp:120,135`에 호출자가 있다. `R2a::NoteFrame`은 `source/src/core/perf_stats.cpp:89`가 호출한다. Release에서 빠지거나 NCE 디렉터리 안에 호출자가 없어도 사용되는 코드다.

## 4. 점검 범위와 검증 결과

- `source/src/core/arm/nce/`의 40개 파일과 `source/src/core/hle/service/ro/ro_nce_windows.{cpp,h}`를 기준으로 조사했다. CMake의 소스 선택, 커널 CPU 생성, NSO/NRO 로더, 프런트엔드 초기화, 진단 스크립트로 참조를 확장했다. 백업·참조용 복제본과 과거 바이너리는 현행 호출의 근거에서 제외했다.
- `source/CMakeLists.txt:301–303`은 ARM64 Android, Linux, Windows에서 NCE를 활성화한다. `source/src/core/CMakeLists.txt:1254–1335`가 Windows 진입 어셈블리와 비Windows 구현을 나눈다. 실제 CPU 생성에는 애플리케이션 프로세스와 NCE 설정이 모두 필요하다(`k_process.cpp:1322–1324`).
- **Windows ARM64 구문 검사 18/18 통과.** 로컬 Release 컴파일 데이터베이스의 NCE C++ 번역 단위 8개와 RO 연동 1개를 Clang 23.1.5의 `-fsyntax-only`로 검사했다. `NXN1X_DIAGNOSTICS=0` 및 `=1`에서 각각 9/9가 종료 코드 0이었다. 진단 켬 검사에는 외부 헤더의 특성 관련 경고가 있었다. 패처는 두 설정 모두 AST를 추가로 대조했다.
- **검증 한계.** 이 결과는 현재 소스의 정적 점검 기준선이다. 링크, 게임 실행, 실제 스레드 중단·재개, Linux/Android ARM64 빌드는 이번 점검에 포함하지 않았다. 구문 검사만으로 향후 삭제의 무회귀를 판정할 수 없다.
- **재확인용 스냅샷.** 위 42개 파일의 합성 SHA-256은 `0420907521297e81590d4ce6947f0a748b5dedf24c48e86ec326adb10a0d45bf`다. 저장소 루트 기준 슬래시 경로로 정렬한 뒤, 각 행을 경로·탭·원본 바이트의 소문자 SHA-256·LF 순으로 연결하여 다시 SHA-256을 계산했다.

## 5. 수정 후 통과 기준

1. 첫 변경의 diff를 위의 private 래퍼 9개로 한정한다. 같은 컴파일러와 Release 옵션으로 전후 `patcher.cpp`를 컴파일해 실행 코드와 재배치 정보를 비교하고, Windows ARM64 전체 링크와 두 진단 설정의 구문 검사를 확인한다.
2. 같은 타이틀 빌드·설정·저장 지점에서 기존 실행 파일과 변경본을 비교한다. 두 실행 모두 NCE로 진입하는지, 동일한 게임플레이 지점까지 도달하는지, 새 예외·중단·정지가 생기는지 확인한다. 로더까지 손대는 후속 변경에는 동적 NRO 로드와 분할 패치 경로를 포함한다.
3. x18 인코딩이나 스레드 중단 코드를 다룰 때는 별도 변경으로 검증한다. 전자는 생성 명령어와 복구 메타데이터의 일치, 후자는 멀티코어 중단·재개 반복과 스택 경계 복원을 통과 기준으로 삼는다. 비Windows 상수는 해당 플랫폼 빌드에서 확인한다.
