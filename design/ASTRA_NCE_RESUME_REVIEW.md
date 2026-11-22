# Windows ARM64 NCE 예외·리다이렉트·복귀 설계 검수

- 검수일: 2026-11-20
- 대상: `ASTRA_NCE_RESUME_REVIEW_REQUEST.md`
- 범위: 요청서의 상태 전이·불변식·Q1–Q6와 Microsoft/Arm 문서 대조.
- 제한: 프로젝트 소스, 원시 크래시 로그, 해당 Windows 빌드의 디스어셈블리는 첨부되지 않았다. 요청서의 코드 위치는 감사 대상이며, 실제 구현을 읽었다는 의미가 아니다.

## 1. 결론

**tebfix5는 보고된 SP 범위 불일치를 공통 처리하는 개선이지만, 전체 NCE 복귀 설계의 완성을 입증하지는 못한다.**

**브리지를 유지하고, 경계를 선택하는 정책과 최종 재개 상태 검증을 공통화해야 한다.** 실제 SP 전환 시점의 게시까지 제거하고 VEH의 SP 보정만 남기는 방식은 채택할 근거가 없다. 정상 진입, 직접 `NtContinue`, cross-thread `SetThreadContext`는 자체 VEH를 반드시 거치는 경로가 아니기 때문이다.

우선 보완할 항목은 다음과 같다.

| 우선순위 | 항목 | 현재 판정 |
|---|---|---|
| 1 | 대상 thread·TEB·GuestContext의 일치 | Q2는 미확인. 잘못된 x4는 다른 스레드의 TEB를 수정하고 현재 x18까지 잘못 설정한다. SP 보정으로 해결되지 않는다. |
| 1 | `ContextFlags`의 x18 선택 | **공식 헤더의 `CONTEXT_FULL`은 x18을 포함하지 않는다.** W5의 설명보다 실제 플래그 확인이 선행돼야 한다. |
| 1 | 중단된 PC의 전환 단계와 원래 guest 상태 | G뿐 아니라 X·E1·E2·R·VEH 안에서 중단됐을 때의 재구성 규칙이 필요하다. |
| 2 | I1의 미해결 분기 | 맞지 않는 SP를 그대로 두고 count만 증가시키는 분기가 있으므로 “모든 재개에서 수용”이라는 무조건적 불변식은 성립하지 않는다. |
| 2 | I2의 보장 범위 | 자체 fit 이전의 OS dispatch·선행 handler·wrapper prologue는 설명되지 않았다. |
| 2 | P6·중첩 예외·비복귀 종료 | 최종 CONTEXT 검증과 handler 상태 정리를 같은 계약에 포함해야 한다. |
| 2 | allocation과 실제 stack의 구분 | 넓은 allocation에 SP가 포함된다는 사실은 논리적 stack과 접근 가능한 공간의 유효성을 보장하지 않는다. |

위 표는 **문서에서 확인된 공백과 미검증 항목**이다. 모든 항목이 현재 바이너리에서 실제 버그로 발생했다고 판정한 것은 아니다.

## 2. Windows 전제 W1–W5의 정정

### 2.1 W1: 두 번째 허용 조건에도 상한이 필요하다

요청서가 인용한 역분석을 식으로 기술하려면 다음과 같이 써야 한다.

```text
StackLimit <= SP <= StackBase

또는

StackLimit != DeallocationStack
AND StackLimit - 0x1000 <= SP <= StackBase
```

요청서의 두 번째 조건에는 `SP <= StackBase`가 빠져 있다. Cygwin의 AArch64 작업자가 직접 확인한 기록에서도 두 경로 모두 StackBase를 상한으로 사용한다. 다만 그 기록은 `ExceptionRecord == NULL` 경로에 관한 관찰이다. [S1]

`ntdll+0x34720`, `+0x34a10`, `TEB+0x1478`와 정확한 검사 분기는 **대상 OS/ntdll 빌드에 종속된 관찰**로 기록해야 한다. Microsoft의 공개 `RtlRestoreContext` 문서는 이 내부 검사식을 API 계약으로 보장하지 않는다. 실제 지원 빌드의 버전·모듈 식별 정보와 디스어셈블리를 연결해야 한다. [S2]

추가 4 KiB 허용을 실제 stack 여유 공간이나 접근 권한으로 취급해서도 안 된다.

### 2.2 W2: “bounds 밖이면 모든 중첩 예외가 unhandled”는 너무 강하다

VEH는 frame 기반 handler가 아니며, debugger first-chance 이후와 stack unwind 이전에 호출된다. 따라서 잘못된 stack bounds가 frame dispatch/unwind를 방해한다는 주장과, 중첩 예외가 무조건 처리되지 않는다는 주장은 구분해야 한다. 다른 VEH가 처리하는 경로도 있다. [S3]

bounds를 맞추는 것은 다음을 만들어 주지 않는다.

- 해당 예외를 처리할 실제 handler.
- 올바른 unwind 정보와 보존된 frame.
- 추가 스택 공간, 올바른 접근 권한, guard 상태.
- 올바른 host x18 및 SP 정렬.

Windows ARM64는 SP의 16바이트 정렬을 요구하고, unwind는 prologue/epilogue 중간 상태도 설명할 수 있어야 한다. [S4][S5]

### 2.3 W3·W4: 관측 범위와 일반 보장을 분리해야 한다

- **W3:** 해당 구현에서 VEH가 guest stack에서 실행된다는 관측은 이번 검토의 전제로 사용할 수 있다. 그러나 모든 overflow·중첩·비동기 진입의 상세 stack 배치까지 이 문장으로 보장할 수는 없다.
- **W4:** 현재 `NtContinue` 경로가 W1 검사를 같은 방식으로 수행하지 않는다는 설명은 대상 빌드의 관찰로 남겨야 한다. **검사 회피 자체를 올바른 stack 상태의 근거로 삼으면 안 된다.**

### 2.4 W5: x18 미복원은 먼저 ContextFlags를 확인해야 한다

Microsoft의 `winnt.h` 원문을 확인한 결과는 다음과 같다. [S6]

| 상수 | 확인 결과 |
|---|---|
| `CONTEXT_ARM64_X18` | `0x00400010` |
| `CONTEXT_ARM64_FULL` | CONTROL·INTEGER·FLOATING_POINT 조합, 값 `0x00400007`. **x18 제외.** |
| `CONTEXT_ARM64_ALL` | x18 포함. |
| ARM64의 `CONTEXT_FULL` | `CONTEXT_ARM64_FULL`의 별칭. |

헤더는 user-mode x18이 TEB 포인터이므로 의도 없이 바꾸지 않도록 FULL에서 제외한다고 설명한다. `SetThreadContext` 역시 `ContextFlags`에 따라 설정할 부분을 선택한다고 문서화돼 있다. [S6][S7]

따라서 다음을 확인해야 한다.

1. Get/Set/Continue 각각에 전달한 실제 flags.
2. 각 단계의 x18 필드가 유효한 값인지.
3. 그 값이 물리적 host TEB인지, 논리적 guest x18인지.
4. x18 substitution window에서 캡처한 물리 상태를 어떻게 guest 상태로 해석하는지.

플래그의 포함 여부는 아키텍처 비트를 공유하므로 다음처럼 전체 마스크로 판정한다.

```cpp
const bool has_x18 =
    (ctx.ContextFlags & CONTEXT_ARM64_X18) == CONTEXT_ARM64_X18;
```

필요한 필드를 유효한 값으로 구성하고 해당 플래그를 선택해야 한다. **이 검수로 실제 현상의 원인이 플래그 누락이라고 확정한 것은 아니다.**

“kernel exception delivery 중인 대상에 SetThreadContext를 적용하면 pending 예외가 새 context로 전달된다”는 W5의 나머지 설명도 공개 API가 보장하는 세부 동작은 아니다. 원래 fault와 bridge 자체의 fault를 구분할 기록이 필요하다. [S7]

## 3. I1–I4에서 빠진 상태

### 3.1 I1: 숫자상 SP 수용과 정상 재개는 다르다

I1에는 직접적인 논리적 공백이 있다.

> SP가 기존 bounds에 맞지 않고, host/guest 분류에도 실패하면 그대로 두고 count한다.

이 분기에서 그대로 `CONTINUE_EXECUTION`하면 I1의 사후조건은 충족되지 않는다. “지금까지 0회”는 발생 이력이며 불변식의 증명이 아니다.

또한 다음 구분이 필요하다.

| 현재 판정 | 부족한 조건 |
|---|---|
| SP가 기존 bounds 안이면 유지 | bounds가 현재 stack identity와 mapping 세대에 대응하는지. 이전의 과도한 확장을 그대로 신뢰하지 않는지. |
| host reservation 안이면 저장한 host bounds 사용 | 예약 범위와 현재 접근 가능한 stack 범위의 구분. 선택한 최종 bounds가 해당 SP를 실제로 수용하는지. |
| guest 주소이면 해당 allocation 사용 | 실제 fiber stack의 하위 범위, 접근 권한, 필요한 여유 공간, mapping의 유효성. |
| 분류 실패면 유지·count | 정상 재개와 구별되는 명시적 실패 결과 및 정의된 처리 경로. |

`VirtualQuery`는 같은 상태·보호 속성의 연속 페이지 구간을 알려준다. allocation 안의 사용자 정의 fiber stack 크기를 알려주는 함수가 아니다. Windows thread stack도 예약 범위와 commit된 범위를 구분한다. [S8][S9]

**보완:** fit 결과를 성공/미지원/잘못된 상태로 구분하고, 모든 정상 재개에는 최종 사후조건을 요구해야 한다. 게스트 stack이 임의 heap allocation에서 만들어져 정확한 하위 범위를 알 수 없다면, allocation 기반 보정의 한계를 명시해야 한다.

`SP == StackBase`도 별도 경계 사례다. SP는 비어 있는 stack의 상단을 가리킬 수 있으므로, SP 자체를 메모리 주소로 조회하면 다음 allocation을 선택할 수 있다. 알려진 stack 구간을 기준으로 판정해야 한다.

### 3.2 I2: 자체 VEH에 도착하기 전의 구간이 빠져 있다

현재 설명에는 다음 순서가 포함되지 않는다.

1. guest가 다른 fiber로 SP를 변경한다.
2. TEB bounds는 이전 stack을 가리킨다.
3. 예외가 발생하고 OS dispatch가 시작된다.
4. 선행 handler 또는 자체 wrapper의 prologue가 실행된다.
5. 자체 `FitTebToSp`가 실행된다.

**5번에서의 보정이 3~4번을 소급해서 보호하지는 못한다.** 이 구간이 실제 대상 빌드에서 항상 실패한다는 뜻은 아니다. 현재 I2의 증명 범위 밖이라는 뜻이다. VEH는 예외 처리의 최초 실행 지점이 아니다. [S3]

“첫 명령부터 보장”이라고 쓰려면 C++ 첫 문장뿐 아니라 실제 entry assembly, compiler prologue, stack probe, TLS 접근과 삽입된 instrumentation을 확인해야 한다.

또한 bounds를 맞췄더라도 guest stack의 여유 공간이나 host ABI가 부적합하면 handler 자체가 안전해지지 않는다. Windows ABI에서 user-mode x18은 TEB다. guest x18을 허용하는 구간과 host 코드 진입 상태를 분리해야 한다. [S4]

### 3.3 브리지 중간 PC로 복귀하는 조건부 누락

요청서의 host bridge 순서는 `x18 복구 → host bounds 게시 → host SP 설정`이다.

다음 세 조건이 함께 허용되면 SP-only fit과 충돌한다.

- bounds 게시 후 SP 변경 전의 **중간 PC**에서 중단.
- 자체 handler가 현재 SP에 맞춰 bounds를 guest로 보정.
- 게시부터 다시 실행하지 않고 그 중간 PC로 복귀.

| 단계 | 실제 SP | TEB bounds | 다음 동작 |
|---|---|---|---|
| bridge 진입 | guest | guest | host bounds 게시 |
| 게시 직후 | guest | host | `mov sp, host_sp` 대기 |
| 중첩 handler의 fit | guest | guest | 중간 PC로 복귀 |
| SP 전환 완료 | host | guest | **불일치가 남음** |

이것은 **조건부 상태 전이 반례**이며, 현재 바이너리에서 재현한 크래시가 아니다.

- P8이 bridge **첫 명령**으로 복귀해 경계를 다시 게시하면 이 반례에 해당하지 않는다.
- 해당 중단을 항상 안전한 bridge 시작점으로 redirect하는 구현이라면 이 경로를 막을 수 있다.
- guest 진입 방향도 게시와 SP 전환 사이의 중단 정책을 확인해야 한다.

따라서 각 bridge PC 구간에 대해 “그대로 재개”, “게시부터 재실행”, “완성된 host/guest 상태로 정규화” 중 어떤 처리가 허용되는지 정의해야 한다. 재시작 시 필요한 bridge 인수도 보존돼야 한다.

bounds 두 값을 한 명령으로 저장하더라도 SP까지 동시에 바뀌지는 않는다. Special user APC는 alertable wait 없이 user code를 중단할 수 있으며 중첩도 가능하므로, 단순히 전환 구간이 짧다는 이유로 비동기 진입을 제외할 수 없다. [S10]

### 3.4 I3: 로그 제거는 호출 경로 전체에 적용돼야 한다

진단을 host stack의 `PrepareRunThread`로 옮긴 방향은 타당하다. 다만 I3은 wrapper만의 속성이 아니라 **모든 간접 호출까지 포함한 속성**이다.

Microsoft는 VEH에서 동기화 객체를 획득하거나 메모리를 할당하는 호출을 피하도록 안내한다. [S11]

검사 범위는 `FitTebToSp`, 주소/할당 조회, ring 기록, `OnWatchedWrite`, `HandleNceReadFault`, inline fault, cache-op, BRK/x18 fallback, `RedirectToHost`, `ContinueContext`다.

각 경로에서 할당·락·첫 호출 초기화·assert/CRT·출력뿐 아니라 보호된 guest 데이터와 잘못된 metadata 접근도 확인해야 한다. counter와 ring도 유효한 host 저장소와 검증된 인덱스를 사용해야 한다.

### 3.5 I4: 전달받은 x4가 정본 TEB라는 독립적 근거가 필요하다

다른 스레드의 TEB가 x4로 전달되면 bridge는 다음 두 동작을 수행한다.

1. 다른 스레드의 TEB bounds를 수정한다.
2. 현재 스레드의 물리 x18을 그 잘못된 TEB로 설정한다.

이후 SP가 숫자상 어떤 bounds에 들어간다는 사실은 이 문제를 복구하지 못한다.

따라서 동일한 native thread 수명에 다음을 연결해야 한다.

- 보유 중인 thread handle과 thread identity.
- host ABI가 성립하는 초기화 구간에서 확보한 정본 TEB.
- 해당 host stack 정보.
- 현재 GuestContext와 guest 실행 epoch.
- redirect 세대와 변환 전 guest 상태.

`m_windows_break->Teb()`는 이 등록 정보와 일치할 때만 사용할 수 있다. 잘못된 물리 x18을 통해 읽은 TLS로 그 x18의 소속을 다시 검증하는 방식은 독립적인 확인이 아니다.

## 4. Q1–Q6 답변

### Q1. P6의 직접 NtContinue는 stale bounds여도 안전한가?

**그렇게 승인할 수 없다. 다만 P6가 항상 잘못됐다고 단정할 수도 없다.**

현재 handler SP와 최종 `Context.Sp`가 동일한 정확한 stack 범위에 있고, 이후 bounds가 바뀌지 않았다면 entry fit이 둘을 모두 수용할 수 있다.

필요한 계약은 다음과 같다.

- P6도 최종 `Context.Sp`, PC, target identity, 전환 단계를 검사한다.
- handler의 현재 SP와 재개 대상 SP를 구분한다.
- 둘이 다른 stack이면, 현재 C++/OS 복귀 코드가 아직 이전 SP를 사용하는 동안 미래 bounds를 무작정 게시하지 않는다.
- `ContinueContext`가 실패해서 반환하는 경로까지 정리한다.

cache-op 처리 후 PC+4로 가는 것은 해당 명령을 완료한 경로의 계약이다. **원래 fault를 아직 처리하지 않은 일반 메모리 AV에 같은 PC 증가 규칙을 적용해서는 안 된다.**

### Q2. m_windows_break->Teb()는 항상 중단 대상 스레드인가?

**현재 자료로 확인할 수 없다. 요청서에는 잘못된 binding의 기존 사례가 있으므로 우선 감사 대상이다.**

`arm_nce_windows.cpp`의 두 cross-thread transform과 break 객체 생성·재사용·해제 수명을 함께 확인해야 한다. Suspend/Get/Set/Resume 대상이 동일한 보유 handle인지, 그 대상의 TEB·GuestContext·epoch를 사용하는지 확인한다.

동일 TID 숫자나 호출자의 TLS만으로 충분하다고 보지 않는다. 스레드 종료·재생성 및 guest 실행 epoch 변경도 구별해야 한다.

### Q3. P8에서 원래 fault PC로 재진입하면 guest 상태가 유실되는가?

**원래 PC로 돌아간다는 사실만으로 상태 보존이 입증되지 않는다.**

#### 원래 guest 상태와 bridge 실행 상태를 분리해야 한다

`RedirectToHost`는 PC와 x1–x4, x16을 bridge 용도로 바꾼다. 변환 전에 원래 guest 상태를 별도로 확보해야 한다. 이미 redirect 중인 상태를 다시 캡처했을 때 최초 guest 상태를 덮어쓰면 안 된다.

보존 대상에는 SP·PC·일반 레지스터·조건 플래그·guest x18·SIMD/FP 상태가 포함된다. 공개 `ARM64_NT_CONTEXT`에는 관련 필드가 있지만, 구조체가 존재한다는 사실과 실제 캡처·복원에 필요한 flags/값이 채워졌다는 사실은 다르다. [S12]

x18 substitution window나 entry/bridge 중간이라면 물리 CONTEXT를 논리 guest 상태로 해석할 단계별 규칙도 필요하다.

#### 메모리 명령의 미완료와 부작용 부재를 동일시하면 안 된다

faulting PC가 남아 있다는 사실만으로 명령의 메모리 효과가 전혀 없었다고 판단해서는 안 된다. Arm은 여러 메모리 접근을 수행하는 명령의 precise exception에서도 조건에 따라 영향을 받은 상태가 달라질 수 있음을 규정한다. [S13]

**이 요청서만으로 특정 STP의 일부 저장이 실제 발생했는지는 확인되지 않는다.** 원래 명령의 재시도 상태와 아키텍처의 재실행 의미를 보존해야 한다. page-watch나 snapshot 처리도 “fault 발생 = 명령 전체가 아직 아무것도 쓰지 않음”을 전제로 삼아서는 안 된다.

일반 RAM의 재실행과 Device/MMIO 또는 host가 대신 수행하는 부작용 있는 접근은 구분해야 한다.

#### Exclusive monitor 소실은 허용된 실패 경로로 처리해야 한다

Arm은 context switch 때 local monitor를 해제해야 하며 ERET에서도 해제된다고 설명한다. 중간에 중단된 Load-Exclusive/Store-Exclusive 흐름에서는 Store-Exclusive가 실패할 수 있다. [S14]

공개 CONTEXT에는 exclusive reservation을 저장하는 필드가 없다. 필요한 것은 reservation 보존 주장보다 **STXR/STLXR 실패 결과와 guest 재시도 흐름의 보존**이다. 실패를 성공으로 바꾸거나 명령을 생략해서는 안 된다. [S12][S14]

#### P8의 원인 판정에 필요한 연결 기록

같은 redirect 세대에 대해 다음을 연결한다.

```text
변환 전 raw CONTEXT 및 해석된 guest 상태
변환 후 bridge CONTEXT와 ContextFlags
ExceptionRecord.ExceptionAddress / exception code / fault VA
handler 진입 당시 PC / SP / x18 / TEB bounds / 전환 단계
```

bridge PC와 guest fault VA가 함께 기록됐다는 사실만으로, pending guest fault인지 실제 bridge fault인지 확정하지 않는다.

### Q4. heap allocation이 fiber stack보다 큰 경우 I2 fit은 올바른가?

**W1의 숫자 검사를 통과시키는 보정과 실제 stack 검증을 구분해야 한다.**

큰 allocation은 논리적 fiber stack과 같지 않다. allocation 전체를 bounds로 게시하면 W1을 통과하더라도 실제 stack 하한 침범이나 다른 데이터 영역 침범이 가려질 수 있다.

알려진 stack은 논리적 범위·mapping 세대·접근 가능한 영역을 기준으로 판정한다. guest fiber가 native thread 사이에서 이동한다면 stack을 영구적으로 한 host thread에 고정할 필요는 없지만, **현재 guest 실행 문맥과의 연결**은 일치해야 한다.

하위 stack 범위를 알 수 없는 경우에는 allocation 기반 복구의 제한을 명시한다. bounds 수정만으로 commit·guard·unwind 정보가 생기지는 않는다. [S8][S9]

### Q5. P1–P9에 빠진 경로가 있는가?

**있다. §5의 추가 상태표가 필요하다.** 특히 cross-thread break가 G에서만 발생한다고 가정하면 안 된다. 정상 entry/exit와 X·R·VEH 안에서의 중단을 구별해야 한다.

APC·debugger·다른 handler는 실제 사용/지원 여부를 확인해 범위에 포함하거나, 제외 조건을 명시한다.

### Q6. t_crash_depth가 1400–1500이어도 무해한가?

**무해성은 확인되지 않았다.**

요청서 설명대로 성공한 `NtContinue`가 wrapper로 돌아오지 않으면, 그 값은 실제 nesting depth를 나타내지 않는다. 종료 정리를 건너뛴 진입 횟수가 누적될 수 있다.

필수 감사 대상은 다음과 같다.

- 모든 read/write: 재진입 제한, fatal/assert, 로그 억제, 배열·ring 인덱스 등에 쓰이는지.
- 일반 return, `CONTINUE_SEARCH`, 비복귀 transfer의 accounting.
- counter 외에 lock guard·scope guard·임시 상태의 정리가 생략되는지.
- 실패 반환 때 복구·감소가 중복되거나 누락되는지.
- nested handler가 outer handler로 돌아가는 경우와 outer frame까지 버리는 경우.
- 타입·overflow·새 실행 진입 시 초기화.

총 진입 횟수와 활성 깊이를 분리하고, **비복귀 전이에서 필요한 정리는 transfer 전에 명시적으로 수행해야 한다.** RAII 소멸자에 decrement를 넣는 것만으로는 정상 return이나 C++ unwind를 거치지 않는 전이를 해결하지 못한다.

중첩 프레임을 모두 버리는 전이를 무조건 `--depth` 하나로 처리하거나, outer handler가 남아 있는데 0으로 초기화하는 방식도 맞지 않는다. 복귀 대상에 실제 남는 handler 범위를 기준으로 정리해야 한다.

## 5. 추가해야 할 resume/dispatch 상태표

| 추가 상태 | 도달 순서 | 요구 사항 |
|---|---|---|
| 자체 VEH 이전 | guest fiber SP 변경 → 예외 → OS/선행 handler → wrapper | fit 이전의 ABI·stack 사용 조건과 실패 범위 명시 |
| wrapper prologue/epilogue | 진입/종료 기계어 실행 → 중간 fault | 실제 기계어와 필요한 unwind 정보 확인 |
| bridge 중간 복귀 | 미래 bounds 게시 → 중단 → 현재 SP fit → 중간 PC 재개 | §3.3의 단계별 재개 규칙 |
| X·E1·E2·R 중단 | cross-thread suspend가 전환 중간을 캡처 | 물리 CONTEXT를 guest 상태로 재구성하거나 정의된 안전 경계로 처리 |
| H 또는 VEH 내부 중단 | host 작업/handler 실행 중 P7 발생 | 일반 guest snapshot으로 오인하지 않음. 버리는 host frame의 정리와 소유 상태를 검증 |
| `CONTINUE_SEARCH` | 자체 handler 미처리 → 다음 VEH/SEH | context·bounds·handler accounting의 인계 규칙 |
| debugger 직접 재개 | first-chance → `DBG_CONTINUE` 또는 외부 context 변경 | 자체 VEH fit을 거치지 않는 경로의 지원 계약 |
| 일반/special APC | alertable dispatch 또는 비동기 user-code 중단 | callback 진입·복귀의 ABI와 전환 단계, 중첩 처리 |
| 다른 VEH/continue handler | 자체 처리 전후에 다른 callback 실행 | CONTEXT/bounds 변경과 순서 확인 |
| guard/stack overflow | stack 공간/보호 상태 때문에 예외 발생 | bounds 확대와 실제 공간·guard 복구를 구분 |
| NtContinue 실패 반환 | 준비한 비복귀 전이가 실패하여 return | 현재 실행 stack 상태와 accounting의 실패 처리 |
| 나머지 patcher fallback·정상 종료 | trap 외 경로 또는 직접 stack 전환 | 전체 전이 목록과 공통 검증 연결 |

Debugger는 VEH보다 먼저 예외를 처리할 수 있고, `DBG_CONTINUE`는 예외 처리를 끝내고 thread를 재개한다. 따라서 “모든 예외 재개가 자체 VEH를 통과한다”는 전제는 사용할 수 없다. [S3][S15]

일반 APC는 alertable 상태에서 실행되지만 special user APC는 그렇지 않은 user code도 중단할 수 있고 중첩될 수 있다. 이 차이를 상태표에 반영해야 한다. [S10]

## 6. 권장 구조: 공통 선택 정책 + 단계별 게시

공통화할 대상은 다음 세 질문이다.

1. **어느 native thread와 guest 실행 문맥인가?**
2. **현재 실행 stack과 최종 재개 stack은 각각 무엇인가?**
3. **현재 PC 단계에서 어느 상태로 재개하는 것이 유효한가?**

이를 판단하는 입력에는 target thread 수명, 정본 TEB, GuestContext/epoch, 원래 guest snapshot, PC 단계, 현재 SP, 최종 CONTEXT, stack 범위·mapping 세대가 필요하다.

| 지점 | 책임 |
|---|---|
| thread/guest 실행 등록 | target identity·TEB·host stack·GuestContext/epoch 연결 |
| VEH 진입 | 실제 실행 상태 확인과 최소 보정. 원래 fault 정보 보존 |
| 재개 준비 공통부 | 최종 PC/SP/레지스터·stack·TEB·전환 단계·ContextFlags 검증 |
| E1/E2/X/host bridge | 실제 전환 위치에서 필요한 x18·bounds·SP 게시/복원 |
| 중간 PC 복구 | 그대로 재개 가능한지 또는 어느 전환 구간을 재실행해야 하는지 결정 |
| host 복귀 이후 | 복잡한 처리와 로그, 누적 진단 출력 |

**“정책이 한곳에 있다”와 “모든 경로의 실제 store가 한곳에 있다”를 동일시할 필요는 없다.** 같은 판단 결과를 entry·bridge·handler가 각자의 유효한 시점에 게시할 수 있다.

현재 bridge를 그대로 유지하면 자동으로 완성된다는 결론도 아니다. target TEB와 인수의 수명, 중간 PC 정책, 게스트 원본 상태 보존을 함께 보완해야 한다.

## 7. 최소 코드 감사 및 검증 기준

아래는 이번에 실행한 테스트가 아니라, 해당 구현을 완료 판정하기 위한 범위다.

### 7.1 코드 감사

| 위치 | 확인 사항 |
|---|---|
| `arm_nce_windows.cpp`의 `FitTebToSp`, wrapper, Impl | I1 실패 분기, 현재/재개 SP 구분, 최초 기계어, 호출 경로, 모든 종료 |
| 같은 파일의 cross-thread transform 두 곳 | 동일 대상 handle/TEB/GuestContext/epoch, G 외 상태 분류 |
| `windows_cross_thread_break.h` | Suspend/Get/Set/Resume 결과 처리, 대상 수명과 중복 redirect |
| `windows_nce_transition.cpp` | 변환 전 원본 보존, 최종 CONTEXT·flags, 비복귀 정리와 실패 반환 |
| `windows_nce_entry.asm` | 게시와 SP 변경 사이의 모든 PC, x18 복구 시점, 인수 보존 |
| `windows_x18_fallback_trap.cpp`, `patcher.cpp` | 모든 x18 대체 구간과 trap 외 fallback, 중단 상태 재구성 |
| `t_crash_depth` 전체 참조 | 활성 깊이의 실제 사용과 종료 accounting |

### 7.2 정보량이 높은 검증 묶음

| 검증 | 통과 기준 |
|---|---|
| P1–P6를 host/기존 guest/다른 fiber stack에서 실행 | 최종 재개 SP와 선택 stack 일치. P6도 같은 검증에 포함 |
| P7을 G·X·E1·E2·R·H·VEH에서 중단 | 각 단계의 정의된 결과. bridge CONTEXT를 원래 guest snapshot으로 덮어쓰지 않음 |
| 잘못된 TEB/오래된 epoch를 전달 | 다른 thread TEB를 수정하기 전에 탐지 |
| bounds 게시와 SP 변경 사이에 중단 주입 | 중간 PC 재개와 bridge 시작점 재개 모두 완료 상태의 SP/TEB/단계가 일치 |
| ContextFlags별 캡처·복원 | x18 선택 여부와 유효값이 일치. 필요한 일반·SIMD/FP·조건 플래그 보존 |
| SP 상단 경계·미확인 stack·작은 여유 공간·guard | 다른 allocation 오인이나 미지원 상태의 정상 재개 처리 없음 |
| STP 등 다중 접근 fault와 exclusive 중단 | 대상 명령/메모리 속성에 맞는 재실행. STXR 실패 결과와 guest retry 보존 |
| P4–P6 반복·중첩 예외·ContinueSearch·실패 반환 | 활성 handler 수와 accounting 일치. 정리 누락이나 중복 없음 |
| 실제 지원하는 APC/debugger/다른 handler 경로 | 자체 VEH를 우회해도 정의된 상태로 복귀 |

Windows 버전·ntdll 식별 정보·빌드 옵션과 진단 기록을 같은 실행에 연결한다. 단순 장시간 실행보다 **어느 상태 전이가 검증됐는지**를 남기는 것이 완료 판정에 직접적이다.

## 8. 현재 크래시 표의 판정 수정

- **17:18:** 요청서의 `FATAL NVN command memory`는 직접적인 종료 지점을 뒷받침한다. 문구는 **“직접 종료 지점은 NVN command-memory fatal; NCE와의 인과관계는 미확인”**으로 수정하는 것이 정확하다. 이것만으로 NCE 상태 손상이 상류 원인이 아니었다고 입증되지는 않는다.
- **16:21·04:15·09:24:** 로그가 없고 같은 ucrt abort 위치라는 이유로 동일한 `die()`라고 분류할 수 없다. **원인 미확인**으로 남긴다.
- **17:06:** 로그가 fit보다 먼저 실행됐다는 설명은 해당 빌드의 호출 순서 확인을 근거로 명시한다. 로그 줄이 0개였다는 사실만으로 순서를 증명하지는 못한다.
- **16384 entry fits, 0 left:** 관측된 entry fit 결과다. pre-wrapper 경로, debugger 우회, 모든 직접 NtContinue와 중간 PC 복귀의 완전성을 의미하지 않는다.

## 9. 근거

공식 API/ABI, 공개 공식 헤더, Arm 아키텍처 자료와 구현 역분석 기록을 구분했다. 요청서의 crash 수치·코드 위치·tebfix 이력은 사용자 제공 자료이며 원시 증거를 독립 재현하지 않았다.

- **[S1] 구현 역분석 기록:** [Cygwin AArch64 signal handling 개발 메일](https://cygwin.com/pipermail/cygwin-patches/2026q3/015251.html). `ExceptionRecord == NULL`의 RtlRestoreContext SP 검사 관찰. Microsoft API 보장은 아님.
- **[S2] 공식 API:** [RtlRestoreContext](https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-rtlrestorecontext).
- **[S3] 공식 설명:** [Vectored Exception Handling](https://learn.microsoft.com/en-us/windows/win32/debug/vectored-exception-handling).
- **[S4] 공식 ABI:** [ARM64 Windows ABI conventions](https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170).
- **[S5] 공식 unwind 규약:** [ARM64 exception handling](https://learn.microsoft.com/en-us/cpp/build/arm64-exception-handling?view=msvc-170).
- **[S6] Microsoft 공식 헤더:** [win32metadata / winnt.h](https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/um/winnt.h). 2026-11-20 확인한 raw 원문의 ARM64 ContextFlags 정의와 주석.
- **[S7] 공식 API:** [SetThreadContext](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadcontext).
- **[S8] 공식 API:** [VirtualQuery](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualquery).
- **[S9] 공식 설명:** [Thread Stack Size](https://learn.microsoft.com/en-us/windows/win32/procthread/thread-stack-size).
- **[S10] 공식 API:** [QueueUserAPC2](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-queueuserapc2).
- **[S11] 공식 callback 계약:** [PVECTORED_EXCEPTION_HANDLER](https://learn.microsoft.com/en-us/windows/win32/api/winnt/nc-winnt-pvectored_exception_handler).
- **[S12] 공식 CONTEXT 정의:** [ARM64_NT_CONTEXT](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-arm64_nt_context).
- **[S13] Arm 공식 정오표:** [DDI0487 I.a Known Issues, §2.111 D20303, 83쪽](https://documentation-service.arm.com/static/644c18cbaa78c007af74fa2c). precise exception과 다중 메모리 접근에 대한 조건부 규정. 이 정오표만으로 특정 STP의 관측 결과를 확정하지 않음.
- **[S14] Arm 공식 설명:** [Software Synchronization Primitives in A64, §3.2.4, 21쪽](https://documentation-service.arm.com/static/68c223238a337a2bc6645c0a). context switch·ERET와 local monitor 해제.
- **[S15] 공식 API:** [ContinueDebugEvent](https://learn.microsoft.com/en-us/windows/win32/api/debugapi/nf-debugapi-continuedebugevent).

[S1]: https://cygwin.com/pipermail/cygwin-patches/2026q3/015251.html
[S2]: https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-rtlrestorecontext
[S3]: https://learn.microsoft.com/en-us/windows/win32/debug/vectored-exception-handling
[S4]: https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170
[S5]: https://learn.microsoft.com/en-us/cpp/build/arm64-exception-handling?view=msvc-170
[S6]: https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/um/winnt.h
[S7]: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadcontext
[S8]: https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualquery
[S9]: https://learn.microsoft.com/en-us/windows/win32/procthread/thread-stack-size
[S10]: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-queueuserapc2
[S11]: https://learn.microsoft.com/en-us/windows/win32/api/winnt/nc-winnt-pvectored_exception_handler
[S12]: https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-arm64_nt_context
[S13]: https://documentation-service.arm.com/static/644c18cbaa78c007af74fa2c
[S14]: https://documentation-service.arm.com/static/68c223238a337a2bc6645c0a
[S15]: https://learn.microsoft.com/en-us/windows/win32/api/debugapi/nf-debugapi-continuedebugevent

