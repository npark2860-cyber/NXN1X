# OoT full barrier 조사와 cbar1 (dispatch 사이 guest barrier)

2026-11-21 07:00 KST. NPARK 06:54 승인("barrier 먼저", "ㄱㄱ").

## 1. 측정 (ab_import_002257, import+ootfadd, 660초)

- r107: full barrier **13,265/s**입니다(약 30 fps 기준 프레임당 약 440개, ootgap 창에서는 gap 안에만 262개).
- ootgap 줄:
  - span 14.56 ms = timed pass 7.50 + part 경계 0.14 + gap 6.93
  - gap 중 "full barrier와 그보다 가벼운 것만 있는 gap"이 5.59 ms(139.2개)
  - full barrier가 2개인 gap이 **78.8개, 5.36 ms**로 가장 큽니다.
  - 비용이 큰 gap 위치는 모두 dispatch 앞이고, 기록 순서는 `… DISPATCH BARRIER (binds) DISPATCH`입니다. GPU 명령은 "pipeline barrier 1, full barrier 2"입니다.
    - pass #14 앞 2.5 ms, #5 앞 0.7 ms, #8 앞 0.5 ms 등
- s64a: 제출당 OP_BARRIER 25.2개입니다. 그중 pass가 열린 상태가 12.9개입니다.
  - flag 분포: 0x1 ORDER_PRIMITIVES 9.2, 0x2 ORDER_FRAGMENTS 13.2, 0x8 ORDER_INDIRECT_DATA 3.0, 0x10 INVALIDATE_TEXTURE 20.7, 0x20 INVALIDATE_SHADER 3.0, 0x80 0.7

## 2. full barrier를 넣는 곳 (gfx_vk_exec.c, gfx_vk_core.c)

| 곳 | 목적 | stage/access |
|---|---|---|
| OP_BARRIER(exec.c, guest nvnCommandBufferBarrier) | guest가 요구한 순서. flag는 보지 않음(s64가 pass 안의 INVALIDATE_TEXTURE만 미룸) | ALL→ALL, 장치 access 전부(r89 bit 1), s63 중복 제거 |
| dispatch 앞(exec.c r89) | 앞 작업 → 이 dispatch | 직전이 dispatch이고 그 뒤 full이 없으면 COMPUTE\|TRANSFER→COMPUTE\|TRANSFER(+DRAW_INDIRECT). 아니면 full |
| dispatch 뒤(r89) | 이 dispatch의 쓰기 → 이후 모든 것 | COMPUTE/SHADER_WRITE → ALL_COMMANDS/장치 access 전부 |
| pass 시작(exec.c pass begin) | 앞 작업 → attachment와 셰이더 읽기 | full |
| pass 끝(core.c 3193 등) | attachment 쓰기 → 다음 | full |
| copy/fill 앞뒤(core.c mtl_sub_copy/fill) | transfer 순서 | full ×2 |
| indirect dispatch c0 복사(r75) | GPU가 쓴 group count → driver 세그먼트 | full ×2 |

**OoT에서 full이 남는 이유**:
- r89(DM_VK_BAR bit 2, 기본 켜짐)는 이미 있습니다. 연속 dispatch 사이는 compute 전용 barrier로 바꿉니다.
- 그런데 OoT는 dispatch마다 guest barrier를 기록합니다.
- OP_BARRIER가 full을 기록하면서 g_bar_seq를 올리므로 r89의 조건(`g_disp_seq == g_bar_seq`)이 깨집니다. 그래서 다음 dispatch 앞에도 full이 하나 더 들어갑니다.
- 결과는 gap 하나에 full 2개입니다.

## 3. Eden 원본 비교 (vk_rasterizer.cpp)

- DispatchCompute: dispatch 앞에 barrier 1개를 둡니다. `GRAPHICS|COMPUTE|TRANSFER, MEMORY_WRITE → COMPUTE_SHADER, MEMORY_READ`이고, 뒤에는 없습니다.
- guest WFI(WaitForIdle): `SetEvent + WaitEvents` 또는 b46 지연입니다. stage는 draw indirect부터 compute, transfer까지입니다.
- 즉 Eden도 dispatch 사이에 dst가 COMPUTE인 barrier 하나만 둡니다.
- cbar1 뒤 NXN1X도 dispatch 사이에 같은 구성이 됩니다. 앞 dispatch의 r89 뒤 barrier와 다음 dispatch의 compute 전용 앞 barrier입니다.

## 4. 수정 cbar1 (기본 켜짐, `DM_VK_CBAR=0`이면 예전처럼)

OP_BARRIER를 생략하는 조건은 둘이며 모두 만족해야 합니다.
1. 이 명령 버퍼의 마지막 GPU 작업이 r89 뒤 barrier까지 마친 dispatch이고, 그 뒤 full barrier가 없습니다(`g_disp_seq == g_bar_seq`, 같은 cb, pass 닫힘).
2. 다음 GPU 작업 레코드가 dispatch입니다.
   - lookahead(`vk_cbar_next_dispatch`)는 상태만 바꾸는 레코드를 건너뜁니다. 대상은 BIND_*, SET_*_POOL, PUSH_CONST이고, SIGNAL_EVENT도 건너뜁니다(완료 스레드에서 처리되므로 GPU 순서와 무관).
   - JUMP는 따라갑니다.
   - 그 밖의 레코드(draw, pass, copy, fill, clear, counter report, UPDATE_UB, CALL, END, 알 수 없는 것)가 나오면 full을 유지합니다.

정확성 근거:
- 앞 dispatch의 쓰기: r89 뒤 barrier(SHADER_WRITE → ALL_COMMANDS, 장치 access 전부)가 이후 모든 읽기에 보이게 합니다. 여기에는 indirect, vertex/index, uniform, transfer가 포함됩니다.
- WAR(다음 dispatch가 앞 dispatch가 읽은 것을 씀): 위 barrier의 실행 의존(COMPUTE → ALL_COMMANDS)이 지켜 줍니다.
- 사이에 기록되는 것(디스크립터 준비, upload): 다음 dispatch 앞 barrier(COMPUTE|TRANSFER → COMPUTE|TRANSFER, indirect이면 DRAW_INDIRECT와 INDIRECT_COMMAND_READ 추가)가 지켜 줍니다. ubd land 같은 full이 끼면 조건 1이 깨져 앞 barrier가 full이 됩니다.
- texture 캐시 무효화(0x10): Vulkan에서는 위 메모리 barrier의 SHADER_READ dst가 그 역할을 합니다.
- ORDER_INDIRECT_DATA(0x8): indirect dispatch 앞 barrier의 DRAW_INDIRECT가 지켜 줍니다.

남는 위험:
- dispatch 사이의 다른 경로에서 g_bar_seq를 올리지 않는 GPU 명령이 있으면 그 명령은 r89 앞 barrier(COMPUTE|TRANSFER)로만 순서가 정해집니다. r107 기준으로 copy와 fill은 모두 full로 감싸여 있습니다.
- 확인 방법: DM_VK_CBAR=0과 A/B 비교, 화면 비교.

예상 절감: full 2개가 있는 gap 78.8개/프레임(5.36 ms)이 compute 전용 barrier 1개로 바뀝니다. **프레임당 2-4 ms**로 추정합니다(미측정). 남는 것은 dispatch 사이의 실제 의존 대기입니다.

## 5. 테스트

셀프테스트 170-177(lookahead):
- 생략: binds 뒤 dispatch, JUMP를 따라간 dispatch
- 유지: draw, copy, counter, UPDATE_UB, list 끝, CALL

## 6. 로그

- `cbar1: guest barrier … left to their compute barriers (N so far, M kept full)`(2의 거듭제곱 번째마다)
- r107의 full barrier/s가 13k에서 줄어야 합니다.

## 7. cbar1 실행 결과와 측정 착시 점검 (07:10 KST, 실행 중 로그 기준)

- **cbar1 동작**:
  - guest barrier 26만 개를 생략했고, 92만 개는 full로 유지했습니다.
  - r107 full barrier는 13.3k/s에서 10.1-10.5k/s가 됐습니다.
  - 프레임 GPU(longframe)는 34-36 ms로 그대로입니다.
  - 큰 dispatch 앞 gap은 이제 "pipeline barrier 2"(full 없음)인데도 0.6-1.3 ms가 남았습니다.
- **측정 착시 가능성(높음)**:
  - dispatch 시간은 TOP_OF_PIPE(시작) → BOTTOM_OF_PIPE(끝) 타임스탬프입니다(core.c vk_pass_ts).
  - OoT 122 ms 사례에서 BOTTOM이 실제 완료보다 먼저 찍혀, 앞 dispatch(1bba09ec의 float atomic)의 작업이 "다음 dispatch(d987a9de) 앞 gap"으로 보였습니다. FADD 수정으로 앞 dispatch가 빨라지자 gap이 0.4-1.4 ms로 줄었습니다.
  - 지금 gap도 full barrier를 compute 전용으로 바꿨는데 줄지 않았습니다. barrier 대기라면 줄었어야 합니다.
  - 그래서 이 gap들은 GPU 유휴가 아니라 **앞 dispatch의 꼬리(실제 작업)**일 가능성이 높습니다. 그렇다면 gap 15.7 ms의 상당 부분은 dispatch 작업 자체이고, barrier를 더 줄여도 이득은 작습니다.
- **(가) bb9076e7 앞 6.57 ms**:
  - 기록 순서: `… END SIGNAL_EVENT BIND_PROGRAM BIND_SSBO DISPATCH BARRIER BIND_TEX1 BIND_UBO_SMALL`. 앞의 timed pass(#13)가 dispatch이고 그 뒤에 guest barrier가 있습니다.
  - full 2개가 남은 이유는 cbar1 조건 1(같은 cb, 그 사이 full 없음)이 깨졌기 때문으로 봅니다. 리스트 경계 직후의 part cut 등이 원인 후보입니다. 확정은 아닙니다.
  - 착시라면 이 6.57 ms는 #13 dispatch의 꼬리입니다.
  - pass 목록 줄은 상위 8개만 남겨서 #13이 어떤 셰이더인지 이 로그로는 알 수 없습니다.
  - Eden은 리스트 경계에 별도 barrier를 두지 않고, 다음 dispatch 앞 READ_BARRIER 하나만 둡니다.
- **(나) 범위를 아는 바인딩 비율**(hybrid2 로그 ab_hybrid_004017, 120 제출당):
  - compute 쓰기 바인딩 2,300개와 atomic 1,080개 중 reach가 바인딩보다 작아 잘린 것이 1,860개입니다. 그러니 쓰기 범위를 아는 것은 **최소 55%**입니다(reach가 바인딩 이상이면 이 카운터에 안 잡힘).
  - 읽기 범위는 대부분 크기를 모르는 바인딩이라 청크 끝까지입니다. 그래서 겹침 판정이 거의 "겹침"으로 나와 생략 효과는 작을 것으로 봅니다.
- **확인 실행 제안(한 줄)**: dispatch의 끝 타임스탬프 직전에 COMPUTE→COMPUTE 실행 barrier를 하나 넣는 진단 스위치를 두고 OoT를 한 번 돌립니다. 끝 타임스탬프가 실제 완료 뒤에 찍히게 해서 gap이 진짜 유휴인지, 앞 dispatch 작업인지 가립니다.

## 8. 진단 빌드 tsdiag (NPARK 07:10 승인, 진단만, 기본 꺼짐)

- 무거운 dispatch에 대해 기존 로그에서 알 수 있는 것("NVN2 SSBO" 줄, 바인딩마다 처음 한 번만 찍힘):
  - e855420a39e60874: slot 18, 19가 written인데 주소 0xc / 0x54(c1+0, c1+0x8)라 "outside every pool: zero stand-in"입니다. slot 16도 0x100000000으로 풀 밖입니다.
  - cee08c634ef1f343: slot 13, 14가 같은 패턴입니다.
  - 이 shader들의 쓰기 일부는 zero buffer로 갑니다. c1 오프셋을 포인터로 읽은 것으로 보이며, 원인은 별도로 확인해야 합니다.
  - f4f0bc23, bb9076e7, d987a9de: 바인딩 줄이 없습니다. 모든 바인딩이 풀 안이라 진단 줄이 찍히지 않았습니다.
- 넣은 것:
  1. `DM_VK_TS_FIX=1`: dispatch 끝 타임스탬프 직전에 실행 barrier(COMPUTE_SHADER → ALL_COMMANDS, 메모리 barrier 없음)를 넣습니다. dispatch 시간이 실제 완료까지 포함하게 되고, gap은 진짜 GPU 유휴만 남습니다. 측정용이며 barrier 하나가 더 들어가므로 fps는 약간 낮아질 수 있습니다.
  2. `DM_VK_CSDIAG=1`: compute shader별로 SSBO 바인딩을 종류(읽기 전용, written, atomic)와 메모리(import 청크, 사본, 풀 밖)로 나눠 바이트를 셉니다. 약 600 present마다 import 쓰기+atomic이 많은 상위 16개를 `csdiag:` 줄로 남깁니다.
- 실행: `RUN_AB_IMPORT_TSDIAG.cmd` 한 번(OoT, import, 같은 장소)
- 판정:
  - ootgap `span = timed passes + … + gaps`에서 TS_FIX 뒤 gap이 크게 줄고(15.7 ms에서 몇 ms로) timed pass가 그만큼 늘면 착시가 확정됩니다. 줄일 대상은 dispatch 작업 자체입니다.
  - gap이 그대로면 진짜 GPU 유휴입니다. 그때는 barrier와 제출 경계를 계속 봅니다.
  - csdiag 상위 shader가 "passes by GPU time"의 무거운 dispatch와 겹치고 import 쓰기/atomic MB가 크면 import 쓰기 비용 후보입니다(atomic 스테이징 hybrid3 근거).

## 9. tsdiag 결과 (ab_tsdiag_073921, 07:39 KST, OoT import, TS_FIX + CSDIAG)

- **판정 1: 착시가 아니라 진짜 GPU 유휴입니다.**
  - TS_FIX(끝 타임스탬프를 실제 완료 뒤로) 뒤에도 ootgap은 span 38.03 ms = timed pass 17.24 + 경계 0.28 + **gap 20.51**입니다.
  - gap 비율은 54%로, cbar1 실행(15.68/29.84 = 53%)과 같습니다.
  - gap 내역:
    - full 없음 11.61 ms(122개)
    - full 1개 0.56 ms(71.6개)
    - full 2개 8.24 ms(89.8개)
  - 프레임당 직렬화 지점 약 300개마다 평균 약 70 µs가 유휴입니다.
  - 그래서 barrier "종류"(full → compute 전용)를 바꿔도 이득이 없었습니다. 비용은 직렬화 지점의 **수**입니다.
- **판정 2: import atomic 상위 셰이더(present당 바인딩 MB)**
  - 39e60874(atomic 19.85), a0a84773(19.85), 4ef1f343(18.33), 60dd0e6e(12.66), 1b4e4e55(7.0), f700213f(4.8), b1b379b4(3.2)
  - 무거운 pass와 겹치는 것: 4ef1f343(1.51 ms), 39e60874, a0a84773, b1b379b4, f700213f. 이들은 앞서 큰 gap 위치와 겹치는 셰이더들입니다.
  - 가장 무거운 dispatch인 53d38847(1.94+1.13 ms)과 bdf51d78(1.92+1.10 ms)는 csdiag 상위에 없습니다. import 쓰기나 atomic이 거의 없다는 뜻입니다(이미지 쓰기로 추정).
  - dispatch 합계는 11.53 ms/프레임(145개)입니다.
- **오류**: RED 1줄(nvnCommandBufferSetComputeWorkloadPriority 미구현, 전부터 있던 것). FATAL과 device lost는 없습니다.
- **다음 후보**:
  - (a) 직렬화 지점 수 줄이기.
    - cbar1이 생략한 것은 6.5만, 유지한 full guest barrier는 18만입니다.
    - 유지된 barrier가 어떤 전환(draw→dispatch, dispatch→draw, pass 끝과 연속)에 있는지 나눠서, 연달아 붙은 drain(pass 끝 full + guest full + dispatch 앞 full)을 하나로 합칩니다.
    - "full 2개" gap 89.8개(8.24 ms)가 대상이고, 개당 약 70 µs를 줄이면 **약 3-6 ms/프레임**입니다(추정).
  - (b) atomic 스테이징(hybrid3), 위 4-5개 셰이더 대상. 이 dispatch들의 시간은 합해서 약 3-4 ms입니다. device-local 이득에서 스테이징 복사 비용(atomic 바인딩 약 110 MB/present → GPU 트래픽 약 1-3 ms)을 빼면 **0-2 ms**로 불확실합니다.
  - r75(indirect dispatch의 c0 복사, full 2개+copy)는 프레임당 약 5-9회라 몫이 작습니다.

## 10. cbar2: 연달아 붙은 barrier 합치기 (07:45 빌드, 기본 켜짐, `DM_VK_CBAR2=0`이면 예전처럼)

- 메모리 전용 barrier(버퍼나 이미지 barrier 없음, dependency flag 없음)는 바로 기록하지 않고 명령 버퍼별로 보관합니다.
- 그 명령 버퍼에 GPU 작업이 기록되기 직전에, 보관한 barrier를 **하나로 합쳐** 기록합니다. src와 dst stage, access는 보관한 것들의 합집합입니다.
  - GPU 작업 기록 시점은 s63 래퍼로 잡습니다: draw, dispatch, copy, fill, update, clear, render pass/rendering 시작과 끝, query, timestamp, 명령 버퍼 end. mesh draw는 exec.c에서 직접 flush합니다.
- 보관 대상: vk_img_barrier_all(full)과 vkCmdPipelineBarrier 래퍼를 지나는 메모리 전용 barrier(r89 앞뒤 barrier 포함).
- 버퍼나 이미지 barrier가 있는 호출은 보관된 것을 먼저 기록하고, 그 다음 그대로 기록합니다.
- 명령 버퍼 begin에서는 버려진 기록이 남긴 보관분을 버립니다(이어 붙이지 않음).
- 정확성: 사이에 명령이 없는 barrier 여러 개는 합집합 barrier 하나와 같거나 약합니다(실행 의존과 가시성 모두). 상태 명령(bind, dynamic state, push constant, descriptor)은 barrier의 순서 대상이 아닙니다.
- 남는 위험: s63 래퍼를 지나지 않는 GPU 작업 명령이 있으면 그 명령 뒤로 barrier가 밀립니다.
  - host 코드의 vkCmd 사용은 전수 확인했습니다. 래퍼 밖은 상태 명령, mesh draw(flush 추가), ootgap 측정용 timestamp/query reset뿐입니다.
  - 브리지(nvn_eden_cmd_*)는 상태 명령만 기록합니다.
- 카운터: `cbar2: guest barriers kept full so far …` 줄(1024개부터 2의 거듭제곱 번째마다)에 다음이 나옵니다.
  - 유지된 guest barrier의 전환 종류: dispatch 뒤 / 그 밖 작업 뒤 × 다음이 dispatch / draw·pass / copy·fill / counter·uniform / 기타
  - 보관, 기록, 합쳐진 barrier 수
- 셀프테스트 180-185:
  - 3개를 합집합 1개로 기록
  - 작업이 사이에 있으면 2개
  - 다른 cb의 작업은 보관분을 건드리지 않음
  - 이미지 barrier 앞에 보관분을 먼저 기록
  - 래퍼 경유 보관
  - begin에서 버림
- 기대: "full 2개" gap 89.8개/프레임과 연속 drain이 1개로 바뀝니다. gap 20.5 ms에서 **약 3-6 ms 감소**를 추정합니다(TS_FIX 같은 측정 조건 기준).

## 11. cbar2 결과 (ab_tsdiag_075004, 07:50 KST, OoT import, TS_FIX)

- 셀프테스트 COPY 62/62 PASS. 오류는 기존 RED 1줄(SetComputeWorkloadPriority)뿐입니다.
- ootgap(보정 측정):
  - span **23.68 ms** = timed pass 12.21 + 경계 0.14 + **gap 11.33**
  - 직전 cbar1 실행: span 38.03 = pass 17.24 + gap 20.51(gap 비율 54% → 48%)
  - 120프레임 창의 장면이 같은지는 보장할 수 없습니다. 절대값보다 비율과 카운터로 봅니다.
- cbar2 카운터:
  - barrier 보관 195만, 실제 기록 128만, 합쳐짐 67만(**기록된 barrier 35% 감소**)
  - ootgap의 "full 2개" 수(84.7개, 6.12 ms)와 r107 full/s(12.3k)는 보관 시점에 세므로 합치기 전 숫자입니다. 감소가 반영되지 않습니다.
- 유지된 guest barrier(full)의 전환 분포(26.2만 기준):
  - **그 밖의 작업 → draw/pass 18.0만(69%)**
  - dispatch → 기타 3.2만, 그 밖 → 기타 2.2만, 그 밖 → dispatch 1.0만, 그 밖 → counter/uniform 1.0만, dispatch → draw 0.7만
  - 즉 pass와 pass 사이(렌더 결과를 다음 pass가 읽는 연쇄)의 guest barrier가 대부분입니다.
- fps: 22-29 뒤 36-41(같은 장소인지 확인 필요). 무거운 pass: #239 dispatch 4.31 ms.
- 다음 후보:
  - pass와 pass 사이 guest barrier를 full 대신 실제 의존으로 줄이는 것입니다. 예를 들어 attachment write → fragment/compute read와 같은 graphics 범위 barrier입니다.
  - s64(barrier를 pass 안에 미뤄 두기)와 함께 다시 볼 수 있습니다. 이전 측정에서 s64의 fps 변화는 없었습니다.
  - 남은 gap 11.3 ms 중 이 몫이 얼마인지는 전환별 gap 시간 카운터가 있어야 정할 수 있습니다.

## 12. stage2s1 뒤 프레임 한도 (읽기만, ab_tsdiag_080904 = stage1, ab_tsdiag_082824 = stage2s1)

| | stage1 | stage2s1 |
|---|---|---|
| draw/프레임 | 1,330-1,608 | 1,785-2,665(장면이 약 1.6배 무거움) |
| replay/프레임 | 9.3-11.6 ms | 12.3-17.4 ms |
| replay/draw | 약 7 µs | 약 6-7 µs |
| 그중 descriptor set / pipeline key | 2.7-3.1 / 2.4-2.7 ms | 3.6-5.5 / 3.0-4.9 ms |
| descriptor set 재사용 | 약 10% | 약 6% |
| part cut(s61) | 제출당 0.94회, cut당 draw 222-230 | 1.15회, 376-390 |
| part 경계 | 5.6회, 1.48 ms(회당 0.26 ms) | 6.9회, 4.70 ms(회당 0.68 ms) |
| completion 스레드의 GPU 대기 | 537 ms/s | 320 ms/s(GPU가 더 많이 놂) |
| replay의 slot 대기 | 3 ms/s | 11 ms/s(replay는 GPU를 거의 기다리지 않음) |
| fps | 36-37 | 28-29 |

- **한도는 CPU 쪽(게임 + replay)입니다.** GPU busy는 약 20 ms인데 프레임은 약 35 ms입니다. GPU는 part 경계에서 다음 part 제출을 기다립니다(회당 0.68 ms).
- **stage2s가 CPU 일을 늘리지는 않았습니다.** draw당 replay 시간이 같습니다. 경계가 늘어난 것은 프레임당 draw가 많은 장면이기 때문입니다.
- 창 닫기, 병합, copy-in 기록은 dispatch 몇 개분의 명령이라 replay 몫이 작습니다. 창 수가 프레임당 약 2-3개입니다.
- **다음 후보**:
  - (가) replay의 descriptor set. 프레임당 3.6-5.5 ms이고 94%가 새로 만들어집니다. 재사용 키를 확인해 늘리면 2-4 ms 절감을 추정합니다.
  - (나) pipeline key. 3-4.9 ms입니다. 계산 결과를 캐시하면 1-3 ms를 추정합니다.
  - (다) 게임 스레드 CPU 시간 측정. 지금 로그에는 게임 스레드 CPU 시간이 없습니다("CPU -"). 프레임 35 ms 중 replay 15 ms를 뺀 나머지가 게임인지 대기인지 가르는 카운터가 필요합니다.

## 13. (가)(나)(다) 조사 결과 (읽기만, ab_tsdiag_082824)

**12절의 (가)(나) 해석은 틀렸습니다.** 로그 이름이 실제 일과 다릅니다.

- (가) "descriptor sets 4.73 ms"
  - 이 값은 `g_vkst.ns_desc`이고, Eden 모드에서는 `eden_payload` 시간입니다(gfx_vk_eden.inc:1192-1196). 각 binding의 UBO, SSBO, texture를 찾아 채우는 일이고, set 할당은 들어 있지 않습니다.
  - graphics는 이미 push descriptor를 씁니다. 131,072 draw 중 push가 118,894회입니다. 그중 같은 내용이라 건너뛴 것(s35)은 1,707회(1.4%)뿐입니다.
  - "reused 29332, new 388155"는 실행 시작부터의 누적값입니다. 600프레임당 82k-114k개 늘어나므로 프레임당 약 150-190개입니다. push가 아닌 경로(push_descriptor=0인 pipeline, 또는 exec.c:2798의 비-Eden 경로)에서 나옵니다. 94%라는 숫자는 이 작은 부분의 비율입니다.
  - 드라이버는 `descriptor_buffer=true push_descriptor=true`를 지원합니다. 어느 쪽을 쓸지는 nvn_shader_bridge.cpp:718-719에서 Eden pipeline layout이 정합니다.
  - draw당 비용은 약 1.8 µs로, hybrid 전후가 같습니다: ab_import_071100 4.17 ms/2292 draw = 1.82 µs, 082824 4.73/2665 = 1.78 µs. **비용은 draw 수에 비례하며, 새 회귀는 아닙니다.**
- (나) "pipeline keys 3.88 ms"
  - 이 값은 `ns_pipe`이고, eden_draw에서 payload를 뺀 나머지 전부입니다(gfx_vk_eden.inc:1299).
  - eden_prof로 나누면 draw당 key 0.20 µs, binds and state 0.85 µs, vertex buffers 0.35 µs입니다. binds and state 안에서 push 기록이 0.45 µs입니다.
  - 프레임당으로 바꾸면 key 0.53 ms, binds and state 2.3 ms(push 1.2 ms), vertex buffers 0.93 ms입니다.
  - key는 s45에서 이미 32%를 재사용하고 있습니다(kept 42k, made 89k). 남은 몫은 최대 0.4 ms입니다.
- 결론
  - draw당 replay CPU는 payload 1.8 + 나머지 1.45 = 약 3.2 µs입니다. 2,665 draw면 약 8.6 ms이고, replay 14.8 ms의 나머지 약 6 ms는 dispatch, 경계, pass 시작입니다.
  - 큰 한 방은 없습니다. 의미 있는 후보는 payload를 stage별로 memo하는 것뿐입니다. stage 입력(shader, cbuf의 handle 단어, UBO 바인딩)이 바뀌지 않았으면 그 stage의 entry를 다시 찾지 않습니다.
  - 그 전에 payload를 UBO, SSBO, image 몫으로 나누는 측정이 먼저 필요합니다.
- (다) 게임 스레드 CPU
  - "CPU -"는 `dm_thread_cpu`의 weak stub이 출력합니다(host/dm/gfx/nvn/hle_nvn.c:2101). 실제 구현은 host/dm/host/hle_thread.c:970에 있지만, NXN1X 빌드(source/src/nvn_host/CMakeLists.txt:42)에 링크되지 않습니다.
  - Eden에서는 KScheduler::SwitchThread가 이미 guest 스레드별 시간을 `cur_thread->AddCpuTime(m_core_id, tick_diff)`로 쌓습니다(source/src/core/hle/kernel/k_scheduler.cpp:355).
  - (14절에서 실제로는 strong 정의 대신 hook 포인터를 썼습니다.) 따라서 `dm_thread_cpu`의 strong 정의를 core 쪽에 두면 됩니다. 프로세스의 KThread 목록을 돌며 GetCpuTime 차이를 이름/우선순위별로 합치고, NCE 코어 host 스레드(cpu_manager.cpp:171 RunThread)의 `QueryThreadCycleTime`을 함께 보면 됩니다. 그러면 게임 일과 대기를 가를 수 있습니다.

## 14. cpudiag 진단 빌드 (08:40 KST, 로그만, 기본 꺼짐, TSDIAG cmd에서 켬)

- `DM_VK_PAYDIAG=1`
  - Eden draw timing 줄과 같은 주기(2^17 draw)로 "paydiag:" 두 줄을 남깁니다.
  - 한 줄은 draw 하나의 payload 시간을 종류별로 나눕니다: uniform, storage, image+sampler, texel. 각 종류의 draw당 entry 수도 함께 남깁니다.
  - 다른 줄은 graphics stage별로 두 비율을 셉니다:
    - 입력이 바로 앞 draw와 같은 비율. 입력은 shader, UBO 주소/크기, SSBO의 cbuf 단어, handle, driver segment 바인딩입니다.
    - entry가 같은 비율.
  - "입력 같은데 entry 다름" 수도 셉니다. 이것은 입력 키가 놓치는 경우입니다.
  - 해시는 시간 측정 구간 밖에서 계산합니다.
- `NXN1X_THREADCPU=1`
  - KScheduler::SwitchThread가 AddCpuTime과 같은 tick을 스레드별 잠금 없는 표에 더합니다(sv1 NoteThreadTime).
  - hle_nvn.c의 `dm_thread_cpu`는 `nvnh_thread_cpu_hook`을 통해 sv1 ThreadCpuLine을 부릅니다. 그러면 "threads up to frame … CPU" 줄에 다음이 나옵니다: present 수, 프레임 ms, 바쁜 코어 수, 코어별 idle %, 상위 8개 스레드(tid, prio, core)의 프레임당 run/wait ms.
  - 프로세스 스레드 목록은 돌지 않습니다. 잠금 없이 목록을 돌면 스레드 생성/종료와 경쟁하기 때문입니다.
- 판정
  - (가) payload 재사용의 기대치는 다음 곱으로 추정합니다: stage별 "입력 같음 %" × 그 stage의 몫, 그 결과를 프레임당 draw 수에 곱합니다.
  - 기준:
    - 입력 같음이 30% 미만이면 재사용은 접습니다.
    - 50% 이상이고 "입력 같은데 entry 다름"이 거의 0이면 재사용을 설계합니다.
  - (다) 프레임 약 35 ms 동안:
    - 게임 메인 스레드 run이 25 ms 이상이면 한도는 게임 CPU입니다.
    - run은 작고 wait가 크면 한도는 동기화입니다(GPU, replay, present 대기).

## 15. 비동기 replay (s46) A/B와 s46pc 빌드 (09:13 KST)

- A/B (ab_import_090657 ASYNC=0, ab_async1_090429 ASYNC=1, OoT, 같은 exe와 장소)
  - 게임 스레드의 submit+present는 17-20 ms에서 0.3-0.5 ms로 줄었습니다.
  - fps는 30-33으로 그대로입니다.
  - 이유: replay 스레드가 프레임 내내 바쁩니다(29-33 ms). 같은 replay가 1.5배 느려졌습니다: replay 13-14.5 ms가 18.5-22.6 ms로, descriptor sets가 draw당 1.74 µs에서 2.59 µs로 늘었습니다. 모든 항목이 같은 비율이라 느린 코어로 봅니다.
  - 게임은 여전히 프레임당 5-8 ms를 기다립니다. texture finalize drain이 600프레임에 약 500회 있기 때문입니다.
  - 화면 문제 지표는 없었습니다.
- s46pc 빌드
  - (1) replay 스레드에 guest 코어 스레드 정책을 겁니다: Critical 우선순위, P코어 CPU set, power throttling 해제(nvn_layer.cpp ReplayThreadPolicy). job마다 가장 빠른 코어 등급에서 돌았는지 세어 "s46pc:" 줄에 남깁니다.
  - (2) nvnTextureFinalize가 큐를 기다리지 않습니다. host 쪽 단계(ft_del, mtl_tex_release 힌트)를 객체 사본과 함께 큐의 job(kind 2)으로 넣어 순서대로 처리합니다.
    - 근거: 명령 목록은 텍스처를 tex_snap 사본으로 들고 있고, pool 등록은 job별 사본(mtl_reg_tex)입니다.
    - 예외: 큐에 그 텍스처의 present가 있으면 예전처럼 기다립니다. record 때 객체가 무효였던 경우(포인터를 그대로 보관, [RED] 줄)는 남는 위험입니다.
    - 다른 drain(program, pool, queue, sync)은 그대로 둡니다. program은 명령이 게임의 program 객체를 직접 가리키기 때문입니다.
    - DM_VK_FIN_DEFER=0이면 예전처럼 기다립니다.
    - 결정 로직은 s46_fin.h에 있고, 단위 테스트 t_s46_fin.c(10개)를 통과했습니다.
  - (3) paydiag 해시 시간을 descriptor sets와 pipeline keys에서 뺍니다.
- 판정
  - 성공 기준: ASYNC=1에서 "s46pc" fast 비율 90% 이상, replay가 ASYNC=0 수준(13-15 ms), s46 "game waited" 1 ms 미만, texture drain 프레임당 0.05 미만, fps 38 이상.
  - fast 비율이 높은데도 replay가 느리면, 원인은 코어가 아니라 경합(메모리나 잠금)입니다.

### 15.1 s46pc 회귀와 s46pc2 (09:20 KST)

- s46pc의 ASYNC=1 실행(ab_async1_091855)은 모든 장면이 반토막이었습니다. 가벼운 장면이 22-42 fps(이전 52-60)였고, 그때 GPU 0.34 ms, replay busy 2.7 ms, 게임 대기 0.04 ms, FIN 지연은 0건이었습니다.
- 같은 빌드의 ASYNC=0(ab_import_091646)은 정상이었습니다.
- 원인: replay 스레드를 TIME_CRITICAL로 guest 코어 스레드 6개와 같은 P코어 set에 둔 것으로 봅니다. 그 스레드들, 또는 그들이 기다리는 보통 우선순위 스레드와 경합한 것입니다.
- 무거운 구간도 draw당 11.4 µs로 ASYNC=0(8.6 µs)보다 느렸습니다. 이번 실행은 100% P코어였으므로, 지난번 1.5배도 코어 종류보다 경합 쪽으로 봅니다.
- 읽기만으로 확인한 것: 게임과 replay가 함께 잡는 g_sub_mu는 program/queue finalize 때뿐이라 원인이 아닙니다. 메모리 대역폭, guest 페이지 watch fault, pool lookup 잠금은 로그만으로 가릴 수 없습니다.
- s46pc2: 기본 정책은 power throttling 해제와 ABOVE_NORMAL입니다. DM_VK_S46_POLICY 0-3은 진단용입니다. ASYNC1 cmd에서 NXN1X_THREADCPU=1을 켭니다.

## 16. 파이프라인 멈춤과 30초 캐시 저장 (pcache1, 09:31 KST)

- 9.3초 멈춤(ab_async1_092551 130.5-139.8 s): 처음 만난 compute 셰이더 4개가 차례로 동기 컴파일됐습니다(개당 약 3.1 s). 디스크 캐시(vulkan.bin, vulkan_pipelines.bin, 버전 20)는 정상입니다.
- 장면 진입 2-3 s(매 실행)와 게임 중 약 200 ms 튐: s48 range import(한 번에 130-180 ms)와 zero-copy pool import 때문입니다. 아직 손대지 않았습니다.
- 30초 주기: TickNvnDriverCache가 새 파이프라인이 있을 때마다 30초 간격으로 61 MB 드라이버 캐시를 다시 씁니다. pcache1부터는 플레이 중에 빌드가 20초 조용할 때, 늦어도 5분마다만 저장합니다. NXN1X_PCACHE_PERIODIC=1이면 예전처럼 30초마다 저장합니다.
- s67c 진단: compute 파이프라인 한 개가 50 ms를 넘으면 단계별 시간(flow, translate, passes, spirv+module, layout)과 드라이버 빌드 시간을 남기고, 1 s를 넘으면 SLOW로 표시합니다.
