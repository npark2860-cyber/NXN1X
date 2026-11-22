# NXN1X 2.0

Windows ARM64용 Nintendo Switch / Switch 2 에뮬레이터입니다. NVIDIA N1X(RTX Spark) 노트북에서 게임을 네이티브 코드 실행(NCE)과 NVN 호스트 레이어로 돌립니다.

*English below.*

## 무엇인가

- **뼈대: SnapRYU 2.0.** NXN1X 2.0은 SnapRYU 2.0을 뼈대로 만들었습니다. SnapRYU 2.0은 이름만 남았고 Ryujinx 기반이 아닙니다. 다만 대상 기기가 다르고(SnapRYU: Snapdragon X Elite / Adreno, NXN1X: NVIDIA N1X), NCE 실행 경로(명령 처리, x18/FPCR, 패처)는 NXN1X가 따로 고친 별개 구현입니다.
- **에뮬레이터 코어: Eden**(yuzu 계열). 게임 로더, HLE 서비스, 셰이더 변환기, Vulkan 렌더러가 여기서 왔습니다.
- **NVN 호스트 레이어: nxnative 계열**(`host/`). 게임의 NVN/NVN2 그래픽 호출을 받아 Windows Vulkan으로 재생합니다.

## 왜 N1X용인가

NVIDIA N1X(RTX Spark) 노트북은 Windows ARM64에 NVIDIA GPU를 단 새 조합입니다. 기존 에뮬레이터는 이 기기에서 잘 맞지 않습니다.

- x64용 에뮬레이터는 Windows의 x64 에뮬레이션 계층 위에서 돌아 느립니다.
- ARM64용 작업은 대부분 Snapdragon/Adreno(Turnip 드라이버) 기준이라, N1X의 NVIDIA 드라이버와 코드 경로가 맞지 않습니다.
- N1X에서만 드러나는 문제도 있습니다. 예: 외부에서 가져온(imported) 메모리에서 정점을 읽으면 매우 느림(정점 미러로 해결), P코어/E코어가 섞인 CPU, Adreno 기준으로 잡혀 있던 기본값들.

NXN1X는 이 기기를 기준으로 NCE(게임 ARM64 코드를 직접 실행)와 NVN 호스트 레이어(게임의 그래픽 호출을 NVIDIA Vulkan으로 재생), DLSS를 맞춘 에뮬레이터입니다.

## 주요 기능

| 영역 | 내용 |
|---|---|
| CPU | Windows NCE: 게임 ARM64 코드를 직접 실행. x18·FPCR 처리, 패처, 크래시 링 |
| GPU | NVN 호스트 레이어 → Vulkan 리플레이, Eden 파이프라인. Switch 2(NVN2)의 Ampere 셰이더를 Maxwell IR로 낮춰 변환 |
| 업스케일 | NVIDIA DLSS SR (Streamline), 게임의 NVN2 DLSS 호출을 실제 DLSS로 처리 |
| 프레임 생성 | DLSS-G, 광학 보간(F10으로 실행 중 토글) |
| 게임 파일 | NUP(SPEC v1), NSP/XCI/NCA, 덤프 폴더(`exefs/` + `romfs/`), 폴더 → NUP 변환기 `host/scripts/make_nup.py` |
| 기본 해상도 | Switch 2 게임은 1x (게임별 설정이 우선) |

## 게임 상태 (2026-11-22)

자세한 내용과 알려진 문제는 [기술 문서](docs/NXN1X_2.0_TECHNICAL.md) 5장과 8장에 있습니다.

- **Fire Emblem Fortune's Weave**: 부팅·진행 가능. 튜토리얼 광장이 실기보다 어두움. 프레임 생성을 켠 채 시작하면 device lost(끄고 시작하면 정상).
- **Xenoblade Chronicles DE / 2 (Switch 2 Edition)**: 타이틀·인트로 영상 정상(DLSS 범위 수정).
- **Zelda: Ocarina of Time (Switch 2)**, **Donkey Kong Bananza**, **Mario Kart World**, **Hyrule Warriors: AoI**: 기술 문서 참고.

## 빌드

MSYS2 clangarm64, CMake, Ninja가 필요합니다.

```powershell
$env:PATH="C:\msys64\clangarm64\bin;C:\msys64\usr\bin;$env:PATH"
ninja -C build/release
```

결과: `build\release\bin\NXN1X.exe`. 설정과 로그는 `%APPDATA%\NXN1X\` (로그: `log\nxn1x_log.txt`).

## 문서

- [NXN1X 2.0 기술 문서 (한국어)](docs/NXN1X_2.0_TECHNICAL.md) / [English](docs/NXN1X_2.0_TECHNICAL_EN.md)
- [2.0 변경 사항 (2026-11-22)](docs/RELEASE_2026-11-22.md)
- 설계 문서: [`design/`](design/)

## 라이선스

GPL-3.0-or-later. 부분별 라이선스는 [NOTICE.md](NOTICE.md)를 보세요. Nintendo 키·펌웨어·게임 파일은 포함하지 않으며 배포하지 않습니다. 직접 합법적으로 덤프한 파일을 사용하세요.

---

# NXN1X 2.0 (English)

A Nintendo Switch / Switch 2 emulator for Windows ARM64. It runs games on an NVIDIA N1X (RTX Spark) laptop with native code execution (NCE) and an NVN host layer.

## What it is

- **Skeleton: SnapRYU 2.0.** NXN1X 2.0 is built on the SnapRYU 2.0 skeleton. SnapRYU 2.0 keeps only the name and is not Ryujinx-based. The target device differs (SnapRYU: Snapdragon X Elite / Adreno; NXN1X: NVIDIA N1X), and the NCE execution path (instruction handling, x18/FPCR, patcher) is NXN1X's own separate implementation.
- **Emulator core: Eden** (yuzu lineage): loaders, HLE services, shader recompiler, Vulkan renderer.
- **NVN host layer: derived from nxnative** (`host/`): takes the game's NVN/NVN2 graphics calls and replays them on Windows Vulkan.

## Why N1X

NVIDIA N1X (RTX Spark) laptops are a new combination: Windows ARM64 with an NVIDIA GPU. Existing emulators do not fit them well.

- x64 emulators run through Windows' x64 emulation layer and are slow.
- Most ARM64 work targets Snapdragon/Adreno (the Turnip driver), so its drivers and code paths do not match the N1X's NVIDIA driver.
- Some problems show up only on the N1X: vertex reads from imported memory are very slow (fixed with a vertex mirror), the CPU mixes P-cores and E-cores, and defaults were tuned for Adreno.

NXN1X is built around this device: NCE (game ARM64 code runs directly), an NVN host layer (the game's graphics calls replayed on NVIDIA Vulkan), and DLSS.

## Features

| Area | What |
|---|---|
| CPU | Windows NCE: game ARM64 code runs natively; x18 / FPCR handling, patcher, crash ring |
| GPU | NVN host layer → Vulkan replay with Eden pipelines; Switch 2 (NVN2) Ampere shaders lowered to Maxwell IR |
| Upscaling | NVIDIA DLSS SR through Streamline; the game's NVN2 DLSS calls run real DLSS |
| Frame generation | DLSS-G and optical interpolation (F10 toggles it while running) |
| Game files | NUP (SPEC v1), NSP/XCI/NCA, dump folders (`exefs/` + `romfs/`), folder → NUP packer `host/scripts/make_nup.py` |
| Default resolution | Switch 2 games at 1x (a per-game setting wins) |

## Game status (2026-11-22)

See sections 5 and 8 of the [technical document](docs/NXN1X_2.0_TECHNICAL_EN.md).

- **Fire Emblem Fortune's Weave**: boots and plays. The tutorial square is darker than on the console. Starting with frame generation on loses the device (fine when started with it off).
- **Xenoblade Chronicles DE / 2 (Switch 2 Edition)**: title and intro movies correct (DLSS range fix).
- **Zelda: Ocarina of Time (Switch 2)**, **Donkey Kong Bananza**, **Mario Kart World**, **Hyrule Warriors: AoI**: see the technical document.

## Build

Requires MSYS2 clangarm64, CMake and Ninja.

```powershell
$env:PATH="C:\msys64\clangarm64\bin;C:\msys64\usr\bin;$env:PATH"
ninja -C build/release
```

Output: `build\release\bin\NXN1X.exe`. Settings and logs: `%APPDATA%\NXN1X\` (log: `log\nxn1x_log.txt`).

## Documents

- [NXN1X 2.0 technical document (English)](docs/NXN1X_2.0_TECHNICAL_EN.md) / [한국어](docs/NXN1X_2.0_TECHNICAL.md)
- [2.0 changes (2026-11-22)](docs/RELEASE_2026-11-22.md)
- Design documents: [`design/`](design/)

## License

GPL-3.0-or-later. Per-component licenses are in [NOTICE.md](NOTICE.md). No Nintendo keys, firmware or game files are included or distributed; use your own legally obtained dumps.
