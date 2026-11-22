# ARM64 validation report, 2026-11-19

## DLSS-only output targets and HDR settings

The current [packaged ARM64 executable](../build/NXN1X-DLSS-ARM64/NXN1X.exe) uses the six Streamline DLLs alongside it. Windows denied replacement of the previous executable in build/release/bin. The new package was linked with the Release response file in a separate directory. SHA-256 is 15DC4BDF702C5771A8A487CF9BACF0B372D79151F569CDF45A302459F960E74A.

Graphics now summarizes a single N1X Vulkan device and has an HDR checkbox. Internal HDR settings and guest-rate controls are omitted. Sharpness follows the selected spatial filter. The checkbox writes the existing HDR policy. The presenter still prefers an SDR swapchain, and enabling this setting does not establish HDR10 scanout. Advanced Graphics selects OFF, Auto, 30, 60, 90 or 120 FPS. The build disables LSFG and removes its hotkey. Auto targets the active monitor, measures accepted original tickets against host steady time, and backs off if measured native cadence falls. Driver dynamic MFG is used in Auto only when reported supported. Fixed mode uses a supported integer ratio and may undershoot a nonintegral target. Output targets do not change either guest's counter frequency.

Changes cover common DLSS settings, Streamline and the Vulkan renderer/presenter, dlss_frame_target.h, LSFG build configuration, Qt graphics and hotkey configuration, the overlay, and Korean UI strings. The [Qt widget test](../build/dlss-output-ui.log) passed all six selections, global inheritance, legacy LSFG config, N1X visibility and HDR on/off persistence. The [ARM64 timing test](../build/dlss-target-math.log) passed synthetic 19.2 and 31.25 MHz timing, multiplier limits and pause reset. The [packaged N1X probe](../build/dlss-output-package-probe.log) reported successful FG status and five maximum additional frames. No game was run. Actual frame generation, GPU cost, HDR output and real display cadence are unmeasured for this build.

## Current ARM64 Release with automatic FG inputs

The CPU and Vulkan selectors drive the visibility of emulated GPU controls in both global and per-game dialogs. A running session uses its actual NVN path after fallback. NVN hides emulated texture decoding/recompression, anisotropy overrides, buffer-cache and GPU unswizzle controls. It retains asynchronous shader compilation, which feeds the NVN Eden backend directly. Non-NVN modes disable the DLSS choice and skip interposer initialization. The non-game Qt fixture checked seven advanced controls on an NVN-to-Emulation path change, as well as selection exclusivity and restoration of global settings. The new live path filter is in configure_cpu.*, configure_graphics.*, configure_dialog.cpp, configure_per_game.cpp and shared_widget.*. SDK present FPS and accepted original FPS are logged over the same interval when FG is active. These are SDK and application counts, not a physical display measurement.

The graphics advanced panel now selects Off, NVIDIA DLSS Frame Generation ×2, or LSFG ×2. It saves global and per-game selections and shows the legacy flow controls only for LSFG. The four Mario Kart World and Hyrule Warriors FPS patch controls, the frame pacing target and the old NPU controls are omitted from the panel. Existing config data is retained. The diagnostic environment override is still supported.

Explicit OP_DLSS now tags supported guest depth and motion images without a title profile. A typed pair takes priority over inferred resources. Both the Eden and bundled NVN draw paths observe camera constants. The automatic path seeks a rigid view, perspective projection and matching view-projection in a bound vertex uniform buffer. It compares 16 asynchronous depth and motion samples against camera reprojection and requires three matching measurements to infer the RG attachment and its units/signs. It does not synthesize object motion. Separate or packed matrices, asymmetric projections, ambiguous RG attachments, mixed producing shaders, stationary scenes during calibration and unsupported views can prevent detection. Diagnostic logs include the camera candidate, confirmed motion convention, tag results and SDK option changes.

The implementation changes are in the host NVN replay (gfx_vk_exec.c, gfx_vk_eden.inc, gfx_vk_core.c, gfx_vk_fg_auto.inc, gfx_vk_fg_math.h and nvn_fg_input.h), Vulkan Streamline, device and renderer integration, the common DLSS setting, Qt graphics advanced panel, Korean translations, and the standalone capability and math tests. [ARM64 Release link](../build/dlss-fg-auto-final.log), [Qt settings fixture](../build/dlss-ui-fixture.log), [camera/motion math fixture](../build/dlss-fg-auto-math.log), [N1X packaged runtime probe](../build/dlss-fg-auto-probe.log). The probe reports presentId=1, privateData=1, DLSS-G status=0 and maxGenerated=5. The probe does not run a presentation workload. No game was launched for this build. Detection coverage, frame pacing, visual correctness, performance changes, gameplay Vulkan validation and 4K60 remain unmeasured. Earlier measurements below refer to earlier builds.

| Stage | Status | Evidence |
| --- | --- | --- |
| P0 source audit and ARM64 capability detection | PASS on N1X | Logical device creation, adapter queries, SR output readback, FG state query, orderly shutdown. See [capabilities](DLSS_ARM64_CAPABILITIES.md). |
| Switch 2 explicit SR, fixed input | Pixel smoke test passed; synchronization gate failed | Output initialization green, measured red at center. The validation layer reported two hazards within NGX. No proof for moving content, game resources, or all pixels. |
| Game inputs | Partial | 33 pass attachments and final color captured around ticket 31. Final color 5760×3240 RGBA8 in NVN 2X, guest extent 2880×1620. Some candidate depth images were constant 1.0. No frame-matched dense motion, camera or HUD-less/UI alpha verification. |
| Game SR | Blocked | No explicit OP_DLSS observed in the sampled Bananza opening. Preserve the original guest command and subresource semantics before enabling replacement. The documented Fire Emblem Fortune's Weave test title was absent from the two game libraries checked. |
| Game compute upscaler | Policy verified; gameplay output pending | The initial N1X test build selected original NDSR. The 2X run reached the title screen. No recognized NDSR dispatch was observed there. GSR1 remains selectable for comparison. Native HMMA hardware acceleration and speedup remain unverified. |
| FG ×2, Reflex pacing, 4K output | Blocked | Adapter and pre-render state queries succeeded; no valid game inputs tagged, no generated frames, no external present trace. No quality, latency, power, HDR/VRR or 4K60 result. |

## Regression runs

Donkey Kong Bananza 040074A01BF12000 on driver 621.24, NVN 2X, warmed app cache, no installed DLSS runtime in game folder. Each short run covers the opening; the table reports two successive roughly ten-second application frame summaries. Runs were sequential for the first two rows. Power and clocks were not locked. The timing samples cannot establish a throughput gain or statistical equivalence.

| Build and settings | Window FPS | Output verification |
| --- | --- | --- |
| [Previous release, SR/FG off](../build/dk-cache-Bayonetta-cab2523b/user/log/nxn1x_log.txt) | 3.8, 4.1 | Original game renderer |
| [First audit build, SR/FG off, capture disabled](../build/dk-cache-Bayonetta-cf25d083/user/log/nxn1x_log.txt) | 5.5, 3.2 | Original game renderer |
| [Delivery build, source inspection and validation enabled, SR/FG off](../build/dk-cache-Bayonetta-90e7972b/user/log/nxn1x_log.txt) | 4.4, 3.1 | Selected N1X native NDSR policy; first 34 seconds, zero validation errors. Warnings remain. |
| Game SR or FG on | Unmeasured | No generated presents or physical scanout measurement |

During first game-layer validation, one simultaneous vkDeviceWaitIdle/vkQueueSubmit report appeared. BlitScreen::WaitIdle now takes the shared NVN submit mutex after scheduler.Finish(). It did not recur in two short retests. The previous-release control did not reproduce the race during its short run. [Initial layer log](../build/dk-cache-Bayonetta-950de187/validation.log), [final layer log](../build/dk-cache-Bayonetta-90e7972b/validation.log).

An additional 2X 300-frame sample with save copy and capture had median frame 213.18 ms and median GPU 207.66 ms. Not an untouched 4K reference. A separate accelerated navigation run used the Res3_4X setting. [2X timing](../build/dk-cache-Bayonetta-59564125/user/log/nxn1x_log.txt), [title screen](../build/dk-cache-Bayonetta-59564125/frame_000300.png), [pass evidence](../build/dk-cache-Bayonetta-de8b767e/user/log/nxn1x_log.txt), [captures](../build/dlss-capture).

## Changed files and reproduction

- host/dm/windows/gfx_vk_exec.c selects NDSR mode in ndsr_blit, records the first known direct vkCmdDispatch on request, and inspects resident OP_DLSS inputs.
- host/dm/windows/gfx_vk_present.c samples the final image extent, format, fence and ticket in present_out.
- source/src/video_core/renderer_vulkan/vk_blit_screen.cpp serializes the compositor's device idle wait with NVN.
- tools/dlss_probe.cpp and tools/dlss_sr_fixture.cpp implement the standalone ARM64 capability probe and offscreen SR readback.
- tools/test_ndsr.ps1 starts a process-local diagnostic A/B run. Original is the default and selects guest compute with DM_VK_NDSR=1. GSR1 requires an explicit override. Select the same game and save on each launch.
- docs/DLSS_ARM64_ARCHITECTURE.md, docs/DLSS_ARM64_CAPABILITIES.md and this report record the integration and measurements.

~~~powershell
$env:Path='C:/msys64/clangarm64/bin;C:/msys64/usr/bin;'+$env:Path
C:/msys64/clangarm64/bin/ninja.exe -C build/release bin/NXN1X.exe
~~~

The normal executable was open during linking. The delivered NXN1X-ndsr.exe was linked from the same Release Ninja response file with a separate output name. [Build log](../build/dlss-final-build.log), [binary hash](../build/dlss-final-link.log).

~~~powershell
./tools/test_ndsr.ps1 -Mode Original
./tools/test_ndsr.ps1 -Mode GSR1
~~~

For gameplay verification, stand in the same mine scene on both runs, rotate the camera, then stop it. Preserve the first run's nxn1x_log.txt before opening the second. Check the original log for NDSR original: shader= and stage=vkCmdDispatch and record its shader key and dispatch size; the GSR1 log should identify the replaced network and rectangles. Compare brightness, fine lines and motion, and match frame-time windows after shader warm-up. If the original route renders incorrectly, the GSR1 switch allows immediate comparison. No shader-cache deletion is needed.

## Gameplay regression reported after the initial test

Bananza showed a solid purple screen on the original-compute build. The user's log records original guest compute as the default and advancing presentation tickets, but contains no per-program NDSR dispatch trace. The [upstream fix #506](https://github.com/Eglantine-Community/Eglantine/pull/506) addressed skipped tensor programs including a third program in the mine. Its separate frame-fence change addressed command corruption. NXN1X already selects tensor lowering in LowerNativeProgram; that fact does not confirm Vulkan image writes. The recovery binary NXN1X-ndsr-fix.exe restores GSR1 by default. Opt-in native diagnostics now record selection, pipeline/descriptor failures and command recording per known program, including indirect dispatch. The isolated [NVN 2X boot log](../build/dk-cache-Bayonetta-611935ae/user/log/nxn1x_log.txt) confirms policy selection. Visual verification of the affected mine scene is pending.
## Native NDSR storage image fix

The current ARM64 Release binary is build/release/bin/NXN1X.exe. The NVN Vulkan texture and descriptor caches now distinguish storage views from sampled views. Storage descriptors request storage usage and identity component mapping. The launcher defaults to the original guest compute path. Changes are in host/dm/windows/gfx_vk.h, gfx_vk_tex.c, gfx_vk_exec.c and tools/test_ndsr.ps1. The Release build linked successfully ([build log](../build/ndsr-storage-build.log)). Gameplay output and frame times in the affected scene await user testing.
## ARM64 Vulkan frame generation integration, November 19

The opt-in Release executable at `build/release/bin/NXN1X.exe` uses the Streamline interposer and the six ARM64 DLLs beside it in `streamline/`. The [Release link log](../build/dlss-fg-integrated.log) and [packaged-runtime capability probe](../build/dlss-fg-release-probe.log) record this stage. The executable and all six DLLs were checked for PE machine AA64. The standalone probe created a Vulkan device on the N1X, returned success from slDLSSGGetState with status 0 and a maximum of five generated frames, and shut down successfully. It did not present any frames.

The implementation spans renderer_vulkan/nvn_streamline.*, renderer_vulkan/renderer_vulkan.cpp, renderer_vulkan/vk_present_manager.*, vulkan_common/vulkan_device.cpp, the video_core and yuzu build definitions, the NVN DLSS command recorder, and host/dm/windows/{gfx_vk_exec.c,gfx_vk_present.c,gfx_vk_eden.inc,gfx_vk_fg_capture.inc,nvn_fg_input.h}. tools/dlss_probe.cpp adds the packaged FG-only probe. tools/test_ndsr.ps1 exposes the runtime and capture switches independently.

The SDK receives final color through the intercepted swapchain. Guest depth and motion can be tagged on their producing command buffer. Camera constants and the tags are associated with the original presentation ticket. Missing, stale or ambiguous inputs prevent activation. A shader profile is required even for explicit DLSS commands because the camera matrices must be identified. For Switch 1 NVN replay the profile also identifies the motion-producing fragment shader and color attachment. Other GPU emulation paths have no implemented input mapping here.

Profiles are read from `streamline/profiles/<16 lowercase hex program ID>.fg` and contain whitespace-separated vertex and fragment shader hashes (hex), vertex uniform slot, camera-to-world and projection byte offsets, motion color attachment index (decimal), and X/Y motion scales. A zero fragment hash selects explicit DLSS command inputs. Profile assumptions include a single main perspective view, row-major unjittered matrices and verified motion direction and scale. No verified game profile ships in this build.

For a user-run capture, execute `./tools/test_ndsr.ps1 -FrameGeneration -CaptureFGInputs` and press Ctrl+Shift+F11 in a moving 3D scene. The launcher prints the unique capture directory. The two-frame diagnostic collects bounded 32×32 image samples and up to 4096 bytes per readable vertex uniform slot for the first 32 distinct vertex shaders each frame. A file is a candidate binding and requires interpretation. It is not a RenderDoc capture.

Gameplay was reserved for the user. Source FPS, actual presented FPS, GPU frame times, frame pacing, visual correctness, Vulkan validation during FG presentation and FG-disabled performance regression remain unmeasured. The probe result does not establish 4K60 or generated-frame display.


## 2026-11-19 — Tears of the Kingdom NVN FG discovery and FPS telemetry

The archived NXN1X game log for 0100f2c0115b6000 shows the loader active at 123.056 s and automatic discovery enabled at 124.372 s. It contains no accepted camera candidate, confirmed motion convention, or successful FG present count. That establishes the earliest missing event. The old log does not contain the game's uniform contents or rejection counters.

| Stage | Changed files | Verification |
| --- | --- | --- |
| NVN camera | host/dm/windows/gfx_vk_fg_math.h, gfx_vk_fg_auto.inc, tools/test_nvn_fg_math.c | Synthetic original matrix layout and packed transposed 3×4 view plus transposed projection/product passed. Corrupt products and truncated buffers were rejected. |
| Depth and motion | host/dm/windows/gfx_vk_exec.c | Allow a base-subresource view and multiple fragment shaders in the camera pass. Still require a unique supported velocity attachment, matching depth and reprojection calibration. Every 60 presents the replay reports eligible draws, UBO reads, accepted cameras, candidate and usable pairs, samples, calibration and tag attempts. Actual game bindings remain unverified. |
| Output | source/src/video_core/renderer_vulkan/renderer_vulkan.cpp, vk_present_manager.h/.cpp, nvn_streamline.h/.cpp | Count accepted single-image NVN tickets independently of FG geometry eligibility. Report original FPS and the SDK present count per second over two-second windows. With FG off the output statistic uses accepted originals. Cadence unit test passed. |
| Release | build/NXN1X-TOTK-FG-ARM64/NXN1X.exe, libusb and streamline runtime | Release link succeeded. PE machine 0xAA64. Packaged runtime probe on Spark N1X returned Vulkan 1.4.357, device creation success, DLSS-G state success, status 0, maxGenerated 5. See build/totk-fg-final-release.log and build/totk-fg-probe.log. |

Game performance, image quality, actual generated frames and the 4K 60 FPS target are pending the user's game run. No game was launched for this build.

### Follow-up after the 30 FPS game run

The log for title 0100f2c0115b6000 recorded FG off, original and output both 30.00 FPS, and repeated windows of 60 accepted cameras with zero motion candidates, readbacks and tag attempts. Output geometry eligibility also failed. The old geometry message does not distinguish the individual rectangle and transform checks.

Changed gfx_vk_exec.c to evaluate subsequent passes sharing the current frame's identified camera depth allocation, retaining velocity-format and reprojection checks. Added bounded attachment format diagnostics. Changed renderer_vulkan.cpp, vk_present_manager.h/.cpp and nvn_streamline.h/.cpp to pass the game viewport as a Streamline backbuffer subrectangle, including for letterboxed single-image output. Unhandled guest crop/transform combinations are logged. Release build, existing camera/cadence regression tests and packaged N1X SDK probe passed. Package: build/NXN1X-FG-Region-ARM64. No post-change game run or output FPS measurement.

### 17:20 TotK run — region build remained at 30 FPS

Title 0100f2c0115b6000. The 53.64 s window recorded 30.00 original and 30.00 output FPS, stage depth-motion-tag. At 54.68 s the preceding 60 presents yielded 60 camera observations and zero motion candidates, usable pairs, readbacks or tag attempts. Initial format diagnostics repeatedly sampled a single 800×448 R32_SFLOAT color target alongside D32_SFLOAT depth; the previous log limit hid later pass formats. These observations do not establish whether a dense velocity field exists elsewhere in the game.

Changed host/dm/windows/gfx_vk_exec.c and gfx_vk_eden.inc. Search other depth-bearing graphics passes in the same camera frame and perspective aspect, and inspect sampled depth/velocity pairs within individual compute dispatches. Reuse the existing GPU readback, calibration and SDK submission path. Distinct pass signatures share the bounded diagnostic budget. ARM64 Release linked; existing camera and cadence regressions passed. The packaged runtime reported DLSS-G state success on N1X. Package build/NXN1X-FG-Inputs-ARM64. This change covers additional resource bindings; no captured game bindings or post-change gameplay FPS verification.

### Optical-flow fallback for NVN originals

Windows ARM64 release package: build/NXN1X-OpticalFlow-ARM64/NXN1X.exe. SHA-256 98fff706d9e3da0bca0bcfa1b0bcde7de073d578e3a60cdb38818e5420cdbaca. Build completed with ENABLE_LSFG=OFF.

- Upstream implementation: zortos293/Lichii commit 60ee2ef62a9e844e8cf494ccfe1e310814b998d4. Imported renderer_vulkan/present/fsr3_frame_gen.{cpp,h} and host_shaders/fsr3_{optical_flow,frame_interpolation}.comp. Adjusted image copy synchronization for reuse. Updated video_core and host_shaders CMake lists.
- renderer_vulkan.{cpp,h}, nvn_streamline.{cpp,h}: select optical interpolation for original NVN tickets without eligible DLSS inputs, suppress concurrent generation, measure real-time original cadence, and request up to three intermediate frames. Duplicate tickets, discontinuities and surface changes do not generate from unrelated history. The fallback uses the composited color including HUD.
- vk_present_manager.cpp: enable storage views and paced output without LSFG. Generated frames use available fenced presentation slots. If a slot is busy, the group can contain fewer generated frames. Optical output FPS counts accepted originals plus accepted generated presents. This counter is not a physical scanout measurement.
- Regression: existing cadence tests passed for both guest clocks and 30/60/90/120 targets. Both GLSL modules compiled and passed spirv-val. A separate N1X Vulkan compute test using the release SPIR-V at 3840x2160 measured horizontal motion of -1 flow pixel for a synthetic moving patch. The three center-pixel intensity samples between black and white originals were 82, 127 and 173 at phases 0.25, 0.50 and 0.75. See build/flow-gpu-test.log and build/test-optical-gpu.cpp.
- Packaged Streamline probe passed on N1X (status 0, maxGenerated 5). Game boot, in-game switchback, image quality, sustained presentation rate and physical display refresh remain unverified; no game was launched.

### Donkey Kong Bananza — fractional 60 FPS optical presentation

The previous run averaged 37.27 accepted originals/s at 240.11 s with interpolation off. The 242.13 s reporting window (58 originals, 22 generated) included the transition into 2x at 241.38 s. The old planner rounded down the multiplier using a smoothed estimate updated roughly once per second. The old present pool also tried only its first free entry; its log cannot attribute the remaining missing frames to individual causes.

- Changed renderer_vulkan/dlss_frame_target.h, renderer_vulkan.{cpp,h} and present/fsr3_frame_gen.{cpp,h}. The optical planner budgets output slots against consecutive original arrival times, distributes fractional demand across frame pairs, preserves history on original-only intervals and limits deferred debt. SDK generation still uses its supported integer multipliers.
- Changed vk_present_manager.{cpp,h}. Nonblocking admission scans free frames for a compatible ready fence. Per-window optical scheduling logs requested/submitted totals and acquisition misses by lock contention, empty pool, dimensions/view or pending fence. Accepted-present FPS continues to be logged separately.
- Extended tools/test_dlss_frame_target.cpp. Synthetic one-second intervals gave 43+17, 29+31, 30+30 and 20+40 at target 60; no extras at 60 or 90 originals/s. Rate changes, 30→120, missed-slot carry, duplicate tickets and discontinuities passed. See build/fg-fractional-tests.log.
- Release linked and packaged in build/NXN1X-Fractional-FG-ARM64. ARM64 machine type verified. N1X 4K Vulkan compute smoke test and packaged Streamline device probe passed. Actual post-change game FPS, frame pacing and display scanout remain unmeasured; the user will run the game.
