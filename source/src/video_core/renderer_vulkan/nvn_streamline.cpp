#include "video_core/renderer_vulkan/nvn_streamline.h"
#include "video_core/renderer_vulkan/fg_status.h"
#include "nvn_fg_input.h"
#include "common/dynamic_library.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/dlss_fg_setting.h"

#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include "video_core/renderer_vulkan/dlss_frame_target.h"
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>
#ifdef NXN1X_STREAMLINE_SDK
#include <windows.h>
#include <sl.h>
#include <sl_dlss_g.h>
#include <sl_dlss.h>
#include <sl_matrix_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#endif

namespace Vulkan::NvnStreamline {
namespace {
std::atomic_bool active{};
#ifdef NXN1X_STREAMLINE_SDK
std::mutex mutex;
PFun_slShutdown* shutdown_sl{};
PFun_slGetNewFrameToken* new_token{};
PFun_slSetTagForFrame* set_tags{};
PFun_slSetConstants* constants{};
PFun_slIsFeatureSupported* supported{};
PFun_slGetFeatureFunction* lookup{};
PFun_slEvaluateFeature* evaluate{};
PFun_slDLSSSetOptions* sr_options{};
bool fg_requested{}, sr_history{};
uint64_t sr_network{};
PFun_slDLSSGSetOptions* fg_options{};
PFun_slDLSSGGetState* fg_state{};
PFun_slPCLSetMarker* marker{};
struct Inputs {
    sl::FrameToken* token{};
    uint32_t index{};
    uint64_t ticket{};
    uint64_t generation{};
    uint32_t width{}, height{};
    sl::float4x4 world{}, projection{};
    sl::float2 scale{}, jitter{};
    bool camera{}, tagged{}, ambiguous{}, multiple_inputs{}, explicit_inputs{}, reset{};
};
Inputs pending{}, previous{}, presenting{};
uint64_t generation{};
std::array<Inputs, sl::MAX_FRAMES_IN_FLIGHT> sealed{};
nvn_fg_profile profile{};
std::atomic_bool ready{};
bool enabled{};
using Clock = std::chrono::steady_clock;
Clock::time_point sample_start{};
unsigned sample_originals{}, sample_optical{};
bool sample_generation{};
uint64_t counted_ticket{};
const char* present_stage = "startup";
OriginalCadence cadence;
unsigned max_generated{}, multiplier{1};
HWND render_hwnd{};
unsigned monitor_hz{};
Clock::time_point next_monitor_query{}, retry_auto{};
double native_baseline{};
bool supports_dynamic{}, dynamic_active{};
bool dlssg_failed{};
unsigned sample_shown{}, sample_gen_presents{}, zero_run{};   // dlssgcheck2: frames DLSS-G presents showed, its presents, presents in a row that showed none   // dlssgcheck: DLSS-G was on and the SDK presented under half the originals (11-21: 1 or 2 presents in 2 s, the picture stood still)

unsigned MonitorHz(Clock::time_point now) {
    if (now < next_monitor_query) return monitor_hz;
    next_monitor_query = now + std::chrono::seconds(1);
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    monitor_hz = render_hwnd &&
        GetMonitorInfoW(MonitorFromWindow(render_hwnd, MONITOR_DEFAULTTONEAREST), &info) &&
        EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode) && mode.dmDisplayFrequency > 1
            ? mode.dmDisplayFrequency : 0;
    return monitor_hz;
}
uint64_t last_ticket{};

template <typename F>
bool Function(sl::Feature feature, const char* name, F*& fn) {
    void* ptr{};
    if (lookup(feature, name, ptr) != sl::Result::eOk || !ptr) return false;
    fn = reinterpret_cast<F*>(ptr);
    return true;
}

// dllver (NPARK 11-21: whether DLSS 5 can come by replacing the DLLs alone): every DLL in streamline\ with its file version and machine, the SDK version this
// build was compiled against, the presets its sl_dlss.h names (the runtime has no call that lists them: what the DLL takes is known only by its version), the
// preset asked for each mode (eDefault: the runtime's own per mode), and each mode's optimal render size for 2560x1440 as the loaded runtime answers it.
// version.dll is loaded at run time (no import library added)
void LogRuntime() {
    using SizeFn = DWORD(WINAPI*)(LPCWSTR, LPDWORD);
    using InfoFn = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, LPVOID);
    using QueryFn = BOOL(WINAPI*)(LPCVOID, LPCWSTR, LPVOID*, PUINT);
    const HMODULE ver = LoadLibraryExW(L"version.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto size_fn = ver ? reinterpret_cast<SizeFn>(GetProcAddress(ver, "GetFileVersionInfoSizeW")) : nullptr;
    const auto info_fn = ver ? reinterpret_cast<InfoFn>(GetProcAddress(ver, "GetFileVersionInfoW")) : nullptr;
    const auto query_fn = ver ? reinterpret_cast<QueryFn>(GetProcAddress(ver, "VerQueryValueW")) : nullptr;
    const auto dir = Common::FS::GetExeDirectory() / L"streamline";
    WIN32_FIND_DATAW fd{};
    const HANDLE h = FindFirstFileW((dir / L"*.dll").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            const auto path = dir / fd.cFileName;
            std::string version = "unknown";
            if (size_fn && info_fn && query_fn) {
                DWORD dummy{};
                const DWORD n = size_fn(path.c_str(), &dummy);
                std::vector<char> buf(n);
                VS_FIXEDFILEINFO* fi{}; UINT len{};
                if (n && info_fn(path.c_str(), 0, n, buf.data()) && query_fn(buf.data(), L"\\", reinterpret_cast<LPVOID*>(&fi), &len) && fi && len)
                    version = fmt::format("{}.{}.{}.{}", HIWORD(fi->dwFileVersionMS), LOWORD(fi->dwFileVersionMS), HIWORD(fi->dwFileVersionLS), LOWORD(fi->dwFileVersionLS));
            }
            uint16_t machine = 0;   // the PE header's machine (0xaa64 ARM64, 0x8664 x64)
            if (std::ifstream pe{path, std::ios::binary}) {
                uint32_t off = 0; pe.seekg(0x3c); pe.read(reinterpret_cast<char*>(&off), 4);
                pe.seekg(off + 4); pe.read(reinterpret_cast<char*>(&machine), 2);
            }
            std::string name; for (const wchar_t* p = fd.cFileName; *p; ++p) name += static_cast<char>(*p < 128 ? *p : '?');
            LOG_INFO(Render_Vulkan, "dllver: streamline\\{} version {} machine {:#x}{}", name, version, machine, machine == 0xaa64 ? " (ARM64)" : " (not ARM64: will not load here)");
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    if (ver) FreeLibrary(ver);
    LOG_INFO(Render_Vulkan, "dllver: built against Streamline SDK {}.{}.{}; presets sl_dlss.h names: J (10), K (11, DLAA/Quality/Balanced default, transformer), L (12, Ultra Performance default), "
             "M (13, Performance default); A-D removed, E F deprecated, G H I N revert to default; asked for every mode: eDefault (the runtime's own per mode). A newer nvngx_dlss.dll "
             "with its own models is taken by the same calls; a preset letter past N needs a newer sl_dlss.h",
             SL_VERSION_MAJOR, SL_VERSION_MINOR, SL_VERSION_PATCH);
    PFun_slDLSSGetOptimalSettings* optimal{};
    if (Function(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", optimal)) {
        static constexpr std::pair<sl::DLSSMode, const char*> modes[]{{sl::DLSSMode::eMaxQuality, "Quality"}, {sl::DLSSMode::eBalanced, "Balanced"},
            {sl::DLSSMode::eMaxPerformance, "Performance"}, {sl::DLSSMode::eUltraPerformance, "Ultra Performance"}, {sl::DLSSMode::eDLAA, "DLAA"}};
        std::string line;
        for (const auto& [mode, label] : modes) {
            sl::DLSSOptions o{}; o.mode = mode; o.outputWidth = 2560; o.outputHeight = 1440;
            sl::DLSSOptimalSettings s{};
            const auto r = optimal(o, s);
            line += r == sl::Result::eOk ? fmt::format(" {} {}x{}", label, s.optimalRenderWidth, s.optimalRenderHeight) : fmt::format(" {} (result {})", label, int(r));
        }
        LOG_INFO(Render_Vulkan, "dllver: the loaded DLSS runtime's render sizes for 2560x1440:{}", line);
    }
}

bool CameraConstants(const Inputs& f, sl::Constants& c) {
    const auto& p = f.projection;
    const float z0 = -p[3].z / p[2].z;
    const float z1 = (p[3].w - p[3].z) / (p[2].z - p[2].w);
    if (!std::isfinite(z0) || !std::isfinite(z1) || z0 * z1 <= 0 ||
        std::abs(p[2].w) < 0.99f || std::abs(p[2].w) > 1.01f ||
        std::abs(p[3].w) > 1e-5f || !p[0].x || !p[1].y) return false;
    c.cameraViewToClip = p;
    sl::matrixFullInvert(c.clipToCameraView, p);
    const bool reset = f.reset || !previous.camera || last_ticket + 1 != f.ticket ||
                       previous.width != f.width || previous.height != f.height;
    const auto& prev = reset ? f : previous;
    sl::float4x4 view_delta{}, clip_to_view{};
    sl::calcCameraToPrevCamera(view_delta, f.world, prev.world);
    sl::matrixMul(clip_to_view, c.clipToCameraView, view_delta);
    sl::matrixMul(c.clipToPrevClip, clip_to_view, prev.projection);
    sl::matrixFullInvert(c.prevClipToClip, c.clipToPrevClip);
    for (const auto* matrix : {&c.clipToCameraView, &c.clipToPrevClip, &c.prevClipToClip})
        for (auto& row : matrix->row)
            if (!std::isfinite(row.x) || !std::isfinite(row.y) || !std::isfinite(row.z) || !std::isfinite(row.w)) return false;
    const auto axis = [&](unsigned i, float sign = 1.f) {
        return sl::float3{sign*f.world[i].x, sign*f.world[i].y, sign*f.world[i].z};
    };
    c.cameraPos = axis(3); c.cameraRight = axis(0); c.cameraUp = axis(1); c.cameraFwd = axis(2, p[2].w);
    c.cameraNear = std::min(std::abs(z0), std::abs(z1));
    c.cameraFar = std::max(std::abs(z0), std::abs(z1));
    c.cameraFOV = 2 * std::atan(1.f / std::abs(p[1].y));
    c.cameraAspectRatio = std::abs(p[1].y / p[0].x);
    c.depthInverted = std::abs(z0) > std::abs(z1) ? sl::eTrue : sl::eFalse;
    c.cameraMotionIncluded = sl::eTrue;
    c.motionVectors3D = sl::eFalse;
    c.reset = reset ? sl::eTrue : sl::eFalse;
    c.mvecScale = f.scale; c.jitterOffset = f.jitter;
    return true;
}
#endif
}

bool Active() { return active.load(std::memory_order_acquire); }

std::shared_ptr<Common::DynamicLibrary> Open() {
    const char* requested = std::getenv("NXN1X_DLSS_FG");

#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    fg_requested = requested ? std::string_view(requested) == "1" : bool(Settings::dlss_frame_generation.GetValue());
    const auto dir = Common::FS::GetExeDirectory() / L"streamline";
    const auto mod = LoadLibraryExW((dir / L"sl.interposer.dll").c_str(), nullptr,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!mod) {
        LOG_ERROR(Render_Vulkan, "DLSS-G runtime load failed ({})", GetLastError());
        return {};
    }
    auto library = std::make_shared<Common::DynamicLibrary>(static_cast<void*>(mod));
    PFun_slInit* init{};
    if (!library->GetSymbol("slInit", &init) || !library->GetSymbol("slShutdown", &shutdown_sl) ||
        !library->GetSymbol("slGetNewFrameToken", &new_token) ||
        !library->GetSymbol("slSetTagForFrame", &set_tags) ||
        !library->GetSymbol("slIsFeatureSupported", &supported) ||
        !library->GetSymbol("slGetFeatureFunction", &lookup) ||
        !library->GetSymbol("slEvaluateFeature", &evaluate) ||
        !library->GetSymbol("slSetConstants", &constants)) return {};
    const wchar_t* path = dir.c_str();
    const sl::Feature features[]{sl::kFeatureDLSS, sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
    sl::Preferences prefs{};
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = fg_requested ? 4 : 1;
    prefs.pathsToPlugins = &path;
    prefs.numPathsToPlugins = 1;
    prefs.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                  sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    prefs.renderAPI = sl::RenderAPI::eVulkan;
    prefs.engineVersion = "NXN1X-1";
    prefs.projectId = "ab2efce3-9b51-4891-a73a-53429294495d";
    const auto result = init(prefs, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "Streamline initialization failed ({})", int(result));
        return {};
    }
    active.store(true, std::memory_order_release);
    LOG_INFO(Render_Vulkan, "Streamline Vulkan loader active; DLSS-G awaiting game inputs");
    // Keep the dispatch DLL mapped until Vulkan destroys the last object. Also clear
    // the SDK session if renderer construction unwinds before its destructor runs.
    auto owner = std::shared_ptr<void>(mod, [library](void*) { Shutdown(); });
    return std::shared_ptr<Common::DynamicLibrary>(owner, library.get());
#else
    LOG_ERROR(Render_Vulkan, "This build has no ARM64 Streamline SDK");
    return {};
#endif
}

void Shutdown() {
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    if (active.exchange(false)) {
        cadence = {};
        multiplier = 1;
        if (enabled) fg_options(sl::ViewportHandle{0}, sl::DLSSGOptions{});
        sr_options = nullptr;
        sr_history = false;
        shutdown_sl();
        ready = enabled = false; presenting = previous = pending = {}; sealed = {}; profile = {}; generation = last_ticket = 0;
    }
#endif
}

void Configure(void* physical_device, void* window) {
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    render_hwnd = static_cast<HWND>(window);
    next_monitor_query = retry_auto = {};
    native_baseline = 0;
    sample_start = {};
    sample_originals = 0;
    sample_generation = false;
    counted_ticket = 0;
    sample_optical = 0;
    if (!Active()) return;
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physical_device;
    if (supported(sl::kFeatureDLSS, adapter) == sl::Result::eOk)
        Function(sl::kFeatureDLSS, "slDLSSSetOptions", sr_options);
    LOG_INFO(Render_Vulkan, "Guest NVN DLSS SR available={}", sr_options != nullptr);
    LogRuntime();
    if (!fg_requested) return;
    PFun_slReflexSetOptions* reflex{};
    sl::ReflexOptions options{};
    options.mode = sl::ReflexMode::eLowLatency;
    const bool available = supported(sl::kFeatureDLSS_G, adapter) == sl::Result::eOk &&
        Function(sl::kFeatureDLSS_G, "slDLSSGSetOptions", fg_options) &&
        Function(sl::kFeatureDLSS_G, "slDLSSGGetState", fg_state) &&
        Function(sl::kFeaturePCL, "slPCLSetMarker", marker) &&
        Function(sl::kFeatureReflex, "slReflexSetOptions", reflex) && reflex(options) == sl::Result::eOk;
    if (!available) { LOG_ERROR(Render_Vulkan, "DLSS-G device or Reflex unavailable"); return; }
    sl::DLSSGState capabilities{};
    if (fg_state(sl::ViewportHandle{0}, capabilities, nullptr) != sl::Result::eOk ||
        capabilities.numFramesToGenerateMax == 0) {
        LOG_ERROR(Render_Vulkan, "DLSS-G maximum generation count unavailable");
        return;
    }
    max_generated = capabilities.numFramesToGenerateMax;
    supports_dynamic = capabilities.bIsDynamicMFGSupported == sl::eTrue;
    dynamic_active = false;
    const auto path = Common::FS::GetExeDirectory() / "streamline" / "profiles" /
                      fmt::format("{:016x}.fg", Settings::GetCurrentProgramID());
    std::ifstream in(path);
    nvn_fg_profile p{};
    if (in >> std::hex >> p.vertex >> p.fragment >> std::dec >> p.camera_ubo >>
        p.camera_to_world >> p.view_to_clip >> p.motion_attachment >> p.mv_scale_x >> p.mv_scale_y &&
        p.vertex && p.camera_ubo < 16 && p.motion_attachment < 8 &&
        std::isfinite(p.mv_scale_x) && std::isfinite(p.mv_scale_y) && p.mv_scale_x && p.mv_scale_y) {
        profile = p;
    } else LOG_INFO(Render_Vulkan, "DLSS-G automatic NVN input discovery for title {:016x}", Settings::GetCurrentProgramID());
    ready.store(true, std::memory_order_release);
#endif
}

namespace {
#ifdef NXN1X_STREAMLINE_SDK
// srmode: each DLSS mode's render range for an output size, asked of the runtime once per output size (slDLSSGetOptimalSettings); under mutex
uint32_t range_ow, range_oh, rmin[5][2], rmax[5][2]; bool have_ranges; sl::DLSSMode cur_mode = sl::DLSSMode::eOff;
constexpr sl::DLSSMode order[4]{sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced, sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraPerformance};
constexpr const char* names[4]{"Quality", "Balanced", "Performance", "Ultra Performance"};
void SrRanges(uint32_t ow, uint32_t oh) {
    if (range_ow == ow && range_oh == oh) return;
    range_ow = ow; range_oh = oh; have_ranges = false;
    PFun_slDLSSGetOptimalSettings* optimal{};
    if (!Function(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", optimal)) return;
    have_ranges = true;
    for (int m = 0; m < 4; ++m) {
        sl::DLSSOptions q{}; q.mode = order[m]; q.outputWidth = ow; q.outputHeight = oh; sl::DLSSOptimalSettings s{};
        if (optimal(q, s) != sl::Result::eOk) { rmin[m][0] = rmin[m][1] = 1; rmax[m][0] = rmax[m][1] = 0; continue; }
        rmin[m][0] = s.renderWidthMin; rmin[m][1] = s.renderHeightMin; rmax[m][0] = s.renderWidthMax; rmax[m][1] = s.renderHeightMax;
        LOG_INFO(Render_Vulkan, "srmode: {} for {}x{}: render {}x{} (min {}x{}, max {}x{})", names[m], ow, oh, s.optimalRenderWidth, s.optimalRenderHeight,
                 s.renderWidthMin, s.renderHeightMin, s.renderWidthMax, s.renderHeightMax);
    }
}
#endif
}  // namespace

// srfit (11-21 23:48 run, NPARK: "the picture stops and goes again" at an even 45-49 fps): the inputs of OoT's lower render shares (854x480 to 1138x640)
// are under the smallest size the runtime gives Performance (and Quality, Balanced: 1280x720); Ultra Performance takes 853x480 alone. An input under the
// feature's range is not one the feature takes, and its picture is suspected of standing still while the size stays there. The size DLSS is given:
// the render size raised to Performance's smallest (keeping its shape) when under it; ootdl_prep.comp resamples the colour, depth and motion to it.
// 0: no change (ranges unknown, or the input already in range)
extern "C" int nvn_sr_fit(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, uint32_t out[2]) {
    out[0] = iw; out[1] = ih;
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    if (!Active()) return 0;
    SrRanges(ow, oh);
    if (!have_ranges || !iw || !ih || iw == ow) return 0;
    const uint32_t mw = rmin[2][0], mh = rmin[2][1];
    if (!mw || !mh || (iw >= mw && ih >= mh)) return 0;
    uint32_t w = iw < mw ? mw : iw, h = (uint32_t)((uint64_t)ih * w / iw);
    if (h < mh) { h = mh; w = (uint32_t)((uint64_t)iw * h / ih); }
    if (w > ow) w = ow; if (h > oh) h = oh;
    out[0] = w; out[1] = h;
    return 1;
#else
    return 0;
#endif
}
extern "C" int nvn_sr_evaluate(void* cmd, const nvn_fg_image images[6], uint32_t iw, uint32_t ih,
                               uint32_t ow, uint32_t oh, const float mv[2], const float jitter[2],
                               float exposure, uint32_t flags, int reset, uint64_t network) {
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    if (!Active() || !sr_options) return -1;
    const sl::ViewportHandle vp{1}; // Keep scene reconstruction independent of FG presentation tags.
    sl::DLSSOptions opts{};
    // srmode (11-21, NPARK: black screen with OoT's dynamic resolution, 854x480 to 1920x1080 into 2560x1440): the mode was picked from the ratio alone
    // (854 x 3 > 2560 made a 3x input Performance's), and an input outside a mode's render range is not what that feature was made for. Now each mode's
    // range for this output is asked of the runtime once (slDLSSGetOptimalSettings min / max), the current mode is kept while it takes the input (no
    // new feature, no history reset at each step), else the first of Quality, Balanced, Performance, Ultra Performance whose range holds it; DLAA at 1:1
    SrRanges(ow, oh);

    // srmode2 (11-21 23:41 run: 32 mode changes in 80 s, each a new feature and a stall: OoT's sizes from 854x480 to 1138x640 fit no mode's range, as the
    // runtime gives Quality, Balanced and Performance 1280x720 to 2560x1440 and Ultra Performance 853x480 alone, so the pick went back and forth): a mode
    // is kept while its largest render size takes the input (the feature is made for up to that size; a smaller input is DLSS's dynamic resolution),
    // and the first pick is Performance, the middle of OoT's 33 % to 75 % range, while its largest size takes the input
    const auto fits_max = [&](int m) { return iw <= rmax[m][0] && ih <= rmax[m][1] && rmax[m][0] > 0; };
    sl::DLSSMode pick = sl::DLSSMode::eOff;
    if (iw == ow && ih == oh) pick = sl::DLSSMode::eDLAA;
    else if (have_ranges) {
        for (int m = 0; m < 4; ++m) if (order[m] == cur_mode && fits_max(m)) pick = cur_mode;
        if (pick == sl::DLSSMode::eOff && fits_max(2)) pick = order[2];   // Performance
        for (int m = 0; m < 4 && pick == sl::DLSSMode::eOff; ++m) if (fits_max(m)) pick = order[m];
    }
    if (pick == sl::DLSSMode::eOff)   // (no range holds it, or the runtime gave none: the nearest by ratio)
        pick = iw * 5ull <= ow * 2ull ? sl::DLSSMode::eUltraPerformance : iw * 9ull <= ow * 5ull ? sl::DLSSMode::eMaxPerformance :
               iw * 8ull <= ow * 5ull ? sl::DLSSMode::eBalanced : sl::DLSSMode::eMaxQuality;
    if (pick != cur_mode) {
        static unsigned changes{};
        if (++changes <= 32 || !(changes & (changes - 1)))
            LOG_INFO(Render_Vulkan, "srmode: {}x{} -> {}x{}: mode {} (was {}; {} changes)", iw, ih, ow, oh, int(pick), int(cur_mode), changes);
        cur_mode = pick;
    }
    opts.mode = pick;
    opts.outputWidth = ow;
    opts.outputHeight = oh;
    opts.preExposure = exposure > 0 && std::isfinite(exposure) ? exposure : 1.f;
    opts.colorBuffersHDR = (flags & 1) ? sl::eTrue : sl::eFalse;
    opts.useAutoExposure = (flags & 64) ? sl::eTrue : sl::eFalse;
    sl::FrameToken* frame = pending.token;
    auto result = sr_options(vp, opts);
    if (result == sl::Result::eOk && !frame) result = new_token(frame, nullptr);
    if (result != sl::Result::eOk || !frame) { sr_history = false; return result == sl::Result::eOk ? -2 : int(result); }
    sl::Constants c{};
    if (pending.camera && !pending.ambiguous) CameraConstants(pending, c);
    c.cameraAspectRatio = static_cast<float>(ow) / oh;
    c.cameraPinholeOffset = {0, 0};
    c.jitterOffset = {jitter[0], jitter[1]};
    c.mvecScale = {mv[0] / iw, mv[1] / ih};
    c.depthInverted = flags & 8 ? sl::eTrue : sl::eFalse;
    c.cameraMotionIncluded = sl::eTrue;
    c.motionVectors3D = sl::eFalse;
    c.reset = reset || !sr_history || network != sr_network ? sl::eTrue : sl::eFalse;
    result = constants(c, *frame, vp);
    const sl::BufferType roles[]{sl::kBufferTypeScalingInputColor, sl::kBufferTypeDepth,
        sl::kBufferTypeMotionVectors, sl::kBufferTypeScalingOutputColor,
        sl::kBufferTypeBiasCurrentColorHint, sl::kBufferTypeExposure};
    sl::Resource resources[6]{};
    sl::Extent extents[6]{};
    sl::ResourceTag tags[6]{};
    const sl::BaseStructure* inputs[7]{&vp};
    unsigned count = 1;
    for (unsigned i = 0; i < 6; ++i) {
        const auto& im = images[i];
        if (!im.image) continue;
        auto& r = resources[i];
        r = sl::Resource(sl::ResourceType::eTex2d, im.image, im.memory, im.view, im.layout);
        r.width = im.width; r.height = im.height; r.nativeFormat = im.format;
        r.usage = im.usage; r.flags = im.flags; r.mipLevels = r.arrayLayers = 1;
        extents[i] = {im.y, im.x, i == 5 ? 1u : i == 3 ? ow : iw, i == 5 ? 1u : i == 3 ? oh : ih};
        tags[i] = sl::ResourceTag(&r, roles[i], sl::ResourceLifecycle::eValidUntilEvaluate, &extents[i]);
        inputs[count++] = &tags[i];
    }
    if (result == sl::Result::eOk) result = evaluate(sl::kFeatureDLSS, *frame, inputs, count, cmd);
    sr_history = result == sl::Result::eOk;
    sr_network = network;
    return int(result);
#else
    return -1;
#endif
}

extern "C" int nvn_dlss_sr_wanted(void) { return Settings::dlss_upscale.GetValue() ? 1 : 0; }   // dlsstoggle: the setting (and its hotkey), read live by gfx_vk_exec.c ootdl_on
extern "C" int nvn_fg_collecting() {
#ifdef NXN1X_STREAMLINE_SDK
    // fgtoggle: the setting is read live (F10), so with frame generation switched off the replay collects nothing for it and its segments may go to the workers
    return Active() && ready && Settings::dlss_frame_generation.GetValue();
#else
    return 0;
#endif
}
extern "C" const nvn_fg_profile* nvn_fg_get_profile() {
#ifdef NXN1X_STREAMLINE_SDK
    if (Active() && ready.load(std::memory_order_acquire) && profile.vertex) return &profile;
#endif
    return nullptr;
}
extern "C" void nvn_fg_camera(const float* world, const float* projection) {
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    if (!ready) return;
    for (unsigned i = 0; i < 16; ++i)
        if (!std::isfinite(world[i]) || !std::isfinite(projection[i])) return;
    if (pending.camera && (std::memcmp(&pending.world, world, 64) ||
                           std::memcmp(&pending.projection, projection, 64))) pending.ambiguous = true;
    std::memcpy(&pending.world, world, 64);
    std::memcpy(&pending.projection, projection, 64);
    pending.camera = true;
#endif
}

extern "C" void nvn_fg_seal(uint64_t ticket) {
#ifdef NXN1X_STREAMLINE_SDK
    if (!Active()) return;
    std::scoped_lock lock{mutex};
    pending.ticket = ticket;
    sealed[ticket % sealed.size()] = pending;
    pending = {};
#endif
}

extern "C" void nvn_fg_tag(void* cmd, const nvn_fg_image* depth, const nvn_fg_image* motion,
                           uint32_t width, uint32_t height, const float* scale, const float* jitter, int reset,
                           int explicit_inputs) {
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    if (!Active()) return;
    if (!ready || pending.ambiguous || !width || !height ||
        !std::isfinite(scale[0]) || !std::isfinite(scale[1]) ||
        !std::isfinite(jitter[0]) || !std::isfinite(jitter[1])) return;
    if (presenting.token && generation + 1 - presenting.generation >= sealed.size()) return;
    if (pending.explicit_inputs && !explicit_inputs) return;
    if (pending.tagged && pending.explicit_inputs == bool(explicit_inputs)) {
        pending.multiple_inputs = true;
        return;
    }
    pending.tagged = pending.multiple_inputs = false;
    pending.explicit_inputs = explicit_inputs != 0;
    auto& frame = pending.token;
    if (new_token(frame, nullptr) != sl::Result::eOk || !frame) return;
    pending.index = *frame;
    pending.generation = ++generation;
    pending.width = width;
    pending.height = height;
    sl::Resource resources[2]{};
    sl::Extent extents[2]{};
    sl::ResourceTag tags[2]{};
    const nvn_fg_image* inputs[]{depth, motion};
    for (int i = 0; i < 2; ++i) {
        const auto& in = *inputs[i];
        auto& r = resources[i];
        r = sl::Resource(sl::ResourceType::eTex2d, in.image, in.memory, in.view, in.layout);
        r.width = in.width;
        r.height = in.height;
        r.nativeFormat = in.format;
        r.usage = in.usage;
        r.flags = in.flags;
        r.mipLevels = r.arrayLayers = 1;
        extents[i] = {in.y, in.x, width, height};
        tags[i] = sl::ResourceTag(&r, i ? sl::kBufferTypeMotionVectors : sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eOnlyValidNow, &extents[i]);
    }
    const auto result = set_tags(*frame, sl::ViewportHandle{0}, tags, 2, cmd);
    pending.tagged = result == sl::Result::eOk;
    pending.scale = {scale[0]/float(width), scale[1]/float(height)};
    pending.jitter = {jitter[0], jitter[1]};
    pending.reset = reset != 0;
    static unsigned reports{};
    if (reports++ < 4)
        LOG_INFO(Render_Vulkan, "DLSS-G guest depth and motion {}x{}, tagging result {}", width,
                 height, int(result));
#endif
}

void ResetHistory() {
#ifdef NXN1X_STREAMLINE_SDK
    if (!Active()) return;
    std::scoped_lock lock{mutex};
    sr_history = false;
    previous = pending = presenting = {};
    cadence = {};
    sealed = {};
    if (enabled && fg_options(sl::ViewportHandle{0}, sl::DLSSGOptions{}) == sl::Result::eOk)
        enabled = false;
#endif
}

unsigned OpticalTarget(uint64_t ticket) {
    if (!Settings::dlss_frame_generation.GetValue()) return 0;
    if (Settings::dlss_fg_source.GetValue() == 1) return 0;   // fgsource: DLSS-G only (no optical interpolation)
    const unsigned target = Settings::dlss_target_fps.GetValue();
#ifdef NXN1X_STREAMLINE_SDK
    std::scoped_lock lock{mutex};
    if (enabled) return 0;
    const auto& f = sealed[ticket % sealed.size()];
    // fgfill (11-21 23:48 run, NPARK: "it stutters enormously" with DLSS on): a frame with DLSS-G's inputs was left to DLSS-G alone, which makes whole multiples
    // only, so with the originals at 45-58 fps under the 60 target it generated nothing (multiplier 1) and the optical interpolation stood aside too: 45-58
    // fps went to a 60 Hz present unevenly, where before the inputs existed the optical interpolation filled each frame pair up to 60. Now the optical
    // interpolation fills whenever DLSS-G is not generating (enabled above), inputs or not
    (void)sealed; (void)f;
    return target ? target : MonitorHz(Clock::now());
#else
    return target;
#endif
}

void BeforePresent(uint64_t ticket, const uint32_t* region) {
#ifdef NXN1X_STREAMLINE_SDK
    if (!Active()) return;
    std::scoped_lock lock{mutex};
    if (!ready) return;
    const auto& f = sealed[ticket % sealed.size()];
    sl::Constants c{};
    present_stage = !Settings::dlss_frame_generation.GetValue() ? "switched-off" : Settings::dlss_fg_source.GetValue() == 2 ? "optical-chosen" : dlssg_failed ? "dlssg-not-presenting" : !ticket ? "geometry-or-ticket" :   // fgtoggle: F10 off turns DLSS-G off below
        ticket <= last_ticket || f.ticket != ticket ? "frame-ticket" :
        !f.camera ? "camera" : f.ambiguous ? "conflicting-cameras" :
        !f.tagged ? "depth-motion-tag" : f.multiple_inputs ? "multiple-inputs" :
        !f.token || generation - f.generation >= sealed.size() || uint32_t(*f.token) != f.index ? "token" :
        !CameraConstants(f, c) ? "camera-constants" :
        constants(c, *f.token, sl::ViewportHandle{0}) != sl::Result::eOk ? "sdk-constants" : "ready";
    bool valid = std::string_view(present_stage) == "ready";
    if (valid) {
        sl::Extent area{region[0], region[1], region[2], region[3]};
        sl::ResourceTag color{nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle{}, &area};
        if (set_tags(*f.token, sl::ViewportHandle{0}, &color, 1, nullptr) != sl::Result::eOk) {
            valid = false;
            present_stage = "backbuffer-region";
        }
    }
    const auto now = Clock::now();
    const unsigned preset = Settings::dlss_target_fps.GetValue();
    const unsigned target = preset ? preset : MonitorHz(now);
    unsigned requested = valid && target ? cadence.Multiplier(target, max_generated) : 1;
    bool dynamic = !preset && supports_dynamic && valid && cadence.fps > 0 &&
                   cadence.fps < target * .99;
    if (!preset) {
        if (enabled && native_baseline > 0 && cadence.fps > 0 && cadence.fps < native_baseline * .9)
            retry_auto = now + std::chrono::seconds(3);
        if (now < retry_auto) { requested = 1; dynamic = false; }
        if (!enabled) native_baseline = cadence.fps;
    }
    const bool generate = dynamic || requested > 1;
    if (generate != enabled || (enabled && (dynamic != dynamic_active || (!dynamic && requested != multiplier)))) {
        sl::DLSSGOptions o{};
        o.mode = dynamic ? sl::DLSSGMode::eDynamic : generate ?
            (preset ? sl::DLSSGMode::eOn : sl::DLSSGMode::eAuto) : sl::DLSSGMode::eOff;
        o.numFramesToGenerate = requested > 1 ? requested - 1 : 1;
        // (NX2navi nx2_sl.cpp: off without this flag frees the generator's resources and on makes them again inside the present, a 100-150 ms frame each time;
        // with it off only suspends. The cadence turns generation on and off as the original rate crosses the target: those were stutters too)
        o.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        const auto result = fg_options(sl::ViewportHandle{0}, o);
        if (result == sl::Result::eOk) {
            enabled = generate;
            dynamic_active = dynamic;
            multiplier = requested;

        }
        LOG_INFO(Render_Vulkan, "DLSS-G ticket {} original {:.2f} FPS target {} mode {} fixed multiplier {} result {}",
                 ticket, cadence.fps, target, int(o.mode), dynamic ? 0 : requested, int(result));
    }
    presenting = {};
    if (valid && enabled) {
        presenting = f;
        marker(sl::PCLMarker::ePresentStart, *f.token);
    }
#endif
}

void AfterPresent(uint64_t ticket, uint64_t original_ticket, bool accepted, bool generated) {
#ifdef NXN1X_STREAMLINE_SDK
    if (!Settings::dlss_frame_generation.GetValue() && !ready.load()) { VideoCommon::FgStatus::mode.store(VideoCommon::FgStatus::Off, std::memory_order_relaxed); return; }
    std::scoped_lock lock{mutex};
    const auto& f = presenting;
    const auto now = Clock::now();
    // fgoverlay (NPARK 11-21: "DLSS FG shows in the overlay oddly, late"): the FG row was the 2 s log window's: a change (F10, DLSS-G starting or standing
    // aside, the multiplier) showed up to 2 s later, and a window with both DLSS-G and optical presents counted only DLSS-G's frames. The row's state is now
    // this present's (DLSS-G when this present was a DLSS-G one that showed frames, optical when an optical frame went out in the last 0.25 s), and its rates
    // are a 0.5 s window of their own: originals, and every frame shown (DLSS-G's shown count, optical frames, originals presented without either)
    static Clock::time_point ov_start{}, ov_optical{}, ov_dlssg_at{};
    static u64 ov_orig{}, ov_out{};
    const auto ov_publish = [&](bool dlssg_now) {
        namespace F = VideoCommon::FgStatus;
        const bool on = Settings::dlss_frame_generation.GetValue();
        if (dlssg_now) ov_dlssg_at = now;
        dlssg_now = ov_dlssg_at != Clock::time_point{} && now - ov_dlssg_at < std::chrono::milliseconds(250);   // (held 0.25 s, as the optical state: no flicker between presents)
        const bool optical_now = ov_optical != Clock::time_point{} && now - ov_optical < std::chrono::milliseconds(250);
        F::mode.store(!on ? F::Off : dlssg_now ? F::Dlssg : optical_now ? F::Optical : F::Waiting, std::memory_order_relaxed);
        F::multiplier.store(multiplier, std::memory_order_relaxed);
        if (ov_start == Clock::time_point{}) { ov_start = now; ov_orig = ov_out = 0; return; }
        const double s = std::chrono::duration<double>(now - ov_start).count();
        if (s < 0.5) return;
        F::original_fps10.store(u32(ov_orig / s * 10 + .5), std::memory_order_relaxed);
        F::output_fps10.store(u32(ov_out / s * 10 + .5), std::memory_order_relaxed);
        ov_start = now; ov_orig = ov_out = 0;
    };
    if (generated) { if (accepted) { ++sample_optical; ++ov_out; ov_optical = now; ov_publish(false); } return; }
    if (accepted && original_ticket) cadence.Observe(original_ticket, now);
    else cadence = {};
    previous = {};
    bool ov_dlssg = false; u32 ov_shown = accepted ? 1u : 0u;   // fgoverlay: this present's frames shown
    if (enabled && f.ticket == ticket && f.token) {
        marker(sl::PCLMarker::ePresentEnd, *f.token);
        if (accepted) {
            previous = f; last_ticket = ticket;
        }
        // dlssgcheck2 (NPARK 11-21: "output under the originals at 51 s and 214 s, 0.5-1 fps"): that figure was the measure's, not DLSS-G's. The SDK's
        // numFramesActuallyPresented is what the last present showed (2 with one generated frame: NX2navi nx2_sl.cpp nx2sl_present_marker), and it was read
        // once a 2 s window and divided by the window, so 2 shown became "1 fps". Now it is read after each DLSS-G present and summed. A present that showed
        // nothing (or a status other than OK) twice in a row is output under the originals: DLSS-G goes off for the run at once, with the reason logged
        sl::DLSSGState ps{};
        const auto pr = fg_state(sl::ViewportHandle{0}, ps, nullptr);
        ++sample_gen_presents;
        if (pr == sl::Result::eOk && ps.status == sl::DLSSGStatus::eOk && ps.numFramesActuallyPresented > 0) { sample_shown += ps.numFramesActuallyPresented; zero_run = 0;
            ov_shown = ps.numFramesActuallyPresented; ov_dlssg = ps.numFramesActuallyPresented > 1; }
        else if (++zero_run >= 2 && !dlssg_failed) {
            dlssg_failed = true;
            if (fg_options(sl::ViewportHandle{0}, sl::DLSSGOptions{}) == sl::Result::eOk) enabled = false;
            LOG_WARNING(Render_Vulkan, "dlssgcheck: DLSS-G present at ticket {} showed {} frames (result {}, status {:#x}), {} such presents in a row: output under the originals, DLSS-G off for this run",
                        ticket, ps.numFramesActuallyPresented, int(pr), unsigned(ps.status), zero_run);
        }
    }
    if (accepted && original_ticket && original_ticket != counted_ticket) {
        ++sample_originals; ++ov_orig;
        counted_ticket = original_ticket;
    }
    ov_out += ov_shown; ov_publish(ov_dlssg);   // fgoverlay
    sample_generation |= enabled;
    if (sample_start == Clock::time_point{} || now - sample_start >= std::chrono::seconds(2)) {
        sl::DLSSGState state{};
        const auto result = ready ? fg_state(sl::ViewportHandle{0}, state, nullptr) : sl::Result::eErrorNotInitialized;
        if (sample_start != Clock::time_point{}) {
            const double seconds = std::chrono::duration<double>(now - sample_start).count();
            // (dlssgcheck2: the output of a window with DLSS-G presents is what those presents showed, summed per present, plus the originals presented without it)
            const double shown_fps = (sample_shown + (sample_originals > sample_gen_presents ? sample_originals - sample_gen_presents : 0)) / seconds;
            if (sample_optical && !sample_generation) {
                LOG_INFO(Render_Vulkan, "FG source=optical-flow original FPS={:.2f} output FPS={:.2f} originals={} generated={}",
                         sample_originals / seconds, (sample_originals + sample_optical) / seconds,
                         sample_originals, sample_optical);
            } else if (!sample_generation) {
                LOG_INFO(Render_Vulkan, "DLSS-G FG=off stage={} original FPS={:.2f} output FPS={:.2f} source=accepted-originals",
                         present_stage, sample_originals / seconds, sample_originals / seconds);
            } else if (result == sl::Result::eOk && state.status == sl::DLSSGStatus::eOk) {
                LOG_INFO(Render_Vulkan, "DLSS-G FG={} stage={} original FPS={:.2f} output FPS={:.2f} source=SDK shown={} over {} DLSS-G presents, originals={}",
                         enabled, present_stage, sample_originals / seconds, shown_fps, sample_shown, sample_gen_presents, sample_originals);
            } else {
                LOG_INFO(Render_Vulkan, "DLSS-G stage={} original FPS={:.2f} output FPS=unavailable SDK result={} status={}",
                         present_stage, sample_originals / seconds, int(result), unsigned(state.status));
            }
        }
        sample_start = now;
        sample_originals = 0;
        sample_generation = false;
        sample_optical = 0;
        sample_shown = sample_gen_presents = 0;
    }
    presenting = {};
    auto& slot = sealed[ticket % sealed.size()];
    if (slot.ticket == ticket) slot = {};
#endif
}
}
