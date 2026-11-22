"""Configure/build NXN1X from the local source, NVN host and CPM package cache.
  python tools\\build_nx2.py configure | build | all [--jobs 1..4] [--targets yuzu ...] [--keep-going]
--keep-going: ninja -k 0, every compile error of the build in one log (the first build of new code)"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
BUILD = ROOT / "build/release"
EVIDENCE = ROOT / "docs/build-logs/NX2"
TOOLS = Path("C:/msys64/clangarm64/bin")
CPM = ROOT / "tools/cpm"
QT_TOOL_DIR = ROOT / "tools/qt-utf8"

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def run(label, command):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = EVIDENCE / (stamp + "-" + label)
    folder.mkdir()
    env = os.environ.copy()
    env["PATH"] = str(TOOLS) + os.pathsep + str(CPM / "qt6/6.11.1-mingw-arm64/bin") + os.pathsep + env.get("PATH", "")
    env.pop("MSYS2_LOCATION", None); env.pop("CC", None); env.pop("CXX", None)
    command = list(map(str, command))
    write(folder / "command.json", {"cwd": str(ROOT), "argv": command, "CPM_SOURCE_CACHE": str(CPM)})
    print(label.upper(), "log:", folder / "output.log", flush=True)
    with (folder / "output.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        write(folder / "running.json", {"pid": child.pid, "started_utc": datetime.now(timezone.utc).isoformat()})
        code = child.wait()
    write(folder / "result.json", {"exit_code": code, "finished_utc": datetime.now(timezone.utc).isoformat()})
    print((folder / "output.log").read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
    if code:
        raise SystemExit(label + " failed: " + str(folder))
    return folder

def configure():
    ffmpeg = ROOT / "build/ffmpeg-dav1d-arm64"
    recipe = ROOT / "tools/build_ffmpeg_av1.sh"
    manifest = ffmpeg / ".av1-installed"
    if not manifest.exists() or manifest.stat().st_mtime < recipe.stat().st_mtime:
        run("ffmpeg-dav1d", [TOOLS.parents[1] / "usr/bin/bash.exe", recipe.relative_to(ROOT).as_posix()])
    args = [TOOLS / "cmake.exe", "-S", SOURCE, "-B", BUILD, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=C:/msys64/clangarm64/bin/clang.exe",
        "-DCMAKE_CXX_COMPILER=C:/msys64/clangarm64/bin/clang++.exe", "-DCMAKE_MAKE_PROGRAM=C:/msys64/clangarm64/bin/ninja.exe",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCPM_SOURCE_CACHE=" + str(CPM), "-DCPM_LOCAL_PACKAGES_ONLY=ON",
        "-DNXN1X_FFMPEG_AV1_DIR=" + ffmpeg.as_posix(),
        "-DYUZU_USE_BUNDLED_QT=ON", "-DYUZU_USE_BUNDLED_SDL3=ON", "-DVulkanHeaders_FORCE_BUNDLED=ON",
        "-DVulkanUtilityLibraries_FORCE_BUNDLED=ON", "-DENABLE_LTO=OFF", "-DUSE_CCACHE=OFF", "-DUSE_DISCORD_PRESENCE=OFF",
        "-DENABLE_UPDATE_CHECKER=OFF", "-DENABLE_WEB_SERVICE=ON", "-DENABLE_QT_TRANSLATION=OFF", "-DYUZU_TESTS=OFF",
        "-DYUZU_CMD=OFF", "-DYUZU_ROOM=ON", "-DYUZU_ROOM_STANDALONE=OFF", "-DYUZU_CRASH_DUMPS=OFF",
        "-DUSE_FASTER_LINKER=OFF", "-DENABLE_WIFI_SCAN=OFF", "-DNXN1X_DIAGNOSTICS=OFF",
        "-DCMAKE_C_FLAGS=", "-DCMAKE_CXX_FLAGS=", "-DNXN1X_QT_TOOL_DIR=" + str(QT_TOOL_DIR)]
    return run("configure", args)

def build(jobs, targets, keep_going=False):
    if not 1 <= jobs <= 4:
        raise ValueError("Build concurrency is limited to 1..4 jobs")
    return run("build", [TOOLS / "cmake.exe", "--build", BUILD, "--parallel", jobs, "--target", *targets]
               + (["--", "-k", "0"] if keep_going else []))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("configure", "build", "all"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--targets", nargs="+", default=["yuzu"])
    p.add_argument("--keep-going", action="store_true")
    a = p.parse_args()
    if a.mode in ("configure", "all"): configure()
    if a.mode in ("build", "all"): build(a.jobs, a.targets, a.keep_going)
