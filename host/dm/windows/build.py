#!/usr/bin/env python3
# Windows ARM64 build of the host (the counterpart of dm/host/Makefile; modelled on the Android port's build.py): the same host and graphics sources, the Darwin
# compatibility layer (dm/windows/compat) and the Windows front end / null graphics backend / movie stubs (dm/windows/*.c), built with llvm-mingw's clang for aarch64.
#   build.py --game <title ID> [--llvm-mingw <dir>] [--out <dir>] [--jobs N] [--only-compile] [--keep-going] [--release]
# Output: <out>/dmhost.exe (default dm/windows/out/<title ID>): the guest runs on the main thread, a window (front_win.c) unless DM_HEADLESS=1; modules are read from <dir>/<name>.patched.bin (loader.c).
# Objects are rebuilt when a source or a header it includes changed (clang -MMD dependency files).
import argparse, concurrent.futures as cf, hashlib, os, re, shutil, subprocess, sys
from pathlib import Path

DM = Path(__file__).resolve().parent.parent          # dm/
ap = argparse.ArgumentParser()
ap.add_argument('--game', required=True)
ap.add_argument('--llvm-mingw', default=os.environ.get('LLVM_MINGW') or str(DM.parent.parent / 'tools/llvm-mingw'), help='llvm-mingw folder (its bin/ holds clang)')
ap.add_argument('--out')
ap.add_argument('--jobs', type=int, default=os.cpu_count() or 8)
ap.add_argument('--only-compile', action='store_true', help='compile every source and report errors, no link')
ap.add_argument('--keep-going', action='store_true')
ap.add_argument('-O', default='2')
ap.add_argument('--verbose', action='store_true', help='print warnings')
ap.add_argument('--release', action='store_true', help='no debug information, DM_RELEASE defined')
ap.add_argument('--sdl3', default=os.environ.get('DM_SDL3') or str(DM.parent.parent.parent / 'NXN1X/tools/cpm/sdl3/3.4.8-d57c3b685c-mingw-arm64'),
                help='SDL3 folder whose include/ front_win.c compiles against (r43: SDL3 controllers; the DLL is loaded at run time, nothing is linked)')
ap.add_argument('--gfx', choices=('vk', 'null'), default='vk', help='graphics backend: vk (Vulkan, gfx_vk_*.c, from the Android port) or null (draws nothing, stage 2)')
a = ap.parse_args()

exe = '.exe' if os.name == 'nt' else ''
bin_ = Path(a.llvm_mingw) / 'bin'
CC = str(bin_ / f'clang{exe}') if (bin_ / f'clang{exe}').exists() else shutil.which('clang') or 'clang'
TARGET = '--target=aarch64-w64-mingw32'
out = Path(a.out) if a.out else DM / 'windows/out' / (a.game + ('-release' if a.release else ''))
DBG = [] if a.release else ['-g', '-gcodeview']
obj = out / 'obj'; gen = out / 'gen'; obj.mkdir(parents=True, exist_ok=True); (gen / a.game).mkdir(parents=True, exist_ok=True)

# the game profile header, as the Makefile makes it
subprocess.run([sys.executable, str(DM / 'games/game.py'), 'header', a.game, str(gen / a.game / 'game_profile.h')], check=True)

def srcs(*globs): return [p for g in globs for p in sorted(DM.glob(g))]
HOST = [p for p in srcs('host/*.c') if p.name not in ('hle_movie_avc.c', 'hle_movie_av1.c', 'hle_jit.c')]   # VideoToolbox / dav1d: dm/windows/movie_stub_win.c instead; hle_jit.c: nn::jit (Super Mario 64 only) needs MAP_JIT / pthread_jit_write_protect_np
GFX = srcs('gfx/*.c', 'gfx/nvn/*.c', 'gfx/tex/*.c')
PORT = [p for p in srcs('windows/compat/*.c', 'windows/*.c') if not (p.name.startswith('gfx_vk') if a.gfx == 'null' else p.name == 'gfx_null_win.c')]
ASM = srcs('host/entry.S', 'gfx/nvn/nvn_entry.S', 'gfx/nvn/nvn_thunks.S')
VPX = srcs('third_party/libvpx/**/*.c')

INC = ['-include', str(DM / 'windows/compat/include/darwin_compat.h'), f'-I{DM / "windows/compat/include"}', f'-I{DM / "host"}', f'-I{DM / "gfx/nvn"}', f'-I{DM / "gfx/tex"}',
       f'-I{DM / "gfx/nvn/include"}', f'-I{DM / "third_party/vulkan-headers/include"}', f'-I{gen / a.game}', f'-I{DM / "third_party/libvpx"}', f'-I{DM / "third_party/stb_vorbis"}',
       f'-I{DM / "windows"}', f'-I{DM / "windows/vkgen"}']
# x18 is the TEB on Windows: the target's ABI keeps the compiler off it, as -ffixed-x18 does elsewhere
CFLAGS = [TARGET, f'-O{a.O}'] + DBG + ['-fblocks', '-fno-omit-frame-pointer', '-Wall', '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-but-set-variable',
          '-Wno-format', '-Wno-deprecated-declarations', '-Wno-gnu-folding-constant', '-Werror=implicit-function-declaration', '-Werror=return-type',
          '-Werror=incompatible-pointer-types', '-Werror=int-conversion', '-D_GNU_SOURCE', '-D_UCRT', '-D_FILE_OFFSET_BITS=64', '-DDM_WINDOWS=1',
          '-fno-strict-aliasing'] + (['-DDM_RELEASE=1'] if a.release else [])
SDL_INC = [f'-I{Path(a.sdl3) / "include"}'] if (Path(a.sdl3) / 'include/SDL3/SDL.h').exists() else []
print(f'SDL3 headers: {Path(a.sdl3) / "include"}' if SDL_INC else f'SDL3 headers not found under {a.sdl3}: front_win.c without SDL3 controllers (XInput only)')
VPX_FLAGS = [TARGET, '-O2'] + DBG + ['-D_UCRT', f'-I{DM / "third_party/libvpx"}', f'-I{DM / "third_party/libvpx/config"}', '-w']

EXTRA = os.environ.get('DM_WIN_CFLAGS', '').split()   # extra compiler flags (a cross build's -isystem for the mingw headers)
CFLAGS += EXTRA; VPX_FLAGS += EXTRA
_stamp = obj / 'flags.stamp'; _flags = hashlib.sha256(repr((CFLAGS, INC, VPX_FLAGS, CC)).encode()).hexdigest()
if not _stamp.exists() or _stamp.read_text() != _flags:   # compiler flags changed: every object is rebuilt
    shutil.rmtree(obj, ignore_errors=True); obj.mkdir(parents=True, exist_ok=True); _stamp.write_text(_flags)

def dep_stale(o, d, s):
    if not o.exists() or o.stat().st_mtime < s.stat().st_mtime: return True
    if not d.exists(): return True
    t = o.stat().st_mtime
    txt = d.read_text(errors='replace').replace('\\\n', ' ')
    for f in re.split(r'(?<!\\)\s+', txt.split(':', 1)[1] if ':' in txt else ''):
        f = f.replace('\\ ', ' ').strip()
        if f and Path(f).exists() and Path(f).stat().st_mtime > t: return True
    return False

def compile_one(s):
    rel = s.relative_to(DM); o = obj / (str(rel).replace(os.sep, '_').replace('/', '_') + '.o'); d = o.with_suffix('.d')
    src = s
    if s.suffix == '.S':   # Mach-O assembly → COFF
        src = gen / (s.stem + '.coff.S')
        if not src.exists() or src.stat().st_mtime < s.stat().st_mtime or src.stat().st_mtime < (DM / 'windows/tools/darwin2coff.py').stat().st_mtime:
            subprocess.run([sys.executable, str(DM / 'windows/tools/darwin2coff.py'), str(s), str(src)], check=True)
    if not dep_stale(o, d, src): return s, o, 0, ''
    flags = VPX_FLAGS if 'third_party/libvpx' in str(rel).replace(os.sep, '/') else [TARGET] + DBG if s.suffix == '.S' else CFLAGS + INC + (SDL_INC if s.name == 'front_win.c' else [])
    r = subprocess.run([CC] + flags + ['-MMD', '-MF', str(d), '-c', str(src), '-o', str(o)], capture_output=True, text=True, errors='replace')
    return s, o, r.returncode, r.stderr

def build(files):
    objs, failed = [], []
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for s, o, rc, err in ex.map(compile_one, files):
            objs.append(o)
            if rc: failed.append(s); print(f'--- {s.relative_to(DM)}\n{err.strip()}\n', file=sys.stderr)
            elif err.strip() and a.verbose: print(err, file=sys.stderr)
    return objs, failed

all_files = HOST + GFX + PORT + ASM
objs, failed = build(all_files)
vobjs, vfailed = build(VPX)
failed += vfailed
print(f'compiled {len(all_files) + len(VPX)} sources, {len(failed)} failed' + (': ' + ', '.join(str(f.relative_to(DM)) for f in failed) if failed else ''))
if failed and not a.keep_going: sys.exit(1)
if a.only_compile: sys.exit(1 if failed else 0)

# winpthread linked statically (no DLL beside the exe); the main thread's stack 8 MiB (front_run runs the guest on it)
LIBS = ['-lbcrypt', '-lwinmm', '-lpsapi', '-ladvapi32', '-luser32', '-lgdi32', '-lole32', '-lntdll', '-lkernel32']   # user32 / gdi32: front_win.c's window; ole32: the WASAPI output (compat/audiotoolbox_null.c)
LDFLAGS = [TARGET, '-fuse-ld=lld', '-static', '-pthread', '-Wl,--stack,8388608', '-Wl,--error-limit=0'] + ([] if a.release else ['-g', '-Wl,--pdb='])
target = out / 'dmhost.exe'
r = subprocess.run([CC] + LDFLAGS + ['-o', str(target)] + [str(o) for o in objs + vobjs] + LIBS, capture_output=True, text=True, errors='replace')
print(r.stderr.strip() if r.returncode else f'linked {target}', file=sys.stderr if r.returncode else sys.stdout)
if r.returncode: sys.exit(1)
if r.stderr.strip(): print(r.stderr.strip(), file=sys.stderr)   # linker warnings (e.g. locally defined symbol imported)
