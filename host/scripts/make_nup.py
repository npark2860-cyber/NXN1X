#!/usr/bin/env python3
# A dump folder (exefs/ + romfs/, as a NUP extracts to: <version>/exefs, <version>/romfs) -> one .nup file (SPEC v1, the layout
# source/src/core/file_sys/nup_container.cpp and host/scripts/pack_nup.py read). One package: by default a standalone update
# (kind 1, flag 1: the update holds the whole game, as Pokemon Pokopia's v655360), or the base game with --base.
# Files are stored raw (codec 0) with a SHA-256 each; extents are 16-byte aligned. Meta block: the NPDM, the title name, the
# source dump label, and the display version when given. One read pass: the TOC's size is known before the data, so the data
# is written (and hashed) first and the TOC and the superblock last.
# Usage: make_nup.py <dump dir> <out.nup> --title-id 04007EF01F8AE000 --version 131072 --name "..." [--base] [--display-version 1.2.0]
import argparse, hashlib, os, struct, sys, time, uuid
from pathlib import Path

SUPER = struct.Struct('<4sHHQQ32sQQII16s32s')   # 0x80
TOC = struct.Struct('<4sIQQQQQ')   # 0x30
PKG = struct.Struct('<QIIIIBBHIQI20x')   # 0x40
DIR = struct.Struct('<IIIIQ')   # 0x18
FILE = struct.Struct('<IIQQQ32sBBH12x')   # 0x50
TLV = struct.Struct('<HHI')
NO_DIR = 0xFFFFFFFF
CHUNK = 8 << 20


def lp(p):   # paths past 260 characters on Windows
    p = os.path.abspath(p)
    return '\\\\?\\' + p.replace('/', '\\') if os.name == 'nt' and not p.startswith('\\\\?\\') else p


def align(v, a=16):
    return (v + a - 1) & ~(a - 1)


class Tree:
    def __init__(self):
        self.dirs = []    # [name, [child node ids]]
        self.files = []   # [name, path, size]
        self.pool = bytearray(); self.names = {}

    def name(self, s):
        b = s.encode('utf-8')
        if b not in self.names:
            self.names[b] = len(self.pool); self.pool += b
        return self.names[b], len(b)

    def add_dir(self, name, path):
        idx = len(self.dirs); self.dirs.append([name, []])
        if path is None: return idx
        with os.scandir(lp(path)) as it:
            entries = sorted(it, key=lambda e: e.name)
        kids = []
        for e in entries:
            if e.is_dir(follow_symlinks=False): kids.append(('d', self.add_dir(e.name, os.path.join(path, e.name))))
            elif e.is_file(follow_symlinks=False):
                self.files.append([e.name, os.path.join(path, e.name), e.stat().st_size]); kids.append(('f', len(self.files) - 1))
        self.dirs[idx][1] = kids
        return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--title-id', required=True); ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--name', required=True); ap.add_argument('--base', action='store_true')
    ap.add_argument('--display-version', default='')
    ap.add_argument('--resume', action='store_true', help='keep the files an earlier run already wrote to <out>.part')
    a = ap.parse_args()
    tid = int(a.title_id, 16)
    exefs, romfs = a.dump / 'exefs', a.dump / 'romfs'
    if not (exefs / 'main').is_file() or not (exefs / 'main.npdm').is_file(): sys.exit(f'{exefs}: no main / main.npdm')

    t0 = time.monotonic()
    tree = Tree()
    root = tree.add_dir(a.dump.name if not a.base else 'base', None)
    ex = tree.add_dir('exefs', str(exefs))
    ro = tree.add_dir('romfs', str(romfs)) if romfs.is_dir() else NO_DIR
    tree.dirs[root][1] = [('d', ex)] + ([('d', ro)] if ro != NO_DIR else [])
    nd, nf = len(tree.dirs), len(tree.files)
    total = sum(f[2] for f in tree.files)
    print(f'tree: {nd} dirs, {nf} files, {total} bytes ({time.monotonic() - t0:.1f} s)', flush=True)

    # names into the pool, child lists
    dir_recs, children = [], []
    for name, kids in tree.dirs:
        no, nl = tree.name(name)
        dir_recs.append((no, nl, len(children), len(kids)))
        children += [k if kind == 'd' else nd + k for kind, k in kids]
    file_names = [tree.name(f[0]) for f in tree.files]

    meta = bytearray()
    def tlv(ty, b): meta.extend(TLV.pack(ty, 0, len(b)) + b)
    tlv(1, (exefs / 'main.npdm').read_bytes())
    if a.display_version: tlv(4, a.display_version.encode())
    tlv(5, a.name.encode()); tlv(8, a.dump.name.encode())

    toc_size = TOC.size + PKG.size + nd * DIR.size + nf * FILE.size + 4 * len(children) + len(tree.pool)
    pool_off = TOC.size + PKG.size + nd * DIR.size + nf * FILE.size + 4 * len(children)
    blob = align(SUPER.size + toc_size)
    meta_off = blob
    pos = align(meta_off + len(meta))

    # the layout is fixed by the sorted tree, so every file's offset is known before any data moves
    offs = []
    for _, _, size in tree.files:
        offs.append(pos if size else 0)
        if size: pos = align(pos + size)
    file_size = pos

    tmp = a.out.with_suffix(a.out.suffix + '.part')
    # resume: a .part from an earlier run of the same tree holds every file whose extent ends at or below its size (data goes
    # out in file order); those are hashed from the .part, not read again
    have = os.path.getsize(lp(tmp)) if a.resume and os.path.exists(lp(tmp)) else 0
    recs = [None] * nf
    done = 0; last = time.monotonic()

    def read_src(i):   # small files dominate (Sound/stream: 215k): opened on worker threads so open/scan latency overlaps
        name, path, size = tree.files[i]
        with open(lp(path), 'rb') as f: b = f.read()
        if len(b) != size: raise SystemExit(f'{path}: size changed while reading ({size} -> {len(b)})')
        return b

    with open(lp(tmp), 'r+b' if have else 'wb') as out:
        out.seek(meta_off); out.write(meta)
        first_new = nf
        for i, (name, path, size) in enumerate(tree.files):
            if size and offs[i] + size > have: first_new = i; break
            if size:
                out.seek(offs[i]); h = hashlib.sha256(); left = size
                while left:
                    b = out.read(min(CHUNK, left)); h.update(b); left -= len(b)
                recs[i] = (offs[i], size, h.digest())
            else:
                recs[i] = (0, 0, hashlib.sha256().digest())
            done += size
        if have: print(f'resume: {first_new} files ({done >> 20} MiB) taken from the .part ({time.monotonic() - t0:.0f} s)', flush=True)
        from concurrent.futures import ThreadPoolExecutor
        todo = list(range(first_new, nf))
        with ThreadPoolExecutor(max_workers=32) as pool:
            window = 256; futs = {}
            for k, i in enumerate(todo):
                if k == 0:
                    for j in todo[:window]:
                        if tree.files[j][2] <= (8 << 20): futs[j] = pool.submit(read_src, j)
                nxt = k + window
                if nxt < len(todo) and tree.files[todo[nxt]][2] <= (8 << 20): futs[todo[nxt]] = pool.submit(read_src, todo[nxt])
                name, path, size = tree.files[i]
                if not size:
                    recs[i] = (0, 0, hashlib.sha256().digest()); continue
                out.seek(offs[i]); h = hashlib.sha256()
                if i in futs:
                    b = futs.pop(i).result(); h.update(b); out.write(b)
                else:   # a large file: streamed
                    n = 0
                    with open(lp(path), 'rb') as f:
                        while True:
                            b = f.read(CHUNK)
                            if not b: break
                            h.update(b); out.write(b); n += len(b)
                    if n != size: sys.exit(f'{path}: size changed while reading ({size} -> {n})')
                recs[i] = (offs[i], size, h.digest())
                done += size
                if time.monotonic() - last > 10:
                    last = time.monotonic()
                    print(f'  {i + 1}/{nf} files, {done * 100 // max(total, 1)}% ({done >> 20} MiB), {last - t0:.0f} s', flush=True)
        out.truncate(file_size)

        toc = bytearray(TOC.pack(b'TOC\x01', 1, nd, nf, pool_off, len(tree.pool), 0))
        kind, flags = (0, 0) if a.base else (1, 1)
        toc += PKG.pack(tid, root, ex, ro, a.version, kind, flags, 0, 0, meta_off, len(meta))
        for no, nl, fc, cc in dir_recs: toc += DIR.pack(no, nl, fc, cc, 0)
        for (no, nl), (off, size, sha) in zip(file_names, recs): toc += FILE.pack(no, nl, size, off, size, sha, 0, 0, 0)
        toc += struct.pack(f'<{len(children)}I', *children)
        toc += tree.pool
        assert len(toc) == toc_size
        sup = SUPER.pack(b'NUP\x01', SUPER.size, 1, SUPER.size, toc_size, hashlib.sha256(toc).digest(), blob, file_size, 1, 0, uuid.uuid4().bytes, b'\0' * 32)
        out.seek(0); out.write(sup); out.write(toc)
    os.replace(lp(tmp), lp(a.out))
    print(f'wrote {a.out}: {file_size} bytes, {nf} files, {time.monotonic() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
