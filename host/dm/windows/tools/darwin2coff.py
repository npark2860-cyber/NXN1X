#!/usr/bin/env python3
# darwin2coff.py <in.S> <out.S>: an AArch64 assembly file written for Mach-O (the host's entry.S, nvn_entry.S, the generated nvn_thunks.S) as COFF assembly
# for the Windows build (aarch64-w64-mingw32). Same rules as dm/android/tools/darwin2elf.py: C symbols lose Mach-O's leading underscore, sym@PAGE / sym@PAGEOFF
# become sym / :lo12:sym. COFF has no hidden visibility, so .private_extern lines go; Mach-O's section shorthands become COFF sections; .zerofill becomes a
# zero block in .bss (pushsection / popsection: .zerofill does not change the current section). Local labels (1:, 1f) and numbers are untouched.
# A symbol is "C" when it starts with one underscore followed by a letter.
import re, sys

src, dst = sys.argv[1], sys.argv[2]
SYM = r'_([A-Za-z][A-Za-z0-9_$.]*)'
SECTIONS = [(r'\.cstring\b', '.section .rdata,"dr"'), (r'\.const_data\b', '.section .rdata,"dr"'), (r'\.const\b', '.section .rdata,"dr"')]
out = []
for line in open(src, encoding='utf-8'):
    code, sep, comment = line.partition('//')
    m = re.match(r'^(\s*)\.zerofill\s+[^,]+,[^,]+,\s*' + SYM + r'\s*,\s*(\d+)\s*,\s*(\d+)\s*$', code.rstrip('\r\n'))
    if m:
        ind, name, size, align = m.groups()
        out.append(f'{ind}.pushsection .bss\n{ind}.p2align {align}\n{name}:\n{ind}.zero {size}\n{ind}.popsection' + (' ' + sep + comment if sep else '\n'))
        continue
    if re.match(r'^\s*\.private_extern\b', code): out.append(sep + comment if sep else '\n'); continue
    code = re.sub(SYM + r'@PAGEOFF', r':lo12:\1', code)
    code = re.sub(SYM + r'@PAGE', r'\1', code)
    for pat, rep in SECTIONS: code = re.sub(r'^(\s*)' + pat, lambda mm: mm.group(1) + rep, code)
    code = re.sub(r'(\.globl\s+|\b(?:bl|b|b\.\w+|cbz\s+\w+,|cbnz\s+\w+,|tbz\s+\w+,\s*#\d+,|tbnz\s+\w+,\s*#\d+,|adr\s+\w+,)\s*)' + SYM + r'\b', r'\1\2', code)
    code = re.sub(r'^(\s*)' + SYM + r':', r'\1\2:', code)   # label definitions
    code = re.sub(r'(\.(?:quad|word|xword|8byte)\s+)' + SYM + r'\b', r'\1\2', code)   # data references
    out.append(code + sep + comment)
text = ''.join(out)
if '@PAGE' in text: sys.exit(f'{src}: unconverted @PAGE operand')
if re.search(r'^\s*\.zerofill', text, re.M): sys.exit(f'{src}: unconverted .zerofill')
open(dst, 'w', encoding='utf-8', newline='\n').write(text)
