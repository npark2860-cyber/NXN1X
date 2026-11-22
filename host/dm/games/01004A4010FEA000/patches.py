# Bayonetta 3 (01004A4010FEA000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of a module's NSO build ID;
# CUTS, OBS and FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
from nso_asm import ADRP, B
# r87 cutscene 60 (v1.2.0 main): the title, menus and cutscenes ask the main loop for frame-rate mode 1 (30) and then run two fixed logic steps of 1.0 per frame, of two kinds
# (the first recomputes the timeline, the second latches skips and branches). The stubs ask for mode 0 (60) instead, run one step per frame with the kinds taking turns, and
# scale the step by the measured frame length (clock +0xf0, ms) so the scene keeps pace with its sound. Mechanism found by Witch Time (Zexyen/Witch-Time, docs/STATUS.md) and
# written as B3-express's src/nx/hle/scene60.c; these stubs follow that design on the four call sites of the main loop (each function has only that direct caller).
# State page: +0x200 u32 on (loader.c, DM_CUTSCENE_FPS: unset or 60 = on, 30 = off), +0x204 half (the last frame ran the first kind alone), +0x208 one step this frame,
# +0x20c f32 owed (30 Hz frames behind the sound), +0x210 the last clock call was fixed, +0x214 frames forced to 60, +0x218 last mode asked + 1, +0x21c movie flag restores.
# A ride set piece (state flag 16) keeps steps of exactly 1.0 and runs both kinds in one frame when a whole step is owed; a load or pause (flags 0x4001) resets owed; a seek
# (flag 21) repeats the second kind. A movie scene whose header fps / base is 2 gets flag 28 back (main+0x7077d0 with 1) when only flag 30 is left of 0x50000000.
# Pointers: the state flags at *(main+0x249c388), the eight 112-byte scene slots at *(main+0x24aae68). Stubs use x9-x17, s0-s7 and s16-s20 only (the loop keeps s8 / s9)
def frame_step(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state
    # clock: the loop's clock call (main+0xc1f600: clock x0, step s0, measured w1; 0 = a fixed step)
    fs_site(0x670c24, '77ba1694'); start = here()
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0xb9420211)                                        # ldr w17, [x16, #0x200]
    emit(0x35000051)                                        # cbnz w17, c1
    emit(B(here(), 0xc1f600))                               # b 0xc1f600
    emit(0xa9bd7bfd)                                        # stp x29, x30, [sp, #-48]!
    emit(0x910003fd)                                        # mov x29, sp
    emit(0xf9000be0)                                        # str x0, [sp, #16]
    emit(0xb9001be1)                                        # str w1, [sp, #24]
    emit(ADRP(9, here(), 0x24aae68))                        # adrp x9, 0x24aae68
    emit(0xf9473529)                                        # ldr x9, [x9, #0xe68]
    emit(0xb4000749)                                        # cbz x9, c8
    emit(0x5280000a)                                        # mov w10, #0
    emit(0xaa0903eb)                                        # mov x11, x9
    emit(0xb940016c)                                        # ldr w12, [x11]
    emit(0x3600064c)                                        # tbz w12, #0, cn
    emit(0xf9401d6d)                                        # ldr x13, [x11, #56]
    emit(0xb400060d)                                        # cbz x13, cn
    emit(0xb940a9ac)                                        # ldr w12, [x13, #168]
    emit(0x340005cc)                                        # cbz w12, cn
    emit(0xf94005ae)                                        # ldr x14, [x13, #8]
    emit(0xb400058e)                                        # cbz x14, cn
    emit(0x39c241cf)                                        # ldrsb w15, [x14, #144]
    emit(0x39c245cc)                                        # ldrsb w12, [x14, #145]
    emit(0x3400052c)                                        # cbz w12, cn
    emit(0x1acc0def)                                        # sdiv w15, w15, w12
    emit(0x710009ff)                                        # cmp w15, #2
    emit(0x540004c1)                                        # b.ne cn
    emit(ADRP(14, here(), 0x249c388))                       # adrp x14, 0x249c388
    emit(0xf941c5ce)                                        # ldr x14, [x14, #0x388]
    emit(0x5280000c)                                        # mov w12, #0
    emit(0xb400004e)                                        # cbz x14, cf
    emit(0xb94001cc)                                        # ldr w12, [x14]
    emit(0x52aa000d)                                        # mov w13, #0x50000000
    emit(0x0a0d018c)                                        # and w12, w12, w13
    emit(0x52a8000d)                                        # mov w13, #0x40000000
    emit(0x6b0d019f)                                        # cmp w12, w13
    emit(0x54000381)                                        # b.ne cn
    emit(0xd10343ff)                                        # sub sp, sp, #208
    emit(0xa90007e0)                                        # stp x0, x1, [sp]
    emit(0xa9010fe2)                                        # stp x2, x3, [sp, #16]
    emit(0xa90217e4)                                        # stp x4, x5, [sp, #32]
    emit(0xa9031fe6)                                        # stp x6, x7, [sp, #48]
    emit(0xf90023e8)                                        # str x8, [sp, #64]
    emit(0xad0287e0)                                        # stp q0, q1, [sp, #80]
    emit(0xad038fe2)                                        # stp q2, q3, [sp, #112]
    emit(0xad0497e4)                                        # stp q4, q5, [sp, #144]
    emit(0xad059fe6)                                        # stp q6, q7, [sp, #176]
    emit(0x52800020)                                        # mov w0, #1
    emit(B(here(), 0x7077d0) | 0x80000000)                  # bl 0x7077d0
    emit(0xa94007e0)                                        # ldp x0, x1, [sp]
    emit(0xa9410fe2)                                        # ldp x2, x3, [sp, #16]
    emit(0xa94217e4)                                        # ldp x4, x5, [sp, #32]
    emit(0xa9431fe6)                                        # ldp x6, x7, [sp, #48]
    emit(0xf94023e8)                                        # ldr x8, [sp, #64]
    emit(0xad4287e0)                                        # ldp q0, q1, [sp, #80]
    emit(0xad438fe2)                                        # ldp q2, q3, [sp, #112]
    emit(0xad4497e4)                                        # ldp q4, q5, [sp, #144]
    emit(0xad459fe6)                                        # ldp q6, q7, [sp, #176]
    emit(0x910343ff)                                        # add sp, sp, #208
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0xb9421e11)                                        # ldr w17, [x16, #0x21c]
    emit(0x11000631)                                        # add w17, w17, #1
    emit(0xb9021e11)                                        # str w17, [x16, #0x21c]
    emit(0x14000005)                                        # b c8
    emit(0x9101c16b)                                        # add x11, x11, #112
    emit(0x1100054a)                                        # add w10, w10, #1
    emit(0x7100215f)                                        # cmp w10, #8
    emit(0x54fff94b)                                        # b.lt cl
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0x52800011)                                        # mov w17, #0
    emit(0x35000081)                                        # cbnz w1, c2
    emit(0x1e2e1010)                                        # fmov s16, #1.0
    emit(0x1e302000)                                        # fcmp s0, s16
    emit(0x1a9f17f1)                                        # cset w17, eq
    emit(0xb9020a11)                                        # str w17, [x16, #0x208]
    emit(B(here(), 0xc1f600) | 0x80000000)                  # bl 0xc1f600
    emit(0xb9401be1)                                        # ldr w1, [sp, #24]
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0x35000a41)                                        # cbnz w1, cx
    emit(0xf9400be0)                                        # ldr x0, [sp, #16]
    emit(ADRP(14, here(), 0x249c388))                       # adrp x14, 0x249c388
    emit(0xf941c5ce)                                        # ldr x14, [x14, #0x388]
    emit(0x5280000d)                                        # mov w13, #0
    emit(0xb400004e)                                        # cbz x14, c3
    emit(0xb94001cd)                                        # ldr w13, [x14]
    emit(0xb9420a0c)                                        # ldr w12, [x16, #0x208]
    emit(0x1e2e1001)                                        # fmov s1, #1.0
    emit(0x3400004c)                                        # cbz w12, c4
    emit(0x1e2c1001)                                        # fmov s1, #0.5
    emit(0xbd40e002)                                        # ldr s2, [x0, #0xe0]
    emit(0xbd40f003)                                        # ldr s3, [x0, #0xf0]
    emit(0x528aaaa9)                                        # mov w9, #0x5555
    emit(0x72a840a9)                                        # movk w9, #0x4205, lsl #16
    emit(0x1e270124)                                        # fmov s4, w9
    emit(0x1e241865)                                        # fdiv s5, s3, s4
    emit(0x1e202068)                                        # fcmp s3, #0.0
    emit(0x1e21cca5)                                        # fcsel s5, s5, s1, gt
    emit(0xbd420e06)                                        # ldr s6, [x16, #0x20c]
    emit(0xb9421209)                                        # ldr w9, [x16, #0x210]
    emit(0x34000089)                                        # cbz w9, c5
    emit(0x5288002a)                                        # mov w10, #0x4001
    emit(0x6a0a01bf)                                        # tst w13, w10
    emit(0x54000060)                                        # b.eq c6
    emit(0x1e2703e6)                                        # fmov s6, wzr
    emit(0x1e204025)                                        # fmov s5, s1
    emit(0x52a85689)                                        # mov w9, #0x42b40000
    emit(0x1e270130)                                        # fmov s16, w9
    emit(0x1e2c1013)                                        # fmov s19, #0.5
    emit(0x1e214032)                                        # fneg s18, s1
    emit(0x378002ed)                                        # tbnz w13, #16, cr
    emit(0x1e2528c7)                                        # fadd s7, s6, s5
    emit(0x1e2138e7)                                        # fsub s7, s7, s1
    emit(0x1e3020e0)                                        # fcmp s7, s16
    emit(0x1e27ce07)                                        # fcsel s7, s16, s7, gt
    emit(0x1e204031)                                        # fmov s17, s1
    emit(0x1e2120e0)                                        # fcmp s7, s1
    emit(0x540000ad)                                        # b.le c7
    emit(0x1e2138f4)                                        # fsub s20, s7, s1
    emit(0x1e330a94)                                        # fmul s20, s20, s19
    emit(0x1e342a31)                                        # fadd s17, s17, s20
    emit(0x14000006)                                        # b ck
    emit(0x1e3220e0)                                        # fcmp s7, s18
    emit(0x54000085)                                        # b.pl ck
    emit(0x1e2128f4)                                        # fadd s20, s7, s1
    emit(0x1e330a94)                                        # fmul s20, s20, s19
    emit(0x1e342a31)                                        # fadd s17, s17, s20
    emit(0x1e201014)                                        # fmov s20, #2.0
    emit(0x1e342220)                                        # fcmp s17, s20
    emit(0x1e31ce91)                                        # fcsel s17, s20, s17, gt
    emit(0x1e2128e6)                                        # fadd s6, s7, s1
    emit(0x1e3138c6)                                        # fsub s6, s6, s17
    emit(0x14000015)                                        # b ca
    emit(0x1e2528c7)                                        # fadd s7, s6, s5
    emit(0x1e2138e7)                                        # fsub s7, s7, s1
    emit(0x3400016c)                                        # cbz w12, r2
    emit(0xb9420609)                                        # ldr w9, [x16, #0x204]
    emit(0x35000129)                                        # cbnz w9, r2
    emit(0x1e2120e0)                                        # fcmp s7, s1
    emit(0x540000eb)                                        # b.lt r2
    emit(0xb9020a1f)                                        # str wzr, [x16, #0x208]
    emit(0x5280000c)                                        # mov w12, #0
    emit(0x1e212821)                                        # fadd s1, s1, s1
    emit(0x1e214032)                                        # fneg s18, s1
    emit(0x1e2528c7)                                        # fadd s7, s6, s5
    emit(0x1e2138e7)                                        # fsub s7, s7, s1
    emit(0x1e3220e0)                                        # fcmp s7, s18
    emit(0x54000065)                                        # b.pl r3
    emit(0x1e204246)                                        # fmov s6, s18
    emit(0x14000003)                                        # b r4
    emit(0x1e3020e0)                                        # fcmp s7, s16
    emit(0x1e27ce06)                                        # fcsel s6, s16, s7, gt
    emit(0x1e204031)                                        # fmov s17, s1
    emit(0xbd020e06)                                        # str s6, [x16, #0x20c]
    emit(0x3400004c)                                        # cbz w12, a1
    emit(0x1e312a31)                                        # fadd s17, s17, s17
    emit(0x1e310842)                                        # fmul s2, s2, s17
    emit(0xbd00e002)                                        # str s2, [x0, #0xe0]
    emit(0x52800029)                                        # mov w9, #1
    emit(0xb9021209)                                        # str w9, [x16, #0x210]
    emit(0x14000002)                                        # b co
    emit(0xb902121f)                                        # str wzr, [x16, #0x210]
    emit(0xa8c37bfd)                                        # ldp x29, x30, [sp], #48
    emit(0xd65f03c0)                                        # ret
    fs_put(0x670c24, [B(0x670c24, start) | 0x80000000], 'cutscene 60: clock stub (state +0x200 on)')
    # mode: the loop's frame-rate mode call (main+0xd02de0, w0: 0 = 60, 1 = 30, 2 = 15)
    fs_site(0x670c4c, '65481a94'); start = here()
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0x11000411)                                        # add w17, w0, #1
    emit(0xb9021a11)                                        # str w17, [x16, #0x218]
    emit(0xb9420211)                                        # ldr w17, [x16, #0x200]
    emit(0x340000f1)                                        # cbz w17, m9
    emit(0x7100041f)                                        # cmp w0, #1
    emit(0x540000a1)                                        # b.ne m9
    emit(0x52800000)                                        # mov w0, #0
    emit(0xb9421611)                                        # ldr w17, [x16, #0x214]
    emit(0x11000631)                                        # add w17, w17, #1
    emit(0xb9021611)                                        # str w17, [x16, #0x214]
    emit(B(here(), 0xd02de0))                               # b 0xd02de0
    fs_put(0x670c4c, [B(0x670c4c, start) | 0x80000000], 'cutscene 60: mode stub (state +0x200 on)')
    # kind: the loop's step-kind call (main+0x67037c, after flag 27 FIRST is set in the flags word at x19)
    fs_site(0x670d18, '99fdff97'); start = here()
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0xb9420211)                                        # ldr w17, [x16, #0x200]
    emit(0x34000251)                                        # cbz w17, k9
    emit(0xb9420a11)                                        # ldr w17, [x16, #0x208]
    emit(0xb942060f)                                        # ldr w15, [x16, #0x204]
    emit(0x340000b1)                                        # cbz w17, k2
    emit(0x350000cf)                                        # cbnz w15, k3
    emit(0x5280002f)                                        # mov w15, #1
    emit(0xb902060f)                                        # str w15, [x16, #0x204]
    emit(0x1400000b)                                        # b k9
    emit(0xb902061f)                                        # str wzr, [x16, #0x204]
    emit(0x14000009)                                        # b k9
    emit(0xb902061f)                                        # str wzr, [x16, #0x204]
    emit(ADRP(14, here(), 0x249c388))                       # adrp x14, 0x249c388
    emit(0xf941c5ce)                                        # ldr x14, [x14, #0x388]
    emit(0x5280000d)                                        # mov w13, #0
    emit(0xb400004e)                                        # cbz x14, k4
    emit(0xb94001cd)                                        # ldr w13, [x14]
    emit(0x120479ad)                                        # and w13, w13, #0xf7ffffff
    emit(0xb900026d)                                        # str w13, [x19]
    emit(B(here(), 0x67037c))                               # b 0x67037c
    fs_put(0x670d18, [B(0x670d18, start) | 0x80000000], 'cutscene 60: kind stub (state +0x200 on)')
    # end: the loop's first step-end call (main+0xb20a20; the PASS_TWO test on w21 bit 28 follows at main+0x670d98)
    fs_site(0x670d88, '26bf1294'); start = here()
    emit(0xa9bf7bfd)                                        # stp x29, x30, [sp, #-16]!
    emit(0x910003fd)                                        # mov x29, sp
    emit(B(here(), 0xb20a20) | 0x80000000)                  # bl 0xb20a20
    emit(0xa8c17bfd)                                        # ldp x29, x30, [sp], #16
    emit(ADRP(16, here(), FS_STATE_OFF))                    # adrp x16, state page
    emit(0xb9420211)                                        # ldr w17, [x16, #0x200]
    emit(0x34000171)                                        # cbz w17, e9
    emit(0xb9420a11)                                        # ldr w17, [x16, #0x208]
    emit(0x34000131)                                        # cbz w17, e9
    emit(ADRP(14, here(), 0x249c388))                       # adrp x14, 0x249c388
    emit(0xf941c5ce)                                        # ldr x14, [x14, #0x388]
    emit(0xb40000ae)                                        # cbz x14, e5
    emit(0xb94001cd)                                        # ldr w13, [x14]
    emit(0x36a8006d)                                        # tbz w13, #21, e5
    emit(0xb902061f)                                        # str wzr, [x16, #0x204]
    emit(0x14000002)                                        # b e9
    emit(0x9263fab5)                                        # and x21, x21, #0xffffffffefffffff
    emit(0xd65f03c0)                                        # ret
    fs_put(0x670d88, [B(0x670d88, start) | 0x80000000], 'cutscene 60: end stub (state +0x200 on)')

FRAME_STEP = {'b47cc79b88b7091ea514b2b725c50ae0': frame_step}
