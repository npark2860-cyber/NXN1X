// Windows port: the blocks runtime calls (dm/windows/compat/blocks_runtime.c)
#pragma once
void *_Block_copy(const void *b);
void _Block_release(const void *b);
#define Block_copy(b) ((__typeof__(b))_Block_copy((const void *)(b)))
#define Block_release(b) _Block_release((const void *)(b))
