// Windows port: the blocks runtime the host's few blocks need (-fblocks; libdispatch's dispatch_async etc. in dispatch_win.c). The layouts are clang's
// (the ABI of compiler-rt's BlocksRuntime): a block copied to the heap keeps a reference count in its flags; __block variables are moved to the heap on
// the first copy of a block that captures them.
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void *_NSConcreteStackBlock[32], *_NSConcreteGlobalBlock[32], *_NSConcreteMallocBlock[32], *_NSConcreteAutoBlock[32], *_NSConcreteFinalizingBlock[32], *_NSConcreteWeakBlockVariable[32];
// clang's blocks code generation on COFF refers to the two block classes as DLL imports (__imp_ pointers): give it those pointers here
void *__imp__NSConcreteStackBlock = _NSConcreteStackBlock, *__imp__NSConcreteGlobalBlock = _NSConcreteGlobalBlock;
enum { BLOCK_REFCOUNT_MASK = 0xfffe, BLOCK_NEEDS_FREE = 1 << 24, BLOCK_HAS_COPY_DISPOSE = 1 << 25, BLOCK_IS_GLOBAL = 1 << 28 };
enum { BLOCK_FIELD_IS_OBJECT = 3, BLOCK_FIELD_IS_BLOCK = 7, BLOCK_FIELD_IS_BYREF = 8, BLOCK_FIELD_IS_WEAK = 16, BLOCK_BYREF_CALLER = 128 };
enum { BLOCK_BYREF_HAS_COPY_DISPOSE = 1 << 25 };
struct Block_descriptor { unsigned long reserved, size; void (*copy)(void *dst, const void *src); void (*dispose)(const void *); };
struct Block_layout { void *isa; _Atomic int flags; int reserved; void (*invoke)(void *, ...); struct Block_descriptor *descriptor; };
struct Block_byref { void *isa; struct Block_byref *forwarding; _Atomic int flags; unsigned int size; void (*keep)(struct Block_byref *dst, struct Block_byref *src); void (*destroy)(struct Block_byref *); };

static void retain(_Atomic int *flags) { atomic_fetch_add(flags, 2); }
static int release(_Atomic int *flags) { return ((atomic_fetch_sub(flags, 2) - 2) & BLOCK_REFCOUNT_MASK) == 0; }   // 1: the last reference went

void *_Block_copy(const void *arg) {
    struct Block_layout *b = (struct Block_layout *)arg;
    if (!b) return NULL;
    if (atomic_load(&b->flags) & BLOCK_NEEDS_FREE) { retain(&b->flags); return b; }
    if (atomic_load(&b->flags) & BLOCK_IS_GLOBAL) return b;
    struct Block_layout *c = malloc(b->descriptor->size); if (!c) return NULL;
    memcpy(c, b, b->descriptor->size);
    atomic_store(&c->flags, (atomic_load(&b->flags) & ~BLOCK_REFCOUNT_MASK) | BLOCK_NEEDS_FREE | 2);
    c->isa = _NSConcreteMallocBlock;
    if (atomic_load(&b->flags) & BLOCK_HAS_COPY_DISPOSE) b->descriptor->copy(c, b);
    return c;
}
void _Block_release(const void *arg) {
    struct Block_layout *b = (struct Block_layout *)arg;
    if (!b || !(atomic_load(&b->flags) & BLOCK_NEEDS_FREE)) return;
    if (!release(&b->flags)) return;
    if (atomic_load(&b->flags) & BLOCK_HAS_COPY_DISPOSE) b->descriptor->dispose(b);
    free(b);
}
static void byref_assign(void **dest, struct Block_byref *src) {
    struct Block_byref *f = src->forwarding;
    if (atomic_load(&f->flags) & BLOCK_NEEDS_FREE) retain(&f->flags);
    else if ((atomic_load(&f->flags) & BLOCK_REFCOUNT_MASK) == 0) {   // first copy: move the variable to the heap, both forwarding pointers to the copy
        struct Block_byref *c = malloc(src->size); if (!c) return;
        memcpy(c, src, sizeof *src);
        atomic_store(&c->flags, (atomic_load(&src->flags) & ~BLOCK_REFCOUNT_MASK) | BLOCK_NEEDS_FREE | 4);   // one for the stack variable, one for the caller
        c->forwarding = c; src->forwarding = c;
        if (atomic_load(&src->flags) & BLOCK_BYREF_HAS_COPY_DISPOSE) { c->keep = src->keep; c->destroy = src->destroy; src->keep(c, src); }
        else memcpy((uint8_t *)c + sizeof(void *) * 2 + sizeof(int) * 2, (uint8_t *)src + sizeof(void *) * 2 + sizeof(int) * 2, src->size - (sizeof(void *) * 2 + sizeof(int) * 2));
        f = c;
    }
    *dest = f;
}
void _Block_object_assign(void *dest, const void *obj, int flags) {
    if ((flags & BLOCK_FIELD_IS_BYREF) == BLOCK_FIELD_IS_BYREF) byref_assign(dest, (struct Block_byref *)obj);
    else if ((flags & BLOCK_FIELD_IS_BLOCK) == BLOCK_FIELD_IS_BLOCK) *(void **)dest = _Block_copy(obj);
    else *(const void **)dest = obj;
}
void _Block_object_dispose(const void *obj, int flags) {
    if ((flags & BLOCK_FIELD_IS_BYREF) == BLOCK_FIELD_IS_BYREF) {
        struct Block_byref *b = ((struct Block_byref *)obj)->forwarding;
        if (!(atomic_load(&b->flags) & BLOCK_NEEDS_FREE) || !release(&b->flags)) return;
        if (atomic_load(&b->flags) & BLOCK_BYREF_HAS_COPY_DISPOSE) b->destroy(b);
        free(b);
    } else if ((flags & BLOCK_FIELD_IS_BLOCK) == BLOCK_FIELD_IS_BLOCK) _Block_release(obj);
}
