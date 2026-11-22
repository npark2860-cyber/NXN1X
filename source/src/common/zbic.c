#include "zbic.h"
#include <string.h>

// Keep the modified decoder separate from the standard zstd used for NUP containers.
// Upstream: kinnay/zbic at 11b08f2712264bbed731545085cbd9702096ceb7.
#define ZSTD_ZBIC_SUPPORT 1
#define ZSTDLIB_VISIBLE static
#define ZSTDLIB_HIDDEN static
#define ZSTDERRORLIB_VISIBLE static
#define ZSTDERRORLIB_HIDDEN static
#define g_ZSTD_threading_useless_symbol nxn1x_zbic_threading_symbol
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../../externals/zbic/src/zstd.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

bool nxn1x_decode_zbic(void* dst, size_t dst_size, const void* src, size_t src_size) {
    if (src_size < 4 || memcmp(src, "ZBIC", 4) != 0) {
        return false;
    }
    const size_t result = ZSTD_decompress(dst, dst_size, src, src_size);
    return !ZSTD_isError(result) && result == dst_size;
}
