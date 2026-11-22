#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(nxn1x_zbic_EXPORTS)
__declspec(dllexport)
#elif defined(_WIN32)
__declspec(dllimport)
#else
__attribute__((visibility("default")))
#endif
bool nxn1x_decode_zbic(void* dst, size_t dst_size, const void* src, size_t src_size);

#ifdef __cplusplus
}
#endif
