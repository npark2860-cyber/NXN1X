// Windows port: Apple's libcompression is not available; the capture writer (nvn_capture.c) stores its blocks uncompressed when encode returns 0
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef enum { COMPRESSION_LZ4 = 0x100, COMPRESSION_ZLIB = 0x205, COMPRESSION_LZMA = 0x306, COMPRESSION_LZ4_RAW = 0x101, COMPRESSION_LZFSE = 0x801 } compression_algorithm;
static inline size_t compression_encode_buffer(uint8_t *dst, size_t dn, const uint8_t *src, size_t sn, void *scratch, compression_algorithm a) { (void)dst; (void)dn; (void)src; (void)sn; (void)scratch; (void)a; return 0; }
static inline size_t compression_decode_buffer(uint8_t *dst, size_t dn, const uint8_t *src, size_t sn, void *scratch, compression_algorithm a) { (void)dst; (void)dn; (void)src; (void)sn; (void)scratch; (void)a; return 0; }
static inline size_t compression_encode_scratch_buffer_size(compression_algorithm a) { (void)a; return 0; }
static inline size_t compression_decode_scratch_buffer_size(compression_algorithm a) { (void)a; return 0; }
