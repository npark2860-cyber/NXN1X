// Windows port: getsectiondata over the host's registration tables. The tables are COFF grouped sections "<name>$m" (dm.h DM_SECTION); the linker
// merges every "<name>$..." contribution into one "<name>" section sorted by the text after '$', so the 8-byte markers in "$a" and "$z" defined here
// bound the entries. Each entry is a struct of pointers (8-byte aligned, a multiple of 8 in size), so no padding comes between them.
#include <mach-o/getsect.h>
#include <string.h>
const struct mach_header_64 _mh_execute_header = {0xfeedfacf};
#define BOUNDS(n) \
    __attribute__((used, section(#n "$a"), aligned(8))) static const uint64_t n##_start = 0; \
    __attribute__((used, section(#n "$z"), aligned(8))) static const uint64_t n##_end = 0;
BOUNDS(swhle) BOUNDS(swforce) BOUNDS(swnvn)
uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *seg, const char *sect, unsigned long *size) {
    (void)mh; (void)seg;
    const uint64_t *a = NULL, *b = NULL;
    if (!strcmp(sect, "swhle")) { a = &swhle_start; b = &swhle_end; }
    else if (!strcmp(sect, "swforce")) { a = &swforce_start; b = &swforce_end; }
    else if (!strcmp(sect, "swnvn")) { a = &swnvn_start; b = &swnvn_end; }
    if (!a) { if (size) *size = 0; return NULL; }
    const uint8_t *lo = (const uint8_t *)(a + 1), *hi = (const uint8_t *)b;
    if (size) *size = (unsigned long)(hi - lo);
    return (uint8_t *)lo;
}
