// Windows port: the host's registration tables are COFF grouped sections "<name>$m" (dm.h DM_SECTION), which the linker sorts between the
// "<name>$a" and "<name>$z" markers that dm/windows/compat/sections.c defines; getsectiondata looks the section name up among those
// (the segment name is ignored). Any other section: NULL.
#pragma once
#include <stdint.h>
struct mach_header_64 { uint32_t magic; };
extern const struct mach_header_64 _mh_execute_header;
uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *seg, const char *sect, unsigned long *size);
