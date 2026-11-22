// Windows port: the CoreFoundation the host uses: a mutable pointer-keyed dictionary with no callbacks (identity keys, values not retained), as an
// open-addressing hash table (dm/windows/compat/darwin_compat.c), and the app bundle lookups, which find no bundle (a run outside an app)
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
typedef struct dm_cfdict *CFMutableDictionaryRef, *CFDictionaryRef;
typedef const void *CFAllocatorRef, *CFTypeRef, *CFStringRef, *CFBundleRef;
typedef long CFIndex;
typedef unsigned long CFTypeID;
typedef uint32_t CFStringEncoding;
#define kCFStringEncodingUTF8 0x08000100
#define CFSTR(s) ((CFStringRef)(s))
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex cap, const void *kcb, const void *vcb);
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *key);
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *key, const void *value);
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *key);
CFIndex CFDictionaryGetCount(CFDictionaryRef d);
static inline CFBundleRef CFBundleGetMainBundle(void) { return NULL; }
static inline CFTypeRef CFBundleGetValueForInfoDictionaryKey(CFBundleRef b, CFStringRef k) { (void)b; (void)k; return NULL; }
static inline CFTypeID CFGetTypeID(CFTypeRef c) { (void)c; return 0; }
static inline CFTypeID CFStringGetTypeID(void) { return 7; }
static inline bool CFStringGetCString(CFStringRef s, char *buf, CFIndex cap, CFStringEncoding e) { (void)s; (void)buf; (void)cap; (void)e; return false; }
