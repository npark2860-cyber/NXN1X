// SPDX-License-Identifier: GPL-2.0-or-later
// CPU only: diagnostic provenance and nested time accounting, no Vulkan loader.
#include "../nvn_foundation_diag.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void) {
    nvf0_submission s = {.id = 10, .frame = 7};
    nvf0_accepted(&s, 1, 1);
    CHECK(!nvf0_error(&s, 0, "end", 2, 1));
    CHECK(!nvf0_error(&s, 1, "not-ready", 2, 1));
    CHECK(!s.errors && !s.first_result);
    CHECK(nvf0_error(&s, -2, "part-main", 2, 3));
    CHECK(s.first_result == -2 && s.first_part == 2 && s.first_error_event == 3);
    CHECK(!strcmp(s.first_site, "part-main"));
    CHECK(!nvf0_error(&s, 0, "final-main", 3, 4));
    CHECK(nvf0_error(&s, -4, "dropped-upload", 3, 5));
    CHECK(s.errors == 2 && s.first_result == -2 && s.first_part == 2);
    CHECK(s.accepted_parts == 1 && s.accepted_last_part == 1);
    nvf0_submission pending = s; // completion metadata must survive slot reuse
    s = (nvf0_submission){.id = 11, .frame = 8};
    CHECK(!s.first_result && !s.accepted_parts && !s.first_site);
    CHECK(pending.first_result == -2 && pending.accepted_parts == 1);
    nvf0_accepted(&s, 4, 4);
    nvf0_accepted(&s, 5, 0);
    CHECK(s.accepted_parts == 4 && s.accepted_last_part == 4);
    nvf0_submission off = {0};
    CHECK(!nvf0_error(&off, -4, "off", 1, 1));
    nvf0_accepted(&off, 1, 1);
    CHECK(!off.errors && !off.accepted_parts);
    CHECK(!nvf0_error(NULL, -4, "null", 1, 1));

    // Restore -> allocation -> trim -> save -> transfer: five inclusive times
    // overlap, but the self times partition the root interval exactly.
    nvf0_scope scopes[5], *top = NULL;
    const unsigned kinds[] = {NVF0_RESTORE, NVF0_IMAGE_ALLOC, NVF0_TRIM, NVF0_SAVE, NVF0_TRANSFER};
    for (unsigned i = 0; i < 5; i++) {
        CHECK(nvf0_enter(&scopes[i], &top, 100 + i, kinds[i], i * 10, 42, 90, 512));
        CHECK(scopes[i].parent == (i ? 99 + i : 0));
        CHECK(scopes[i].submission == 42 && scopes[i].frame == 90);
    }
    uint64_t self_sum = 0;
    for (int i = 4; i >= 0; i--) {
        CHECK(nvf0_leave(&scopes[i], &top, 100 - i * 10));
        CHECK(scopes[i].elapsed == 100 - (uint64_t)i * 20);
        CHECK(scopes[i].self_ns == 20);
        self_sum += scopes[i].self_ns;
    }
    CHECK(!top && self_sum == scopes[0].elapsed);

    nvf0_scope depth[NVF0_MAX_DEPTH + 1];
    for (unsigned i = 0; i < NVF0_MAX_DEPTH; i++)
        CHECK(nvf0_enter(&depth[i], &top, 200 + i, NVF0_TRIM, i, 0, 0, 0));
    CHECK(!nvf0_enter(&depth[NVF0_MAX_DEPTH], &top, 300, NVF0_SAVE, 10, 0, 0, 0));
    CHECK(top == &depth[NVF0_MAX_DEPTH - 1] && !depth[NVF0_MAX_DEPTH].id);
    CHECK(!nvf0_leave(&depth[0], &top, 20)); // a mismatch cannot pop a live child
    CHECK(top == &depth[NVF0_MAX_DEPTH - 1]);
    for (int i = NVF0_MAX_DEPTH - 1; i >= 0; i--)
        CHECK(nvf0_leave(&depth[i], &top, 20));
    CHECK(!top);
    CHECK(!nvf0_enter(&scopes[0], &top, 0, NVF0_TRIM, 0, 0, 0, 0));
    CHECK(!nvf0_enter(&scopes[0], &top, 1, NVF0_KIND_COUNT, 0, 0, 0, 0));
    CHECK(!top && !scopes[0].id);

    CHECK(nvf0_enter(&scopes[0], &top, 1, NVF0_TRIM, 10, 0, 0, 0));
    CHECK(nvf0_leave(&scopes[0], &top, 5));
    CHECK(!scopes[0].elapsed && !scopes[0].self_ns); // clock reversal cannot wrap
    CHECK(nvf0_enter(&scopes[0], &top, 1, NVF0_TRIM, 0, 0, 0, 0));
    scopes[0].child_ns = UINT64_MAX - 1;
    CHECK(nvf0_enter(&scopes[1], &top, 2, NVF0_SAVE, 0, 0, 0, 0));
    CHECK(nvf0_leave(&scopes[1], &top, 5));
    CHECK(scopes[0].child_ns == UINT64_MAX);
    CHECK(nvf0_leave(&scopes[0], &top, 6));
    CHECK(!scopes[0].self_ns && !top);
    printf("PASS %u diagnostic checks; CPU only\n", checks);
    return 0;
}
