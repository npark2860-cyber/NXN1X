#include "../nvn_image_eligibility.h"
#include <stdio.h>
#include <stdlib.h>
static unsigned checks;
#define CHECK(x) do { checks++; if(!(x)) {fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); exit(1);} } while(0)
int main(void) {
    nvni_state s={.image=10,.archive_image=10,.version=3,.archive_version=3,
        .last_use=5,.cpu_done=5,.supported_layout=1,.archive_complete=1};
    CHECK(nvni_eligibility(s).source==NVNI_ARCHIVE);
    s.legacy_guest=1; CHECK(nvni_eligibility(s).source==NVNI_LEGACY_GUEST); s.legacy_guest=0;
    s.version++; CHECK(nvni_eligibility(s).source==NVNI_SAVE_REQUIRED); // partial write invalidates whole snapshot
    s.version--; s.image++; CHECK(nvni_eligibility(s).blocked==NVNI_IDENTITY); // same pointer, new incarnation
    s.image--; s.last_use++; CHECK(nvni_eligibility(s).blocked==NVNI_PIN); // GPU done alone is insufficient
    s.last_use--; s.shared_keys=1; CHECK(nvni_eligibility(s).blocked==NVNI_PIN);
    s.shared_keys=0; s.external_reader=1; CHECK(nvni_eligibility(s).blocked==NVNI_PIN);
    s.external_reader=0; s.transfer_pending=1; CHECK(nvni_eligibility(s).blocked==NVNI_PIN);
    s.transfer_pending=0; s.archive_complete=0; CHECK(nvni_eligibility(s).source==NVNI_SAVE_REQUIRED);
    s.supported_layout=0; CHECK(nvni_eligibility(s).source==NVNI_KEEP); // sparse/MSAA/unsupported
    s.legacy_guest=1; CHECK(nvni_eligibility(s).source==NVNI_LEGACY_GUEST);
    s.legacy_guest=0; s.archive_complete=1; s.version=s.archive_version=UINT64_MAX;
    CHECK(nvni_eligibility(s).source==NVNI_KEEP); // exhausted version never becomes clean
    s.image=0; CHECK(nvni_eligibility(s).blocked==NVNI_IDENTITY);
    printf("PASS eligibility %u checks; no guest writeback, GPU NOT_RUN\n",checks);
}
