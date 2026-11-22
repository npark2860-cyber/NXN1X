// s46pc: nvnTextureFinalize's plan with replay work queued (s46_fin.h)
#include <stdio.h>
#include <stdlib.h>
#include "../nvn/s46_fin.h"

typedef struct { int kind, pad; const void *t; uint64_t other; } job;   // the fields the plan reads, at other offsets than hle_nvn.c's s46_job
static int fails;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, m); fails++; } } while (0)
static int plan(int defer, uint64_t head, uint64_t tail, const job *q, unsigned cap, const void *tex) {
    return s46_fin_plan(defer, head, tail, q, cap, sizeof *q, offsetof(job, kind), offsetof(job, t), tex);
}
int main(void) {
    enum { CAP = 4 };
    job q[CAP] = {{0}}; int a = 0, b = 0;
    CHECK(plan(1, 0, 0, q, CAP, &a) == S46F_NOW, "empty queue: at once");
    CHECK(plan(0, 7, 7, q, CAP, &a) == S46F_NOW, "empty queue, deferral off: at once");
    q[0] = (job){0, 0, NULL, 0}; q[1] = (job){0, 0, NULL, 0};
    CHECK(plan(0, 2, 0, q, CAP, &a) == S46F_DRAIN, "deferral off: wait for the queue as before");
    CHECK(plan(1, 2, 0, q, CAP, &a) == S46F_QUEUE, "submits queued: finalization queued behind them");
    q[1] = (job){1, 0, &b, 0};
    CHECK(plan(1, 2, 0, q, CAP, &a) == S46F_QUEUE, "a present of another texture: queued");
    q[1] = (job){1, 0, &a, 0};
    CHECK(plan(1, 2, 0, q, CAP, &a) == S46F_DRAIN, "a queued present of this texture: wait");
    q[1] = (job){2, 0, &a, 0};
    CHECK(plan(1, 2, 0, q, CAP, &a) == S46F_QUEUE, "a finalize job with the pointer is not a present");
    // wrap-around: tail 6 -> slots 2, 3, 0 (head 9); the present in slot 0 is inside, slot 1 is not
    q[2] = (job){0, 0, NULL, 0}; q[3] = (job){0, 0, NULL, 0}; q[0] = (job){1, 0, &a, 0}; q[1] = (job){1, 0, &b, 0};
    CHECK(plan(1, 9, 6, q, CAP, &a) == S46F_DRAIN, "wrapped queue: the present in slot 0 is seen");
    CHECK(plan(1, 9, 6, q, CAP, &b) == S46F_QUEUE, "wrapped queue: slot 1 is outside [tail, head)");
    CHECK(plan(1, 8, 6, q, CAP, &a) == S46F_QUEUE, "slots 2, 3 only");
    if (fails) { printf("t_s46_fin: %d failed\n", fails); return 1; }
    printf("t_s46_fin: all passed\n"); return 0;
}
