// Windows port: POSIX signals on top of mingw's <signal.h> (which has only signal() / raise() and SIGINT, SIGILL, SIGFPE, SIGSEGV, SIGTERM, SIGABRT).
// Delivery (dm/windows/compat/posix_win.c):
//   faults arrive through a vectored exception handler that maps the exception to SIGSEGV / SIGBUS / SIGILL / SIGTRAP / SIGFPE and calls the sigaction
//   handler with a Darwin-layout ucontext (sys/ucontext.h); register changes the handler makes are written back to the thread.
//   pthread_kill suspends the target, points it at a trampoline that runs the handler on its own stack, and resumes it.
//   alarm() runs the SIGALRM handler from a timer thread; SIGABRT / SIGTERM / SIGINT handlers also take the CRT's signal() path (abort(), Ctrl+C).
//   SA_ONSTACK handlers run on the thread's sigaltstack; the masks and the other SA_ flags are accepted and not used.
#pragma once
#include_next <signal.h>
#include <sys/types.h>
#include <stddef.h>
#define SIGHUP 1
#define SIGQUIT 3
#define SIGTRAP 5
#define SIGBUS 7
#define SIGKILL 9
#define SIGUSR1 10
#define SIGUSR2 12
#define SIGPIPE 13
#define SIGALRM 14
#define SIGCHLD 17
#define SIGCONT 18
#define SIGSTOP 19
#define SIGWINCH 28
#define DM_NSIG 32
#define SA_ONSTACK 0x0001
#define SA_RESTART 0x0002
#define SA_RESETHAND 0x0004
#define SA_NODEFER 0x0010
#define SA_SIGINFO 0x0040
#define SS_ONSTACK 1
#define SS_DISABLE 4
#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2
#define SEGV_MAPERR 1
#define SEGV_ACCERR 2
#define BUS_ADRALN 1
#define ILL_ILLOPC 1
#define TRAP_BRKPT 1

typedef struct { int si_signo, si_errno, si_code; int si_pid, si_uid, si_status; void *si_addr; long si_band; } siginfo_t;
struct sigaction {
    union { void (*sa_handler)(int); void (*sa_sigaction)(int, siginfo_t *, void *); };
    sigset_t sa_mask;
    int sa_flags;
};
typedef struct { void *ss_sp; size_t ss_size; int ss_flags; } stack_t;
int sigaction(int sig, const struct sigaction *sa, struct sigaction *old);
static inline int sigemptyset(sigset_t *s) { *s = 0; return 0; }
static inline int sigfillset(sigset_t *s) { *s = (sigset_t)~(sigset_t)0; return 0; }
static inline int sigaddset(sigset_t *s, int n) { *s |= (sigset_t)1 << n; return 0; }
static inline int sigdelset(sigset_t *s, int n) { *s &= ~((sigset_t)1 << n); return 0; }
static inline int sigismember(const sigset_t *s, int n) { return (int)((*s >> n) & 1); }
static inline int sigprocmask(int how, const sigset_t *s, sigset_t *old) { (void)how; (void)s; if (old) *old = 0; return 0; }
int sigaltstack(const stack_t *ss, stack_t *old);
int kill(pid_t pid, int sig);
#include <sys/ucontext.h>   // Darwin's <signal.h> brings ucontext_t along
// The Windows-only step of a fault handler (crash.c on_crash), after its own fault hooks declined: 1 = the fault was raised inside another DLL, and the handler
// should return so that the exception goes on to that DLL's handlers (__try / SEH); a fault no one handles comes back through the unhandled-exception filter,
// where this answers 0
int dm_win_pass_fault(void *uc);
