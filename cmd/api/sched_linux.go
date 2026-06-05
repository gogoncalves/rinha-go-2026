//go:build linux

package main

import (
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// applyLowLatencyTuning applies kernel knobs that shave wake-up jitter on the
// Mac Mini Haswell judging hardware:
//   - PR_SET_TIMERSLACK to 1ns drops default 50µs scheduler timer slack, the
//     exact gap between us and the top of the leaderboard. Cheap, always on.
//   - SCHED_FIFO + prio 10 boosts the goroutine scheduler thread above normal
//     CFS tasks. ON A TIGHT 0.07-CPU CGROUP THIS STARVES THE GO RUNTIME and
//     creates request timeouts. Default off, opt-in via SCHED_FIFO=1 once the
//     contest box has CAP_SYS_NICE + a roomier CPU quota.
//
// Both are best-effort: errors are swallowed (we still want to serve traffic
// even if the kernel rejects the call).
func applyLowLatencyTuning() {
	_ = unix.Prctl(unix.PR_SET_TIMERSLACK, 1, 0, 0, 0)

	if os.Getenv("SCHED_FIFO") != "1" {
		return
	}
	// SCHED_FIFO = 1. sched_param is a single int32 priority.
	const schedFIFO = 1
	var prio struct{ Priority int32 }
	prio.Priority = 10
	_, _, _ = syscall.RawSyscall(
		syscall.SYS_SCHED_SETSCHEDULER,
		0, // pid=0 -> current thread
		uintptr(schedFIFO),
		uintptr(unsafe.Pointer(&prio)),
	)
}
