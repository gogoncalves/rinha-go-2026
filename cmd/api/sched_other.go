//go:build !linux

package main

// Non-linux stub for the timerslack + SCHED_FIFO tuning. The judge runs Linux,
// but darwin/CI must still compile.
func applyLowLatencyTuning() {}
