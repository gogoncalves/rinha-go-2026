//go:build linux

package index

import "golang.org/x/sys/unix"

// mapPopulate exposes MAP_POPULATE so the platform-agnostic Open() can pass
// it into mmap. On linux it is 0x8000 on every arch we ship (amd64).
const mapPopulate = unix.MAP_POPULATE
