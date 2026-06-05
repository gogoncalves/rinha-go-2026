//go:build !linux

package index

// MAP_POPULATE is a Linux-only flag. On other platforms (darwin/dev hosts) we
// silently fall back to a regular MAP_PRIVATE mapping.
const mapPopulate = 0
