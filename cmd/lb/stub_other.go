//go:build !linux

package main

import "log"

func main() { log.Fatalf("lb only builds on linux") }
