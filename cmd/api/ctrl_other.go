//go:build !linux

package main

import (
	"errors"
	"log"

	idx "rinha-go/internal/index"
)

func mlockAll() error { return nil }

func runCtrl(ix *idx.Index, sockPath string) {
	_ = ix
	_ = sockPath
	log.Fatalf("CTRL_SOCK_PATH mode requires Linux")
}

var _ = errors.New
