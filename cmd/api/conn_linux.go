//go:build linux

package main

import (
	"golang.org/x/sys/unix"

	idx "rinha-go/internal/index"
	"rinha-go/internal/jsonp"
	"rinha-go/internal/server"
)

// handleRawConn services a single TCP fd that was handed to us by the LB
// over SCM_RIGHTS together with the first bytes the LB had already read.
// We avoid net.FileConn / os.NewFile to skip the runtime poller registration
// hot path; raw unix.Read/unix.Write is fine for the per-request workload.
func handleRawConn(ix *idx.Index, fd int, prefix []byte) {
	defer unix.Close(fd)

	bp := bufPool.Get().(*[]byte)
	defer bufPool.Put(bp)
	buf := *bp
	payload := payloadPool.Get().(*jsonp.Payload)
	defer payloadPool.Put(payload)

	pos := 0
	if len(prefix) > 0 {
		if len(prefix) > len(buf) {
			return
		}
		copy(buf, prefix)
		pos = len(prefix)
	}
	for {
		// Try to parse any complete requests already in buf.
		for pos > 0 {
			req, perr := server.ParseRequest(buf[:pos])
			if perr != nil {
				if perr == server.ErrIncomplete {
					break
				}
				return
			}
			resp := server.Respond(ix, req, payload)
			if err := writeAll(fd, resp); err != nil {
				return
			}
			if req.End >= pos {
				pos = 0
			} else {
				copy(buf, buf[req.End:pos])
				pos -= req.End
			}
		}
		if pos == len(buf) {
			return
		}
		n, err := unix.Read(fd, buf[pos:])
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			return
		}
		if n == 0 {
			return
		}
		pos += n
	}
}

func writeAll(fd int, p []byte) error {
	for len(p) > 0 {
		n, err := unix.Write(fd, p)
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			return err
		}
		p = p[n:]
	}
	return nil
}
