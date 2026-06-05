//go:build linux

package main

import (
	"log"
	"os"
	"syscall"
	"unsafe"

	idx "rinha-go/internal/index"

	"golang.org/x/sys/unix"
)

const maxPrefix = 16 * 1024

func mlockAll() error {
	const MclCurrent = 1
	const MclFuture = 2
	return unix.Mlockall(MclCurrent | MclFuture)
}

func runCtrl(ix *idx.Index, sockPath string) {
	_ = os.Remove(sockPath)
	fd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_SEQPACKET|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		log.Fatalf("ctrl socket: %v", err)
	}
	sa := &unix.SockaddrUnix{Name: sockPath}
	if err := unix.Bind(fd, sa); err != nil {
		log.Fatalf("ctrl bind %s: %v", sockPath, err)
	}
	_ = os.Chmod(sockPath, 0o666)
	if err := unix.Listen(fd, 8); err != nil {
		log.Fatalf("ctrl listen: %v", err)
	}
	log.Printf("api waiting on ctrl %s", sockPath)
	for {
		cfd, _, aerr := unix.Accept(fd)
		if aerr != nil {
			log.Printf("ctrl accept: %v", aerr)
			continue
		}
		go ctrlLoop(ix, cfd)
	}
}

func ctrlLoop(ix *idx.Index, ctrlFD int) {
	defer unix.Close(ctrlFD)
	oob := make([]byte, unix.CmsgSpace(4))
	for {
		fdInt, prefix, err := recvFDWithBytes(ctrlFD, oob)
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			log.Printf("ctrl recvmsg: %v", err)
			return
		}
		if fdInt < 0 {
			continue
		}
		// Copy prefix because the recv buffer is reused.
		var prefixCopy []byte
		if len(prefix) > 0 {
			prefixCopy = make([]byte, len(prefix))
			copy(prefixCopy, prefix)
		}
		go handleConnFD(ix, fdInt, prefixCopy)
	}
}

var recvPrefixBuf = make([]byte, maxPrefix)

// recvFDWithBytes reads exactly one seqpacket message containing 2-byte
// little-endian length + payload, plus a single fd in cmsg. Returns fd, the
// prefix bytes (referencing a per-goroutine buffer; caller must copy if it
// must outlive the next call), and error.
func recvFDWithBytes(ctrlFD int, oob []byte) (int, []byte, error) {
	var lenBuf [2]byte
	iovs := [2]unix.Iovec{
		{Base: &lenBuf[0], Len: 2},
		{Base: &recvPrefixBuf[0], Len: uint64(len(recvPrefixBuf))},
	}
	var msg unix.Msghdr
	msg.Iov = &iovs[0]
	msg.SetIovlen(2)
	msg.Control = &oob[0]
	msg.SetControllen(len(oob))

	r, _, errno := unix.Syscall(unix.SYS_RECVMSG, uintptr(ctrlFD), uintptr(unsafe.Pointer(&msg)), 0)
	if errno != 0 {
		return -1, nil, errno
	}
	rn := int(r)
	if rn <= 0 {
		return -1, nil, unix.ECONNRESET
	}
	if rn < 2 {
		return -1, nil, unix.EBADMSG
	}
	plen := int(lenBuf[0]) | int(lenBuf[1])<<8
	if plen > len(recvPrefixBuf) || rn != 2+plen {
		return -1, nil, unix.EBADMSG
	}
	cmsgs, err := unix.ParseSocketControlMessage(oob[:msg.Controllen])
	if err != nil {
		return -1, nil, err
	}
	var fdInt int = -1
	for _, m := range cmsgs {
		fds, err := unix.ParseUnixRights(&m)
		if err != nil {
			continue
		}
		for _, fd := range fds {
			if fdInt < 0 {
				fdInt = fd
			} else {
				_ = unix.Close(fd)
			}
		}
	}
	if fdInt < 0 {
		return -1, nil, unix.EBADMSG
	}
	return fdInt, recvPrefixBuf[:plen], nil
}

func handleConnFD(ix *idx.Index, fdInt int, prefix []byte) {
	syscall.SetNonblock(fdInt, false)
	handleRawConn(ix, fdInt, prefix)
}
