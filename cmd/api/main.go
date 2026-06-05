package main

import (
	"errors"
	"io"
	"log"
	"math/rand"
	"net"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	idx "rinha-go/internal/index"
	"rinha-go/internal/jsonp"
	"rinha-go/internal/knn"
	"rinha-go/internal/server"
)

const readBufSize = 8192

var (
	bufPool = sync.Pool{
		New: func() any {
			b := make([]byte, readBufSize)
			return &b
		},
	}
	payloadPool = sync.Pool{
		New: func() any {
			return new(jsonp.Payload)
		},
	}
)

func main() {
	runtime.GOMAXPROCS(1)

	// Tighten scheduler timer slack and try SCHED_FIFO before we open the
	// index, so the warmup loop already runs at the lower jitter.
	applyLowLatencyTuning()

	indexPath := envOr("INDEX_PATH", "/data/index.bin")
	ix, err := idx.Open(indexPath)
	if err != nil {
		log.Fatalf("open index: %v", err)
	}

	if mlockEnabled() {
		if err := mlockAll(); err != nil {
			log.Printf("mlockall: %v (continuing)", err)
		}
	}

	// Self-warmup primes caches/TLB before we accept any traffic. Configurable
	// via WARMUP_MS. Default 700ms matches v10; throttled cgroups make longer
	// warmups stall the LB connect handshake long enough to break the harness.
	warmupMs := envInt("WARMUP_MS", 700)
	warmup(ix, time.Duration(warmupMs)*time.Millisecond)

	if ctrl := os.Getenv("CTRL_SOCK_PATH"); ctrl != "" {
		runCtrl(ix, ctrl)
		return
	}
	runTCP(ix)
}

// warmup primes L2/L3 caches and TLB by running random KNN scores for
// approximately d. Uses xorshift to generate vectors cheaply.
func warmup(ix *idx.Index, d time.Duration) {
	if d <= 0 {
		return
	}
	t0 := time.Now()
	var q [knn.DIMS]float32
	// Seed xorshift from current monotonic time.
	rng := uint64(time.Now().UnixNano()) | 1
	count := 0
	for {
		if time.Since(t0) >= d {
			break
		}
		// Tight batch to avoid syscalls between iterations.
		for k := 0; k < 64; k++ {
			for i := 0; i < knn.DIMS; i++ {
				rng ^= rng << 13
				rng ^= rng >> 7
				rng ^= rng << 17
				// Float in roughly [-1, 1].
				q[i] = float32(int32(rng>>32)) / float32(1<<31)
			}
			_ = knn.Score(ix, q)
			count++
		}
	}
	log.Printf("warmup: %d iters in %v", count, time.Since(t0))
}

func runTCP(ix *idx.Index) {
	port := envOr("PORT", "9999")
	addr := ":" + port
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	log.Printf("api listening on %s", addr)
	for {
		c, err := ln.Accept()
		if err != nil {
			log.Printf("accept: %v", err)
			continue
		}
		tc, _ := c.(*net.TCPConn)
		if tc != nil {
			_ = tc.SetNoDelay(true)
			_ = tc.SetKeepAlive(true)
		}
		go handleConn(ix, c)
	}
}

func handleConn(ix *idx.Index, c net.Conn) {
	defer c.Close()
	bp := bufPool.Get().(*[]byte)
	defer bufPool.Put(bp)
	buf := *bp
	payload := payloadPool.Get().(*jsonp.Payload)
	defer payloadPool.Put(payload)

	pos := 0
	for {
		if pos == len(buf) {
			return
		}
		n, err := c.Read(buf[pos:])
		if err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			return
		}
		pos += n
		for pos > 0 {
			req, perr := server.ParseRequest(buf[:pos])
			if perr != nil {
				if perr == server.ErrIncomplete {
					break
				}
				return
			}
			resp := server.Respond(ix, req, payload)
			if _, werr := c.Write(resp); werr != nil {
				return
			}
			if req.End >= pos {
				pos = 0
			} else {
				copy(buf, buf[req.End:pos])
				pos -= req.End
			}
		}
	}
}

func envOr(k, d string) string {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	return v
}

func envInt(k string, d int) int {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return d
	}
	return n
}

func mlockEnabled() bool {
	v := os.Getenv("MLOCK")
	if v == "" {
		return true
	}
	return v != "0"
}

var _ = errors.Is
var _ = io.EOF
var _ = rand.Int
