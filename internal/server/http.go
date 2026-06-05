package server

import (
	"bytes"

	idx "rinha-go/internal/index"
	"rinha-go/internal/jsonp"
	"rinha-go/internal/knn"
	"rinha-go/internal/normalize"
	"rinha-go/internal/tree"
)

const (
	approvedHdr = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 35\r\nconnection: keep-alive\r\n\r\n"
	deniedHdr   = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 36\r\nconnection: keep-alive\r\n\r\n"
)

var (
	ReadyResp    = []byte("HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 2\r\nconnection: keep-alive\r\n\r\nok")
	NotFoundResp = []byte("HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: keep-alive\r\n\r\n")
	BadResp      = []byte("HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")

	scores = [6][]byte{
		[]byte(approvedHdr + `{"approved":true,"fraud_score":0.0}`),
		[]byte(approvedHdr + `{"approved":true,"fraud_score":0.2}`),
		[]byte(approvedHdr + `{"approved":true,"fraud_score":0.4}`),
		[]byte(deniedHdr + `{"approved":false,"fraud_score":0.6}`),
		[]byte(deniedHdr + `{"approved":false,"fraud_score":0.8}`),
		[]byte(deniedHdr + `{"approved":false,"fraud_score":1.0}`),
	}
)

type Method int

const (
	MethodOther Method = iota
	MethodGetReady
	MethodPostScore
)

type Parsed struct {
	M    Method
	Body []byte
	End  int
}

func ParseRequest(buf []byte) (Parsed, error) {
	if len(buf) < 16 {
		return Parsed{}, ErrIncomplete
	}
	if bytes.HasPrefix(buf, []byte("POST /fraud-score")) {
		hdrEnd := bytes.Index(buf, []byte("\r\n\r\n"))
		if hdrEnd < 0 {
			return Parsed{}, ErrIncomplete
		}
		cl, ok := findContentLength(buf[:hdrEnd+2])
		if !ok {
			return Parsed{}, ErrBad
		}
		bodyStart := hdrEnd + 4
		total := bodyStart + cl
		if len(buf) < total {
			return Parsed{}, ErrIncomplete
		}
		return Parsed{M: MethodPostScore, Body: buf[bodyStart:total], End: total}, nil
	}
	if bytes.HasPrefix(buf, []byte("GET /ready")) {
		sep := bytes.Index(buf, []byte("\r\n\r\n"))
		if sep < 0 {
			return Parsed{}, ErrIncomplete
		}
		return Parsed{M: MethodGetReady, End: sep + 4}, nil
	}
	sep := bytes.Index(buf, []byte("\r\n\r\n"))
	if sep < 0 {
		return Parsed{}, ErrIncomplete
	}
	return Parsed{M: MethodOther, End: sep + 4}, nil
}

func findContentLength(hdrs []byte) (int, bool) {
	i := 0
	for i < len(hdrs) {
		nl := bytes.IndexByte(hdrs[i:], '\n')
		if nl < 0 {
			return 0, false
		}
		nl += i
		lineEnd := nl
		if nl > i && hdrs[nl-1] == '\r' {
			lineEnd = nl - 1
		}
		line := hdrs[i:lineEnd]
		i = nl + 1
		if len(line) < 15 {
			continue
		}
		if asciiEqlIgnoreCase(line[:15], "content-length:") {
			j := 15
			for j < len(line) && (line[j] == ' ' || line[j] == '\t') {
				j++
			}
			v := 0
			for j < len(line) && line[j] >= '0' && line[j] <= '9' {
				v = v*10 + int(line[j]-'0')
				j++
			}
			return v, true
		}
	}
	return 0, false
}

func asciiEqlIgnoreCase(a []byte, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for k := 0; k < len(a); k++ {
		x := a[k]
		y := b[k]
		if x >= 'A' && x <= 'Z' {
			x += 32
		}
		if y >= 'A' && y <= 'Z' {
			y += 32
		}
		if x != y {
			return false
		}
	}
	return true
}

func Respond(ix *idx.Index, p Parsed, payload *jsonp.Payload) []byte {
	switch p.M {
	case MethodGetReady:
		return ReadyResp
	case MethodPostScore:
		if err := jsonp.Parse(p.Body, payload); err != nil {
			return scores[0]
		}
		v := normalize.Vectorize(payload)
		qq := knn.Quantize(v)
		// Decision-tree fast-path: if the leaf is Mode A-safe, skip KNN entirely.
		// LeafSafeA was calibrated to introduce no FP/FN against an exact-kNN
		// reference, so any leaf flagged safe lets us return the tree's count
		// directly. Otherwise we fall through to the regular KNN pipeline.
		var frauds uint32
		tr := tree.Predict([16]int16(qq))
		if tree.LeafSafeA(tr.LeafID) {
			frauds = uint32(tr.Count)
		} else {
			frauds = knn.ScoreQ(ix, qq)
		}
		if frauds > 5 {
			frauds = 5
		}
		return scores[frauds]
	}
	return NotFoundResp
}

var (
	ErrIncomplete = errIncomplete{}
	ErrBad        = errBad{}
)

type errIncomplete struct{}

func (errIncomplete) Error() string { return "incomplete" }

type errBad struct{}

func (errBad) Error() string { return "bad" }
