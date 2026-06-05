package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"time"
)

const (
	DIMS           = 14
	PADDED         = 16
	LANES          = 8
	QuantScale     = 10000.0
	QuantMax       = 10000.0
	Magic          = 0x52494E48
	Version        = 4
	K              = 4096
	KMeansIters    = 15
	KMeansSample   = 500000
	SeedDefault    = 0xDEADBEEFCAFEBABE
)

type Header struct {
	Magic    uint32
	Version  uint32
	K        uint32
	N        uint32
	NBlocks  uint32
	Scale    float32
	Reserved [40]byte
}

type Rng struct{ s uint64 }

func newRng(seed uint64) *Rng {
	if seed == 0 {
		seed = 1
	}
	return &Rng{s: seed}
}
func (r *Rng) next() uint64 {
	x := r.s
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	r.s = x
	return x
}
func (r *Rng) pick(n int) int { return int(r.next() % uint64(n)) }
func (r *Rng) unit() float64 {
	return float64(r.next()>>11) / float64(uint64(1)<<53)
}

func main() {
	inPath := os.Getenv("INPUT")
	outPath := os.Getenv("OUTPUT")
	if inPath == "" || outPath == "" {
		log.Fatalf("set INPUT and OUTPUT")
	}

	t0 := time.Now()
	bytes, err := os.ReadFile(inPath)
	if err != nil {
		log.Fatalf("read %s: %v", inPath, err)
	}
	log.Printf("read %d MB in %v", len(bytes)/1024/1024, time.Since(t0))

	t0 = time.Now()
	vecs, labels := parseAll(bytes)
	log.Printf("parsed %d vecs in %v", len(vecs), time.Since(t0))
	if len(vecs) == 0 {
		log.Fatalf("no vectors")
	}

	t0 = time.Now()
	centers := make([][DIMS]float32, K)
	kmeansPlusPlus(vecs, centers, KMeansIters)
	log.Printf("k-means++ %v", time.Since(t0))

	t0 = time.Now()
	n := len(vecs)
	assignments := make([]uint32, n)
	counts := make([]uint32, K)
	for i, v := range vecs {
		c := nearest(centers, v)
		assignments[i] = c
		counts[c]++
	}
	log.Printf("assign %v", time.Since(t0))

	blocksPer := make([]uint32, K)
	var totalBlocks uint32
	for i, c := range counts {
		b := (c + LANES - 1) / LANES
		blocksPer[i] = b
		totalBlocks += b
	}
	blockOffsets := make([]uint32, K+1)
	for i := 0; i < K; i++ {
		blockOffsets[i+1] = blockOffsets[i] + blocksPer[i]
	}

	bboxMin := make([][PADDED]int16, K)
	bboxMax := make([][PADDED]int16, K)
	for i := 0; i < K; i++ {
		for j := 0; j < PADDED; j++ {
			bboxMin[i][j] = math.MaxInt16
			bboxMax[i][j] = math.MinInt16
		}
	}

	codesWords := int(totalBlocks) * PADDED * LANES
	codes := make([]int16, codesWords)
	lbls := make([]byte, int(totalBlocks)*LANES)

	clusterIdx := make([][]uint32, K)
	for ci, c := range counts {
		clusterIdx[ci] = make([]uint32, 0, c)
	}
	for i, c := range assignments {
		clusterIdx[c] = append(clusterIdx[c], uint32(i))
	}
	for ci, arr := range clusterIdx {
		center := centers[ci]
		sort.Slice(arr, func(a, b int) bool {
			return dist2F32(vecs[arr[a]], center) < dist2F32(vecs[arr[b]], center)
		})
	}

	curBlock := make([]uint32, K)
	curLane := make([]uint8, K)
	for i := 0; i < K; i++ {
		curBlock[i] = blockOffsets[i]
	}
	for c, arr := range clusterIdx {
		for _, i := range arr {
			v := vecs[i]
			var q [PADDED]int16
			for j := 0; j < DIMS; j++ {
				x := v[j] * QuantScale
				if x > QuantMax {
					x = QuantMax
				} else if x < -QuantMax {
					x = -QuantMax
				}
				q[j] = int16(roundF32(x))
			}
			for j := 0; j < PADDED; j++ {
				if q[j] < bboxMin[c][j] {
					bboxMin[c][j] = q[j]
				}
				if q[j] > bboxMax[c][j] {
					bboxMax[c][j] = q[j]
				}
			}
			blockIdx := int(curBlock[c])
			lane := int(curLane[c])
			blockBase := blockIdx * PADDED * LANES
			for p := 0; p < PADDED/2; p++ {
				pairBase := blockBase + p*LANES*2
				codes[pairBase+lane*2] = q[p*2]
				codes[pairBase+lane*2+1] = q[p*2+1]
			}
			lbls[blockIdx*LANES+lane] = labels[i]
			curLane[c]++
			if curLane[c] == LANES {
				curLane[c] = 0
				curBlock[c]++
			}
		}
	}

	nCentroidBlocks := (K + LANES - 1) / LANES
	centroidSoA := make([]int16, nCentroidBlocks*PADDED*LANES)
	for ci, c := range centers {
		var q [PADDED]int16
		for j := 0; j < DIMS; j++ {
			x := c[j] * QuantScale
			if x > QuantMax {
				x = QuantMax
			} else if x < -QuantMax {
				x = -QuantMax
			}
			q[j] = int16(roundF32(x))
		}
		blockIdx := ci / LANES
		lane := ci % LANES
		blockBase := blockIdx * PADDED * LANES
		for p := 0; p < PADDED/2; p++ {
			pairBase := blockBase + p*LANES*2
			centroidSoA[pairBase+lane*2] = q[p*2]
			centroidSoA[pairBase+lane*2+1] = q[p*2+1]
		}
	}

	t0 = time.Now()
	out, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("create %s: %v", outPath, err)
	}
	hdr := Header{
		Magic:   Magic,
		Version: Version,
		K:       K,
		N:       uint32(n),
		NBlocks: totalBlocks,
		Scale:   QuantScale,
	}
	if err := binary.Write(out, binary.LittleEndian, &hdr); err != nil {
		log.Fatalf("write hdr: %v", err)
	}
	if err := writeI16(out, centroidSoA); err != nil {
		log.Fatal(err)
	}
	for _, b := range bboxMin {
		if err := writeI16(out, b[:]); err != nil {
			log.Fatal(err)
		}
	}
	for _, b := range bboxMax {
		if err := writeI16(out, b[:]); err != nil {
			log.Fatal(err)
		}
	}
	if err := writeU32(out, blockOffsets); err != nil {
		log.Fatal(err)
	}
	if err := writeU32(out, counts); err != nil {
		log.Fatal(err)
	}
	if err := writeI16(out, codes); err != nil {
		log.Fatal(err)
	}
	if _, err := out.Write(lbls); err != nil {
		log.Fatal(err)
	}
	if err := out.Close(); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %s in %v - n=%d k=%d blocks=%d", outPath, time.Since(t0), n, K, totalBlocks)
}

func writeI16(w *os.File, vs []int16) error {
	buf := make([]byte, len(vs)*2)
	for i, v := range vs {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}
	_, err := w.Write(buf)
	return err
}

func writeU32(w *os.File, vs []uint32) error {
	buf := make([]byte, len(vs)*4)
	for i, v := range vs {
		binary.LittleEndian.PutUint32(buf[i*4:], v)
	}
	_, err := w.Write(buf)
	return err
}

func roundF32(x float32) float32 {
	if x >= 0 {
		return float32(math.Floor(float64(x) + 0.5))
	}
	return float32(math.Ceil(float64(x) - 0.5))
}

func dist2F32(a, b [DIMS]float32) float32 {
	var s float32
	for j := 0; j < DIMS; j++ {
		d := a[j] - b[j]
		s += d * d
	}
	return s
}

func nearest(centers [][DIMS]float32, p [DIMS]float32) uint32 {
	var best uint32
	bestD := float32(math.Inf(1))
	for i, c := range centers {
		d := dist2F32(p, c)
		if d < bestD {
			bestD = d
			best = uint32(i)
		}
	}
	return best
}

func kmeansPlusPlus(points [][DIMS]float32, centers [][DIMS]float32, iters int) {
	rng := newRng(SeedDefault)
	sampleN := KMeansSample
	if sampleN > len(points) {
		sampleN = len(points)
	}
	sample := make([][DIMS]float32, sampleN)
	for i := 0; i < sampleN; i++ {
		sample[i] = points[rng.pick(len(points))]
	}
	centers[0] = sample[rng.pick(sampleN)]
	minDist := make([]float32, sampleN)
	for i := 0; i < sampleN; i++ {
		minDist[i] = dist2F32(sample[i], centers[0])
	}
	for ci := 1; ci < len(centers); ci++ {
		var sum float64
		for _, d := range minDist {
			sum += float64(d)
		}
		if sum <= 0 {
			centers[ci] = sample[rng.pick(sampleN)]
		} else {
			target := rng.unit() * sum
			var acc float64
			chosen := sampleN - 1
			for i, d := range minDist {
				acc += float64(d)
				if acc >= target {
					chosen = i
					break
				}
			}
			centers[ci] = sample[chosen]
		}
		for i := 0; i < sampleN; i++ {
			d := dist2F32(sample[i], centers[ci])
			if d < minDist[i] {
				minDist[i] = d
			}
		}
		if ci%256 == 0 {
			log.Printf("  k++ %d/%d", ci, len(centers))
		}
	}

	sums := make([][DIMS]float64, len(centers))
	counts := make([]uint64, len(centers))
	for it := 0; it < iters; it++ {
		for i := range sums {
			sums[i] = [DIMS]float64{}
		}
		for i := range counts {
			counts[i] = 0
		}
		var sse float64
		for _, p := range sample {
			best := 0
			bestD := float32(math.Inf(1))
			for ci, c := range centers {
				d := dist2F32(p, c)
				if d < bestD {
					bestD = d
					best = ci
				}
			}
			counts[best]++
			for j := 0; j < DIMS; j++ {
				sums[best][j] += float64(p[j])
			}
			sse += float64(bestD)
		}
		for i := range centers {
			if counts[i] > 0 {
				inv := 1.0 / float64(counts[i])
				for j := 0; j < DIMS; j++ {
					centers[i][j] = float32(sums[i][j] * inv)
				}
			} else {
				centers[i] = sample[rng.pick(sampleN)]
			}
		}
		log.Printf("  iter %d sse=%v", it, sse/float64(len(sample)))
	}
}

func parseAll(buf []byte) ([][DIMS]float32, []byte) {
	var vecs [][DIMS]float32
	var labels []byte
	i := 0
	for i < len(buf) && buf[i] != '[' {
		i++
	}
	if i == len(buf) {
		return nil, nil
	}
	i++
	for i < len(buf) {
		for i < len(buf) && isSpace(buf[i]) {
			i++
		}
		if i >= len(buf) || buf[i] == ']' {
			return vecs, labels
		}
		if buf[i] != '{' {
			i++
			continue
		}
		i++
		var v [DIMS]float32
		var lab byte
		gotV := false
		gotL := false
		for i < len(buf) && buf[i] != '}' {
			for i < len(buf) && (isSpace(buf[i]) || buf[i] == ',') {
				i++
			}
			if i >= len(buf) || buf[i] == '}' {
				break
			}
			if buf[i] != '"' {
				i++
				continue
			}
			i++
			ks := i
			for i < len(buf) && buf[i] != '"' {
				i++
			}
			key := buf[ks:i]
			if i < len(buf) {
				i++
			}
			for i < len(buf) && (isSpace(buf[i]) || buf[i] == ':') {
				i++
			}
			switch string(key) {
			case "vector":
				if err := parseVector(buf, &i, &v); err != nil {
					return nil, nil
				}
				gotV = true
			case "label":
				lab = parseLabel(buf, &i)
				gotL = true
			default:
				skipVal(buf, &i)
			}
		}
		if i < len(buf) {
			i++
		}
		if gotV && gotL {
			vecs = append(vecs, v)
			labels = append(labels, lab)
		}
		for i < len(buf) && (isSpace(buf[i]) || buf[i] == ',') {
			i++
		}
	}
	return vecs, labels
}

func parseVector(buf []byte, i *int, out *[DIMS]float32) error {
	for *i < len(buf) && buf[*i] != '[' {
		*i++
	}
	if *i >= len(buf) {
		return fmt.Errorf("bad")
	}
	*i++
	k := 0
	for *i < len(buf) && k < DIMS {
		for *i < len(buf) && (isSpace(buf[*i]) || buf[*i] == ',') {
			*i++
		}
		if buf[*i] == ']' {
			break
		}
		start := *i
		for *i < len(buf) {
			c := buf[*i]
			if c == ',' || c == ']' || isSpace(c) {
				break
			}
			*i++
		}
		f, err := parseFloatBytes(buf[start:*i])
		if err != nil {
			return err
		}
		out[k] = f
		k++
	}
	for *i < len(buf) && buf[*i] != ']' {
		*i++
	}
	if *i < len(buf) {
		*i++
	}
	return nil
}

func parseFloatBytes(b []byte) (float32, error) {
	s := string(b)
	var f float64
	_, err := fmt.Sscanf(s, "%g", &f)
	return float32(f), err
}

func parseLabel(buf []byte, i *int) byte {
	for *i < len(buf) && buf[*i] != '"' {
		*i++
	}
	*i++
	start := *i
	for *i < len(buf) && buf[*i] != '"' {
		*i++
	}
	s := buf[start:*i]
	*i++
	if string(s) == "fraud" {
		return 1
	}
	return 0
}

func skipVal(buf []byte, i *int) {
	for *i < len(buf) && isSpace(buf[*i]) {
		*i++
	}
	if *i >= len(buf) {
		return
	}
	c := buf[*i]
	if c == '"' {
		*i++
		for *i < len(buf) && buf[*i] != '"' {
			*i++
		}
		if *i < len(buf) {
			*i++
		}
		return
	}
	if c == '{' || c == '[' {
		closer := byte('}')
		if c == '[' {
			closer = ']'
		}
		depth := 1
		*i++
		for *i < len(buf) && depth > 0 {
			if buf[*i] == c {
				depth++
			} else if buf[*i] == closer {
				depth--
			}
			*i++
		}
		return
	}
	for *i < len(buf) {
		cc := buf[*i]
		if cc == ',' || cc == '}' || cc == ']' {
			return
		}
		*i++
	}
}

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\r' || c == '\n' }
