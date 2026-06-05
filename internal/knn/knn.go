package knn

import (
	"math"

	idx "rinha-go/internal/index"
)

const (
	DIMS       = idx.DIMS
	PADDED     = idx.PADDED_DIMS
	LANES      = idx.LANES
	TopK       = idx.TopK
	MaxProbes  = idx.MaxProbes
	NPROBE     = idx.NPROBE
	RepairMin  = idx.RepairMin
	RepairMax  = idx.RepairMax
	SeenWords  = idx.SeenWords
	EarlyDist  = idx.EarlyDist
	QuantScale = idx.QuantScale
	QuantMax   = idx.QuantMax
)

const PAIRS = PADDED / 2

const maxI64 = int64(math.MaxInt64)
const maxU32 = uint32(0xFFFFFFFF)

type Probe struct {
	Cluster uint32
	Dist    int64
}

func Quantize(v [DIMS]float32) [PADDED]int16 {
	var out [PADDED]int16
	for i := 0; i < DIMS; i++ {
		x := v[i] * QuantScale
		if x > QuantMax {
			x = QuantMax
		} else if x < -QuantMax {
			x = -QuantMax
		}
		out[i] = int16(roundF32(x))
	}
	return out
}

func roundF32(x float32) float32 {
	if x >= 0 {
		return float32(math.Floor(float64(x) + 0.5))
	}
	return float32(math.Ceil(float64(x) - 0.5))
}

func Score(ix *idx.Index, q [DIMS]float32) uint32 {
	qq := Quantize(q)
	return ScoreQ(ix, qq)
}

// ScoreQ runs the full KNN pipeline on an already-quantized query vector.
// Splitting Score lets callers (e.g. the decision-tree fast-path) avoid
// re-quantizing when they have already produced [PADDED]int16.
func ScoreQ(ix *idx.Index, qq [PADDED]int16) uint32 {
	var probes [MaxProbes]Probe
	for i := range probes {
		probes[i] = Probe{Cluster: maxU32, Dist: maxI64}
	}
	var probeCount int

	ncb := ix.NCentroidBlk
	k := uint32(ix.K)
	for cb := 0; cb < ncb; cb++ {
		dists := blkDist(ix.Centroids, cb, &qq)
		ciBase := uint32(cb) * LANES
		laneMax := uint32(LANES)
		if remaining := k - ciBase; remaining < laneMax {
			laneMax = remaining
		}
		for lane := uint32(0); lane < laneMax; lane++ {
			ci := ciBase + lane
			if ix.Counts[ci] == 0 {
				continue
			}
			insertProbe(&probes, &probeCount, ci, dists[lane])
		}
	}
	heapToSorted(&probes, probeCount)

	var bestD [TopK]int64
	var bestL [TopK]uint8
	for i := range bestD {
		bestD[i] = maxI64
	}

	nInitial := NPROBE
	if probeCount < nInitial {
		nInitial = probeCount
	}
	for pi := 0; pi < nInitial; pi++ {
		scanCluster(ix, &qq, probes[pi].Cluster, &bestD, &bestL)
	}

	var frauds uint32
	for j := 0; j < TopK; j++ {
		frauds += uint32(bestL[j])
	}
	allFraud := frauds == TopK
	tight := bestD[TopK-1] <= EarlyDist
	if allFraud && tight {
		return frauds
	}

	repairFast(ix, &qq, probes[:probeCount], &bestD, &bestL)
	frauds = 0
	for j := 0; j < TopK; j++ {
		frauds += uint32(bestL[j])
	}

	if frauds <= RepairMax {
		repairFull(ix, &qq, probes[:probeCount], &bestD, &bestL)
		frauds = 0
		for j := 0; j < TopK; j++ {
			frauds += uint32(bestL[j])
		}
	}
	return frauds
}

func repairFull(ix *idx.Index, q *[PADDED]int16, probes []Probe, bestD *[TopK]int64, bestL *[TopK]uint8) {
	var seenMask [SeenWords]uint64
	for _, p := range probes {
		if p.Cluster == maxU32 {
			break
		}
		w := int(p.Cluster) / 64
		b := uint(p.Cluster % 64)
		if w < SeenWords {
			seenMask[w] |= uint64(1) << b
		}
	}
	for ci := uint32(0); ci < uint32(ix.K); ci++ {
		if ix.Counts[ci] == 0 {
			continue
		}
		w := int(ci) / 64
		b := uint(ci % 64)
		if (seenMask[w]>>b)&1 != 0 {
			continue
		}
		lb := bboxLowerBound(q, ix, ci)
		if lb >= bestD[TopK-1] {
			continue
		}
		scanCluster(ix, q, ci, bestD, bestL)
	}
}

func repairFast(ix *idx.Index, q *[PADDED]int16, probes []Probe, bestD *[TopK]int64, bestL *[TopK]uint8) {
	for pi := NPROBE; pi < len(probes); pi++ {
		p := probes[pi]
		if p.Cluster == maxU32 {
			break
		}
		lb := bboxLowerBound(q, ix, p.Cluster)
		if lb >= bestD[TopK-1] {
			continue
		}
		scanCluster(ix, q, p.Cluster, bestD, bestL)
		var frauds uint8
		for j := 0; j < TopK; j++ {
			frauds += bestL[j]
		}
		if frauds < RepairMin || frauds > RepairMax {
			if bestD[TopK-1] <= EarlyDist {
				break
			}
		}
	}
}

func scanCluster(ix *idx.Index, q *[PADDED]int16, cluster uint32, bestD *[TopK]int64, bestL *[TopK]uint8) {
	startBlock := int(ix.BlockOffsets[cluster])
	endBlock := int(ix.BlockOffsets[cluster+1])
	total := ix.Counts[cluster]
	if total == 0 {
		return
	}
	var processed uint32
	for blk := startBlock; blk < endBlock; blk++ {
		threshold := bestD[TopK-1]
		dists, pruned := blkDistPrune(ix.Vectors, blk, q, threshold)
		laneN := uint32(LANES)
		if remaining := total - processed; remaining < laneN {
			laneN = remaining
		}
		processed += laneN
		if pruned {
			continue
		}
		labBase := blk * LANES
		for lane := uint32(0); lane < laneN; lane++ {
			d := dists[lane]
			if d < bestD[TopK-1] {
				label := ix.Labels[labBase+int(lane)]
				insertBest(d, label, bestD, bestL)
			}
		}
	}
}

func insertProbe(probes *[MaxProbes]Probe, count *int, cluster uint32, dist int64) {
	if *count < MaxProbes {
		probes[*count] = Probe{Cluster: cluster, Dist: dist}
		heapSiftUp(probes, *count)
		*count++
		return
	}
	if dist >= probes[0].Dist {
		return
	}
	probes[0] = Probe{Cluster: cluster, Dist: dist}
	heapSiftDown(probes, 0, MaxProbes)
}

func heapSiftUp(heap *[MaxProbes]Probe, start int) {
	i := start
	for i > 0 {
		parent := (i - 1) / 2
		if heap[i].Dist > heap[parent].Dist {
			heap[i], heap[parent] = heap[parent], heap[i]
			i = parent
		} else {
			break
		}
	}
}

func heapSiftDown(heap *[MaxProbes]Probe, start, size int) {
	i := start
	for {
		left := 2*i + 1
		right := 2*i + 2
		largest := i
		if left < size && heap[left].Dist > heap[largest].Dist {
			largest = left
		}
		if right < size && heap[right].Dist > heap[largest].Dist {
			largest = right
		}
		if largest == i {
			break
		}
		heap[i], heap[largest] = heap[largest], heap[i]
		i = largest
	}
}

func heapToSorted(probes *[MaxProbes]Probe, count int) {
	n := count
	for n > 1 {
		n--
		probes[0], probes[n] = probes[n], probes[0]
		heapSiftDown(probes, 0, n)
	}
}

func insertBest(dist int64, label uint8, bestD *[TopK]int64, bestL *[TopK]uint8) {
	if dist >= bestD[TopK-1] {
		return
	}
	pos := TopK - 1
	for pos > 0 && dist < bestD[pos-1] {
		bestD[pos] = bestD[pos-1]
		bestL[pos] = bestL[pos-1]
		pos--
	}
	bestD[pos] = dist
	bestL[pos] = label
}


func bboxLowerBound(q *[PADDED]int16, ix *idx.Index, cluster uint32) int64 {
	mnBase := int(cluster) * PADDED
	mn := ix.BBoxMin[mnBase : mnBase+PADDED]
	mx := ix.BBoxMax[mnBase : mnBase+PADDED]
	var sum int64
	for i := 0; i < PADDED; i++ {
		qv := int32(q[i])
		mnv := int32(mn[i])
		mxv := int32(mx[i])
		var gap int32
		if qv < mnv {
			gap = mnv - qv
		} else if qv > mxv {
			gap = qv - mxv
		}
		sum += int64(gap) * int64(gap)
	}
	return sum
}
