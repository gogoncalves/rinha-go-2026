//go:build !amd64

package knn

import idx "rinha-go/internal/index"

func blkDist(vectors []int16, blockIdx int, q *[idx.PADDED_DIMS]int16) [idx.LANES]int64 {
	blockBase := blockIdx * PADDED * LANES
	var acc0, acc1 [LANES]int64
	for p := 0; p < PAIRS; p++ {
		q0 := int32(q[p*2])
		q1 := int32(q[p*2+1])
		pairBase := blockBase + p*LANES*2
		for lane := 0; lane < LANES; lane++ {
			off := pairBase + lane*2
			a := int32(vectors[off])
			b := int32(vectors[off+1])
			d0 := q0 - a
			d1 := q1 - b
			s := int64(d0*d0) + int64(d1*d1)
			if (p & 1) == 0 {
				acc0[lane] += s
			} else {
				acc1[lane] += s
			}
		}
	}
	var out [LANES]int64
	for i := 0; i < LANES; i++ {
		out[i] = acc0[i] + acc1[i]
	}
	return out
}

func blkDistPrune(vectors []int16, blockIdx int, q *[idx.PADDED_DIMS]int16, threshold int64) ([idx.LANES]int64, bool) {
	blockBase := blockIdx * PADDED * LANES
	var acc0, acc1 [LANES]int64
	for p := 0; p < PAIRS; p++ {
		q0 := int32(q[p*2])
		q1 := int32(q[p*2+1])
		pairBase := blockBase + p*LANES*2
		for lane := 0; lane < LANES; lane++ {
			off := pairBase + lane*2
			a := int32(vectors[off])
			b := int32(vectors[off+1])
			d0 := q0 - a
			d1 := q1 - b
			s := int64(d0*d0) + int64(d1*d1)
			if (p & 1) == 0 {
				acc0[lane] += s
			} else {
				acc1[lane] += s
			}
		}
		if p == 2 || p == 4 {
			allOver := true
			for lane := 0; lane < LANES; lane++ {
				if acc0[lane]+acc1[lane] < threshold {
					allOver = false
					break
				}
			}
			if allOver {
				var zero [LANES]int64
				return zero, true
			}
		}
	}
	var out [LANES]int64
	for i := 0; i < LANES; i++ {
		out[i] = acc0[i] + acc1[i]
	}
	return out, false
}
