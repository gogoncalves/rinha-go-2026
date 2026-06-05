//go:build amd64

package knn

import (
	idx "rinha-go/internal/index"
)

//go:noescape
func blkDistAVX2(vectors *int16, blockIdx int, q *int16, out *[8]int64)

//go:noescape
func blkDistPruneAVX2(vectors *int16, blockIdx int, q *int16, threshold int64, out *[8]int64) int32

func blkDist(vectors []int16, blockIdx int, q *[idx.PADDED_DIMS]int16) [idx.LANES]int64 {
	var out [idx.LANES]int64
	blkDistAVX2(&vectors[0], blockIdx, &q[0], &out)
	return out
}

func blkDistPrune(vectors []int16, blockIdx int, q *[idx.PADDED_DIMS]int16, threshold int64) ([idx.LANES]int64, bool) {
	var out [idx.LANES]int64
	r := blkDistPruneAVX2(&vectors[0], blockIdx, &q[0], threshold, &out)
	return out, r != 0
}
