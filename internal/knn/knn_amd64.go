//go:build amd64

package knn

import (
	idx "rinha-go/internal/index"
)

//go:noescape
func blkDistAVX2(vectors *int16, blockIdx int, q *int16, out *[8]int64)

//go:noescape
func blkDistPruneAVX2(vectors *int16, blockIdx int, q *int16, threshold int64, out *[8]int64) int32

//go:noescape
func bboxLowerBoundAVX2(mn *int16, mx *int16, q *int16) int64

//go:noescape
func prefetchBlockT0(vectors *int16, blockIdx int)

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

func bboxLowerBoundFast(q *[idx.PADDED_DIMS]int16, ix *idx.Index, cluster uint32) int64 {
	base := int(cluster) * idx.PADDED_DIMS
	return bboxLowerBoundAVX2(&ix.BBoxMin[base], &ix.BBoxMax[base], &q[0])
}

func prefetchBlock(vectors []int16, blockIdx int) {
	if blockIdx*idx.PADDED_DIMS*idx.LANES >= len(vectors) {
		return
	}
	prefetchBlockT0(&vectors[0], blockIdx)
}
