// AVX2 pairwise squared-distance for KNN block scans.
// Block layout (i16): per block 8 pairs x 8 lanes x 2 i16 = 128 i16 = 256 bytes.
// For pair p, 16 i16 hold (lane0_d0, lane0_d1, lane1_d0, lane1_d1, ..., lane7_d0, lane7_d1).
// We broadcast (q[2p], q[2p+1]) as a 32-bit pair 8 times into a YMM, subtract, then
// VPMADDWD (diff,diff) computes (d0*d0 + d1*d1) per lane as 8 i32.
// We widen i32->i64 (lo/hi halves) and accumulate two YMM lanes (4 i64 each)
// across pairs, using two accumulators (acc0/acc1) to maximize ILP.

#include "textflag.h"

// func blkDistAVX2(vectors *int16, blockIdx int, q *int16, out *[8]int64)
TEXT ·blkDistAVX2(SB), NOSPLIT, $0-32
	MOVQ vectors+0(FP), SI
	MOVQ blockIdx+8(FP), AX
	MOVQ q+16(FP), DX
	MOVQ out+24(FP), DI

	// SI += blockIdx * 256
	SHLQ $8, AX
	ADDQ AX, SI

	// Zero accumulators: Y8..Y11 = acc0_lo, acc0_hi, acc1_lo, acc1_hi (4 i64 each).
	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11

	// ---- pair 0 (acc0) ----
	VPBROADCASTD (DX), Y0           // broadcast q[0..1] as i32
	VMOVDQU (SI), Y1                // 16 i16 = pair0
	VPSUBW Y1, Y0, Y2               // diff = q - block
	VPMADDWD Y2, Y2, Y3             // 8 i32 = d0^2 + d1^2
	VPMOVSXDQ X3, Y4                // lo 4 i32 -> 4 i64
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5                // hi 4 i32 -> 4 i64
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// ---- pair 1 (acc1) ----
	VPBROADCASTD 4(DX), Y0
	VMOVDQU 32(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// ---- pair 2 (acc0) ----
	VPBROADCASTD 8(DX), Y0
	VMOVDQU 64(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// ---- pair 3 (acc1) ----
	VPBROADCASTD 12(DX), Y0
	VMOVDQU 96(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// ---- pair 4 (acc0) ----
	VPBROADCASTD 16(DX), Y0
	VMOVDQU 128(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// ---- pair 5 (acc1) ----
	VPBROADCASTD 20(DX), Y0
	VMOVDQU 160(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// ---- pair 6 (acc0) ----
	VPBROADCASTD 24(DX), Y0
	VMOVDQU 192(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// ---- pair 7 (acc1) ----
	VPBROADCASTD 28(DX), Y0
	VMOVDQU 224(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// merge acc0 + acc1
	VPADDQ Y10, Y8, Y8
	VPADDQ Y11, Y9, Y9

	// write 8 i64
	VMOVDQU Y8, 0(DI)
	VMOVDQU Y9, 32(DI)
	VZEROUPPER
	RET

// func blkDistPruneAVX2(vectors *int16, blockIdx int, q *int16, threshold int64, out *[8]int64) int32
// Returns 1 if pruned (all 8 lanes >= threshold) at checkpoints p=2 or p=4, else 0 (and writes full distances).
TEXT ·blkDistPruneAVX2(SB), NOSPLIT, $0-44
	MOVQ vectors+0(FP), SI
	MOVQ blockIdx+8(FP), AX
	MOVQ q+16(FP), DX
	MOVQ threshold+24(FP), CX
	MOVQ out+32(FP), DI

	SHLQ $8, AX
	ADDQ AX, SI

	// Broadcast threshold to Y12 (4 i64).
	MOVQ CX, X12
	VPBROADCASTQ X12, Y12

	VPXOR Y8, Y8, Y8
	VPXOR Y9, Y9, Y9
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11

	// pair 0 (acc0)
	VPBROADCASTD (DX), Y0
	VMOVDQU (SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// pair 1 (acc1)
	VPBROADCASTD 4(DX), Y0
	VMOVDQU 32(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// pair 2 (acc0)
	VPBROADCASTD 8(DX), Y0
	VMOVDQU 64(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// Checkpoint p=2: partial = acc0 + acc1 per-lane.
	VPADDQ Y10, Y8, Y6
	VPADDQ Y11, Y9, Y7
	// VPCMPGTQ A,B,R -> R = B > A (signed). We want "partial >= thr".
	// Use !(thr > partial) = NAND.
	// Compute Y13 = (thr > partial_lo), Y14 = (thr > partial_hi). If both all-zero -> all lanes >= thr -> prune.
	VPCMPGTQ Y6, Y12, Y13
	VPCMPGTQ Y7, Y12, Y14
	VPOR Y14, Y13, Y15
	VPTEST Y15, Y15
	JNZ pair3p

	// All lanes already >= thr: prune.
	MOVL $1, AX
	VZEROUPPER
	MOVL AX, ret+40(FP)
	RET

pair3p:
	// pair 3 (acc1)
	VPBROADCASTD 12(DX), Y0
	VMOVDQU 96(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// pair 4 (acc0)
	VPBROADCASTD 16(DX), Y0
	VMOVDQU 128(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// Checkpoint p=4.
	VPADDQ Y10, Y8, Y6
	VPADDQ Y11, Y9, Y7
	VPCMPGTQ Y6, Y12, Y13
	VPCMPGTQ Y7, Y12, Y14
	VPOR Y14, Y13, Y15
	VPTEST Y15, Y15
	JNZ pair5p

	MOVL $1, AX
	VZEROUPPER
	MOVL AX, ret+40(FP)
	RET

pair5p:
	// pair 5 (acc1)
	VPBROADCASTD 20(DX), Y0
	VMOVDQU 160(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	// pair 6 (acc0)
	VPBROADCASTD 24(DX), Y0
	VMOVDQU 192(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y8, Y8
	VPADDQ Y5, Y9, Y9

	// pair 7 (acc1)
	VPBROADCASTD 28(DX), Y0
	VMOVDQU 224(SI), Y1
	VPSUBW Y1, Y0, Y2
	VPMADDWD Y2, Y2, Y3
	VPMOVSXDQ X3, Y4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXDQ X5, Y5
	VPADDQ Y4, Y10, Y10
	VPADDQ Y5, Y11, Y11

	VPADDQ Y10, Y8, Y8
	VPADDQ Y11, Y9, Y9
	VMOVDQU Y8, 0(DI)
	VMOVDQU Y9, 32(DI)
	XORL AX, AX
	VZEROUPPER
	MOVL AX, ret+40(FP)
	RET

// func bboxLowerBoundAVX2(mn *int16, mx *int16, q *int16) int64
// Computes sum-of-squared gaps between q and the [mn,mx] box across PADDED=16
// i16 dims using one 256-bit pass. gap = max(max(mn-q,0), max(q-mx,0)).
// VPMADDWD gives 8 i32 lane sums; we widen to 8 i64 and horizontal-add.
TEXT ·bboxLowerBoundAVX2(SB), NOSPLIT, $0-32
	MOVQ mn+0(FP), AX
	MOVQ mx+8(FP), BX
	MOVQ q+16(FP), CX

	VMOVDQU (AX), Y0                // mn (16 i16)
	VMOVDQU (BX), Y1                // mx
	VMOVDQU (CX), Y2                // q

	VPSUBW Y2, Y0, Y3               // below_diff = mn - q
	VPSUBW Y1, Y2, Y4               // above_diff = q - mx
	VPXOR Y5, Y5, Y5
	VPMAXSW Y5, Y3, Y3              // max(below_diff, 0)
	VPMAXSW Y5, Y4, Y4              // max(above_diff, 0)
	VPMAXSW Y4, Y3, Y3              // gap = max(below, above) (only one side is positive per lane)
	VPMADDWD Y3, Y3, Y3             // 8 i32 = pair-sum of squares

	// widen 8 i32 -> 8 i64, sum to one scalar
	VPMOVSXDQ X3, Y6                // low 4 i32 -> 4 i64
	VEXTRACTI128 $1, Y3, X7
	VPMOVSXDQ X7, Y7                // high 4 i32 -> 4 i64
	VPADDQ Y7, Y6, Y6               // 4 i64
	VEXTRACTI128 $1, Y6, X7
	VPADDQ X7, X6, X6               // 2 i64
	VPSHUFD $0x4E, X6, X7           // swap hi/lo 64-bit lanes
	VPADDQ X7, X6, X6               // 1 i64
	VMOVQ X6, AX
	VZEROUPPER
	MOVQ AX, ret+24(FP)
	RET

// func prefetchBlockT0(vectors *int16, blockIdx int)
// Issues PREFETCHT0 on the 4 cache lines (256B) of one block. Used to prime
// L1d ahead of the scanCluster block loop (ahead=4).
TEXT ·prefetchBlockT0(SB), NOSPLIT, $0-16
	MOVQ vectors+0(FP), SI
	MOVQ blockIdx+8(FP), AX
	SHLQ $8, AX                     // blockIdx * 256
	ADDQ AX, SI
	PREFETCHT0 0(SI)
	PREFETCHT0 64(SI)
	PREFETCHT0 128(SI)
	PREFETCHT0 192(SI)
	RET
