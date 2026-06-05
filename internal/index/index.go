package index

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"reflect"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	Magic       uint32 = 0x52494E48
	Version     uint32 = 4
	QuantScale         = 10000.0
	QuantMax           = 10000.0
	DIMS               = 14
	PADDED_DIMS        = 16
	LANES              = 8
	PAIRS              = (DIMS + 1) / 2
	BlockBytes         = PADDED_DIMS * LANES * 2

	NPROBE      = 1
	RepairExtra = 32
	MaxProbes   = NPROBE + RepairExtra
	MaxK        = 4096
	SeenWords   = (MaxK + 63) / 64
	TopK        = 5
	RepairMin   = 1
	RepairMax   = 4
)

const EarlyDist int64 = 1000 * 1000 * DIMS

const HeaderSize = 64

type Header struct {
	Magic    uint32
	Version  uint32
	K        uint32
	N        uint32
	NBlocks  uint32
	Scale    float32
	Reserved [40]byte
}

type Index struct {
	mmap         []byte
	Hdr          Header
	Centroids    []int16
	NCentroidBlk int
	BBoxMin      []int16
	BBoxMax      []int16
	BlockOffsets []uint32
	Counts       []uint32
	Vectors      []int16
	Labels       []byte
	K            int
	N            int
	NBlocks      int
}

func Open(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(st.Size())
	if size < HeaderSize {
		return nil, errors.New("index too small")
	}
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}
	_ = unix.Madvise(data, unix.MADV_WILLNEED)
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	// MADV_HUGEPAGE = 14 — reduce TLB misses on hot mmap region. Opt-in via
	// env: forcing THP on a ~95MB private+mlocked mapping can inflate the
	// cgroup RSS enough to OOM at 140MiB. Default off; set
	// INDEX_HUGEPAGE=1 when running with a larger memory budget.
	if os.Getenv("INDEX_HUGEPAGE") == "1" {
		_ = madviseHugepage(data)
	}

	hdr := Header{
		Magic:   binary.LittleEndian.Uint32(data[0:4]),
		Version: binary.LittleEndian.Uint32(data[4:8]),
		K:       binary.LittleEndian.Uint32(data[8:12]),
		N:       binary.LittleEndian.Uint32(data[12:16]),
		NBlocks: binary.LittleEndian.Uint32(data[16:20]),
	}
	hdr.Scale = float32FromBits(binary.LittleEndian.Uint32(data[20:24]))
	if hdr.Magic != Magic {
		return nil, fmt.Errorf("bad magic 0x%x", hdr.Magic)
	}
	if hdr.Version != Version {
		return nil, fmt.Errorf("bad version %d", hdr.Version)
	}

	k := int(hdr.K)
	n := int(hdr.N)
	nb := int(hdr.NBlocks)

	cur := HeaderSize
	ncb := (k + LANES - 1) / LANES

	centroids := i16Slice(data[cur:], ncb*PADDED_DIMS*LANES)
	cur += ncb * BlockBytes

	bboxMin := i16Slice(data[cur:], k*PADDED_DIMS)
	cur += k * PADDED_DIMS * 2

	bboxMax := i16Slice(data[cur:], k*PADDED_DIMS)
	cur += k * PADDED_DIMS * 2

	blockOffsets := u32Slice(data[cur:], k+1)
	cur += (k + 1) * 4

	counts := u32Slice(data[cur:], k)
	cur += k * 4

	vectors := i16Slice(data[cur:], nb*PADDED_DIMS*LANES)
	cur += nb * BlockBytes

	labels := data[cur : cur+nb*LANES : cur+nb*LANES]

	idx := &Index{
		mmap:         data,
		Hdr:          hdr,
		Centroids:    centroids,
		NCentroidBlk: ncb,
		BBoxMin:      bboxMin,
		BBoxMax:      bboxMax,
		BlockOffsets: blockOffsets,
		Counts:       counts,
		Vectors:      vectors,
		Labels:       labels,
		K:            k,
		N:            n,
		NBlocks:      nb,
	}
	var acc byte
	for i := 0; i < len(data); i += 4096 {
		acc ^= data[i]
	}
	_ = acc
	return idx, nil
}

func (i *Index) Close() error {
	if i.mmap != nil {
		err := unix.Munmap(i.mmap)
		i.mmap = nil
		return err
	}
	return nil
}

func float32FromBits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}

// madviseHugepage calls madvise(MADV_HUGEPAGE) directly. golang.org/x/sys/unix
// does not export this constant on older versions.
func madviseHugepage(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	const MADV_HUGEPAGE = 14
	_, _, errno := unix.Syscall(unix.SYS_MADVISE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(MADV_HUGEPAGE))
	if errno != 0 {
		return errno
	}
	return nil
}

func i16Slice(buf []byte, n int) []int16 {
	if n == 0 {
		return nil
	}
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&buf))
	out := struct {
		Data uintptr
		Len  int
		Cap  int
	}{Data: hdr.Data, Len: n, Cap: n}
	return *(*[]int16)(unsafe.Pointer(&out))
}

func u32Slice(buf []byte, n int) []uint32 {
	if n == 0 {
		return nil
	}
	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&buf))
	out := struct {
		Data uintptr
		Len  int
		Cap  int
	}{Data: hdr.Data, Len: n, Cap: n}
	return *(*[]uint32)(unsafe.Pointer(&out))
}
