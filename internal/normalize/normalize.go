package normalize

import (
	"rinha-go/internal/jsonp"
	"rinha-go/internal/timex"
)

const (
	DIMS        = 14
	PADDED_DIMS = 16

	MaxAmount             = 10000.0
	MaxInstallments       = 12.0
	AmountVsAvgRatio      = 10.0
	MaxMinutes            = 1440.0
	MaxKm                 = 1000.0
	MaxTxCount24h         = 20.0
	MaxMerchantAvgAmount  = 10000.0
)

func clamp01(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func mccRisk(mcc []byte) float32 {
	if len(mcc) != 4 {
		return 0.5
	}
	x := pack(mcc)
	switch x {
	case packLit('5', '4', '1', '1'):
		return 0.15
	case packLit('5', '8', '1', '2'):
		return 0.30
	case packLit('5', '9', '1', '2'):
		return 0.20
	case packLit('5', '9', '4', '4'):
		return 0.45
	case packLit('7', '8', '0', '1'):
		return 0.80
	case packLit('7', '8', '0', '2'):
		return 0.75
	case packLit('7', '9', '9', '5'):
		return 0.85
	case packLit('4', '5', '1', '1'):
		return 0.35
	case packLit('5', '3', '1', '1'):
		return 0.25
	case packLit('5', '9', '9', '9'):
		return 0.50
	}
	return 0.5
}

func pack(s []byte) uint32 {
	return (uint32(s[0]) << 24) | (uint32(s[1]) << 16) | (uint32(s[2]) << 8) | uint32(s[3])
}

func packLit(a, b, c, d byte) uint32 {
	return (uint32(a) << 24) | (uint32(b) << 16) | (uint32(c) << 8) | uint32(d)
}

func Vectorize(p *jsonp.Payload) [DIMS]float32 {
	ts := timex.Parse(p.RequestedAt)
	cur := timex.EpochSeconds(ts)
	dow := timex.DayOfWeek(ts.Year, ts.Month, ts.Day)

	known := false
	for i := 0; i < int(p.KnownN); i++ {
		m := p.KnownMerchants[i]
		if len(m) == len(p.MerchantID) {
			same := true
			for j := 0; j < len(m); j++ {
				if m[j] != p.MerchantID[j] {
					same = false
					break
				}
			}
			if same {
				known = true
				break
			}
		}
	}

	var d5, d6 float32 = -1.0, -1.0
	if p.HasLast {
		lts := timex.Parse(p.LastTimestamp)
		last := timex.EpochSeconds(lts)
		minsRaw := float32(cur-last) / 60.0
		mins := minsRaw
		if mins < 0 {
			mins = 0
		}
		d5 = clamp01(mins / MaxMinutes)
		d6 = clamp01(p.LastKmFromCurrent / MaxKm)
	}

	var avgAmount float32 = p.AvgAmount
	if avgAmount == 0 {
		avgAmount = 1
	}

	knownVal := float32(1.0)
	if known {
		knownVal = 0.0
	}
	online := float32(0)
	if p.IsOnline {
		online = 1
	}
	cardPresent := float32(0)
	if p.CardPresent {
		cardPresent = 1
	}

	return [DIMS]float32{
		clamp01(p.Amount / MaxAmount),
		clamp01(float32(p.Installments) / MaxInstallments),
		clamp01((p.Amount / avgAmount) / AmountVsAvgRatio),
		float32(ts.Hour) / 23.0,
		float32(dow) / 6.0,
		d5,
		d6,
		clamp01(p.KmFromHome / MaxKm),
		clamp01(float32(p.TxCount24h) / MaxTxCount24h),
		online,
		cardPresent,
		knownVal,
		mccRisk(p.MCC),
		clamp01(p.MerchantAvgAmount / MaxMerchantAvgAmount),
	}
}
