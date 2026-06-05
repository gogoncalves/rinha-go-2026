package jsonp

import (
	"errors"
	"strconv"
)

type Payload struct {
	Amount            float32
	Installments      uint32
	RequestedAt       []byte
	AvgAmount         float32
	TxCount24h        uint32
	KnownMerchants    [16][]byte
	KnownN            uint32
	MerchantID        []byte
	MCC               []byte
	MerchantAvgAmount float32
	IsOnline          bool
	CardPresent       bool
	KmFromHome        float32
	HasLast           bool
	LastTimestamp     []byte
	LastKmFromCurrent float32
}

var ErrMalformed = errors.New("malformed")

func peek(buf []byte, i int) int {
	if i < len(buf) {
		return int(buf[i])
	}
	return -1
}

func skipWs(buf []byte, i *int) {
	for *i < len(buf) {
		c := buf[*i]
		if c != ' ' && c != '\t' && c != '\r' && c != '\n' {
			return
		}
		*i++
	}
}

func expect(buf []byte, i *int, c byte) error {
	skipWs(buf, i)
	if *i >= len(buf) || buf[*i] != c {
		return ErrMalformed
	}
	*i++
	return nil
}

func eq(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for j := 0; j < len(a); j++ {
		if a[j] != b[j] {
			return false
		}
	}
	return true
}

func readKey(buf []byte, i *int) ([]byte, error) {
	skipWs(buf, i)
	if *i >= len(buf) || buf[*i] != '"' {
		return nil, ErrMalformed
	}
	*i++
	start := *i
	for *i < len(buf) && buf[*i] != '"' {
		*i++
	}
	if *i >= len(buf) {
		return nil, ErrMalformed
	}
	k := buf[start:*i]
	*i++
	return k, nil
}

func readStr(buf []byte, i *int) ([]byte, error) {
	skipWs(buf, i)
	if *i >= len(buf) || buf[*i] != '"' {
		return nil, ErrMalformed
	}
	*i++
	start := *i
	for *i < len(buf) && buf[*i] != '"' {
		*i++
	}
	if *i >= len(buf) {
		return nil, ErrMalformed
	}
	s := buf[start:*i]
	*i++
	return s, nil
}

func readBool(buf []byte, i *int) (bool, error) {
	skipWs(buf, i)
	if *i+4 <= len(buf) && string(buf[*i:*i+4]) == "true" {
		*i += 4
		return true, nil
	}
	if *i+5 <= len(buf) && string(buf[*i:*i+5]) == "false" {
		*i += 5
		return false, nil
	}
	return false, ErrMalformed
}

func readU32(buf []byte, i *int) (uint32, error) {
	skipWs(buf, i)
	var v uint32
	saw := false
	for *i < len(buf) {
		c := buf[*i]
		if c < '0' || c > '9' {
			break
		}
		v = v*10 + uint32(c-'0')
		saw = true
		*i++
	}
	if !saw {
		return 0, ErrMalformed
	}
	return v, nil
}

func readF32(buf []byte, i *int) (float32, error) {
	skipWs(buf, i)
	start := *i
	if *i < len(buf) && (buf[*i] == '-' || buf[*i] == '+') {
		*i++
	}
	hasExp := false
	for *i < len(buf) {
		c := buf[*i]
		if (c >= '0' && c <= '9') || c == '.' {
			*i++
		} else if c == 'e' || c == 'E' {
			hasExp = true
			*i++
		} else if (c == '+' || c == '-') && hasExp {
			*i++
		} else {
			break
		}
	}
	if *i == start {
		return 0, ErrMalformed
	}
	s := buf[start:*i]
	if hasExp {
		f, err := strconv.ParseFloat(string(s), 32)
		if err != nil {
			return 0, ErrMalformed
		}
		return float32(f), nil
	}
	neg := false
	idx := 0
	if len(s) > 0 && (s[0] == '-' || s[0] == '+') {
		neg = s[0] == '-'
		idx = 1
	}
	var intPart uint64
	for idx < len(s) && s[idx] != '.' {
		intPart = intPart*10 + uint64(s[idx]-'0')
		idx++
	}
	var fracPart uint64
	var fracPow uint64 = 1
	if idx < len(s) && s[idx] == '.' {
		idx++
		for idx < len(s) {
			fracPart = fracPart*10 + uint64(s[idx]-'0')
			fracPow *= 10
			idx++
		}
	}
	v := float32(intPart) + float32(fracPart)/float32(fracPow)
	if neg {
		v = -v
	}
	return v, nil
}

func skipVal(buf []byte, i *int) {
	skipWs(buf, i)
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
		var closer byte
		if c == '{' {
			closer = '}'
		} else {
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

func Parse(buf []byte, p *Payload) error {
	*p = Payload{}
	i := 0
	if err := expect(buf, &i, '{'); err != nil {
		return err
	}
	for {
		skipWs(buf, &i)
		if peek(buf, i) == '}' {
			i++
			break
		}
		k, err := readKey(buf, &i)
		if err != nil {
			return err
		}
		if err := expect(buf, &i, ':'); err != nil {
			return err
		}
		skipWs(buf, &i)
		switch {
		case eq(k, []byte("transaction")):
			if err := readTx(buf, &i, p); err != nil {
				return err
			}
		case eq(k, []byte("customer")):
			if err := readCust(buf, &i, p); err != nil {
				return err
			}
		case eq(k, []byte("merchant")):
			if err := readMer(buf, &i, p); err != nil {
				return err
			}
		case eq(k, []byte("terminal")):
			if err := readTerm(buf, &i, p); err != nil {
				return err
			}
		case eq(k, []byte("last_transaction")):
			if err := readLast(buf, &i, p); err != nil {
				return err
			}
		default:
			skipVal(buf, &i)
		}
		skipWs(buf, &i)
		if peek(buf, i) == ',' {
			i++
		}
	}
	return nil
}

func readTx(buf []byte, i *int, p *Payload) error {
	if err := expect(buf, i, '{'); err != nil {
		return err
	}
	for {
		skipWs(buf, i)
		if peek(buf, *i) == '}' {
			*i++
			return nil
		}
		k, err := readKey(buf, i)
		if err != nil {
			return err
		}
		if err := expect(buf, i, ':'); err != nil {
			return err
		}
		skipWs(buf, i)
		switch {
		case eq(k, []byte("amount")):
			v, err := readF32(buf, i)
			if err != nil {
				return err
			}
			p.Amount = v
		case eq(k, []byte("installments")):
			v, err := readU32(buf, i)
			if err != nil {
				return err
			}
			p.Installments = v
		case eq(k, []byte("requested_at")):
			v, err := readStr(buf, i)
			if err != nil {
				return err
			}
			p.RequestedAt = v
		default:
			skipVal(buf, i)
		}
		skipWs(buf, i)
		if peek(buf, *i) == ',' {
			*i++
		}
	}
}

func readCust(buf []byte, i *int, p *Payload) error {
	if err := expect(buf, i, '{'); err != nil {
		return err
	}
	for {
		skipWs(buf, i)
		if peek(buf, *i) == '}' {
			*i++
			return nil
		}
		k, err := readKey(buf, i)
		if err != nil {
			return err
		}
		if err := expect(buf, i, ':'); err != nil {
			return err
		}
		skipWs(buf, i)
		switch {
		case eq(k, []byte("avg_amount")):
			v, err := readF32(buf, i)
			if err != nil {
				return err
			}
			p.AvgAmount = v
		case eq(k, []byte("tx_count_24h")):
			v, err := readU32(buf, i)
			if err != nil {
				return err
			}
			p.TxCount24h = v
		case eq(k, []byte("known_merchants")):
			if err := readKnown(buf, i, p); err != nil {
				return err
			}
		default:
			skipVal(buf, i)
		}
		skipWs(buf, i)
		if peek(buf, *i) == ',' {
			*i++
		}
	}
}

func readMer(buf []byte, i *int, p *Payload) error {
	if err := expect(buf, i, '{'); err != nil {
		return err
	}
	for {
		skipWs(buf, i)
		if peek(buf, *i) == '}' {
			*i++
			return nil
		}
		k, err := readKey(buf, i)
		if err != nil {
			return err
		}
		if err := expect(buf, i, ':'); err != nil {
			return err
		}
		skipWs(buf, i)
		switch {
		case eq(k, []byte("id")):
			v, err := readStr(buf, i)
			if err != nil {
				return err
			}
			p.MerchantID = v
		case eq(k, []byte("mcc")):
			v, err := readStr(buf, i)
			if err != nil {
				return err
			}
			p.MCC = v
		case eq(k, []byte("avg_amount")):
			v, err := readF32(buf, i)
			if err != nil {
				return err
			}
			p.MerchantAvgAmount = v
		default:
			skipVal(buf, i)
		}
		skipWs(buf, i)
		if peek(buf, *i) == ',' {
			*i++
		}
	}
}

func readTerm(buf []byte, i *int, p *Payload) error {
	if err := expect(buf, i, '{'); err != nil {
		return err
	}
	for {
		skipWs(buf, i)
		if peek(buf, *i) == '}' {
			*i++
			return nil
		}
		k, err := readKey(buf, i)
		if err != nil {
			return err
		}
		if err := expect(buf, i, ':'); err != nil {
			return err
		}
		skipWs(buf, i)
		switch {
		case eq(k, []byte("is_online")):
			v, err := readBool(buf, i)
			if err != nil {
				return err
			}
			p.IsOnline = v
		case eq(k, []byte("card_present")):
			v, err := readBool(buf, i)
			if err != nil {
				return err
			}
			p.CardPresent = v
		case eq(k, []byte("km_from_home")):
			v, err := readF32(buf, i)
			if err != nil {
				return err
			}
			p.KmFromHome = v
		default:
			skipVal(buf, i)
		}
		skipWs(buf, i)
		if peek(buf, *i) == ',' {
			*i++
		}
	}
}

func readLast(buf []byte, i *int, p *Payload) error {
	if *i+4 <= len(buf) && string(buf[*i:*i+4]) == "null" {
		*i += 4
		p.HasLast = false
		return nil
	}
	if err := expect(buf, i, '{'); err != nil {
		return err
	}
	p.HasLast = true
	for {
		skipWs(buf, i)
		if peek(buf, *i) == '}' {
			*i++
			return nil
		}
		k, err := readKey(buf, i)
		if err != nil {
			return err
		}
		if err := expect(buf, i, ':'); err != nil {
			return err
		}
		skipWs(buf, i)
		switch {
		case eq(k, []byte("timestamp")):
			v, err := readStr(buf, i)
			if err != nil {
				return err
			}
			p.LastTimestamp = v
		case eq(k, []byte("km_from_current")):
			v, err := readF32(buf, i)
			if err != nil {
				return err
			}
			p.LastKmFromCurrent = v
		default:
			skipVal(buf, i)
		}
		skipWs(buf, i)
		if peek(buf, *i) == ',' {
			*i++
		}
	}
}

func readKnown(buf []byte, i *int, p *Payload) error {
	if err := expect(buf, i, '['); err != nil {
		return err
	}
	p.KnownN = 0
	for {
		skipWs(buf, i)
		if peek(buf, *i) == ']' {
			*i++
			return nil
		}
		s, err := readStr(buf, i)
		if err != nil {
			return err
		}
		if p.KnownN < uint32(len(p.KnownMerchants)) {
			p.KnownMerchants[p.KnownN] = s
			p.KnownN++
		}
		skipWs(buf, i)
		if peek(buf, *i) == ',' {
			*i++
		}
	}
}
