package timex

type Stamp struct {
	Year   int32
	Month  uint32
	Day    uint32
	Hour   uint32
	Minute uint32
	Second uint32
}

func Parse(s []byte) Stamp {
	if len(s) < 19 {
		return Stamp{}
	}
	return Stamp{
		Year:   int32(s[0]-'0')*1000 + int32(s[1]-'0')*100 + int32(s[2]-'0')*10 + int32(s[3]-'0'),
		Month:  uint32(s[5]-'0')*10 + uint32(s[6]-'0'),
		Day:    uint32(s[8]-'0')*10 + uint32(s[9]-'0'),
		Hour:   uint32(s[11]-'0')*10 + uint32(s[12]-'0'),
		Minute: uint32(s[14]-'0')*10 + uint32(s[15]-'0'),
		Second: uint32(s[17]-'0')*10 + uint32(s[18]-'0'),
	}
}

func DayOfWeek(year int32, month, day uint32) uint32 {
	t := [12]int32{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	y := year
	if month < 3 {
		y -= 1
	}
	raw := (y + floorDiv(y, 4) - floorDiv(y, 100) + floorDiv(y, 400) + t[month-1] + int32(day)) % 7
	s := uint32(((raw % 7) + 7) % 7)
	return (s + 6) % 7
}

func floorDiv(a, b int32) int32 {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q -= 1
	}
	return q
}

func DaysSinceEpoch(year int32, month, day uint32) int64 {
	y := int64(year)
	m := int64(month)
	if m <= 2 {
		y -= 1
	}
	var era int64
	if y >= 0 {
		era = y / 400
	} else {
		era = (y - 399) / 400
	}
	yoe := y - era*400
	var mShift int64
	if m > 2 {
		mShift = m - 3
	} else {
		mShift = m + 9
	}
	doy := (153*mShift+2)/5 + int64(day) - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return era*146097 + doe - 719468
}

func EpochSeconds(s Stamp) int64 {
	return DaysSinceEpoch(s.Year, s.Month, s.Day)*86400 +
		int64(s.Hour)*3600 + int64(s.Minute)*60 + int64(s.Second)
}
