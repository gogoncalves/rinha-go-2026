FROM --platform=linux/amd64 golang:1.22-alpine AS builder

RUN apk add --no-cache curl gzip ca-certificates

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY internal ./internal
COPY cmd ./cmd

ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3

RUN go build -ldflags="-s -w" -gcflags="-l=4 -B" -o /out/api ./cmd/api && \
    go build -ldflags="-s -w" -gcflags="-l=4 -B" -o /out/lb ./cmd/lb && \
    go build -ldflags="-s -w" -gcflags="-l=4 -B" -o /out/buildindex ./cmd/buildindex

ARG REFERENCES_URL=https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz
RUN curl -fsSL -o /tmp/refs.json.gz "${REFERENCES_URL}" \
 && gunzip /tmp/refs.json.gz \
 && INPUT=/tmp/refs.json OUTPUT=/index.bin /out/buildindex \
 && rm -f /tmp/refs.json /out/buildindex

FROM --platform=linux/amd64 alpine:3.20
COPY --from=builder /out/api /api
COPY --from=builder /out/lb /lb
COPY --from=builder /index.bin /data/index.bin
ENV INDEX_PATH=/data/index.bin
EXPOSE 9999
ENTRYPOINT ["/api"]
