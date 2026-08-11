# --- Build stage ---
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
COPY web ./web
# CGO disabled -> static binary, no external deps (pure Go stdlib).
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/distillery ./cmd/server

# --- Runtime stage ---
FROM alpine:3.20
RUN adduser -D -H -u 10001 distillery && mkdir -p /data && chown distillery /data
COPY --from=build /out/distillery /usr/local/bin/distillery
USER distillery
ENV PORT=8080
ENV DATA_PATH=/data/distillery.json
ENV OPEN_BROWSER=false
EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/distillery"]
