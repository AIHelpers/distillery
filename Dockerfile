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
FROM python:3.11-slim AS runtime

# Install system deps + the trainer package with the GGUF extra so
# `python -m trainer.gguf` works for on-demand GGUF conversion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --no-create-home --uid 10001 distillery \
    && mkdir -p /data && chown distillery /data

WORKDIR /srv

# Copy trainer source + build config.
COPY trainer ./trainer

# Install the trainer package + its gguf extra (pure-Python gguf package).
RUN pip install --no-cache-dir "./trainer[gguf]"

# Copy the Go binary.
COPY --from=build /out/distillery /usr/local/bin/distillery

USER distillery
ENV PORT=8080
ENV DATA_PATH=/data/distillery.json
ENV OPEN_BROWSER=false
# Let the Go server invoke `python -m trainer.gguf` / `python -m trainer.run`.
ENV PYTHONPATH=/srv
ENV TRAINING_BACKEND=local
EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/distillery"]