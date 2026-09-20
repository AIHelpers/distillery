.PHONY: run run-local build docker-build docker-run test

build:
	go build -o distillery ./cmd/server

# Default: simulation backend (no GPU/Python needed). GGUF downloads are
# 1KB structural stubs only — there are no real trained weights in this mode.
run: build
	./distillery

# Real fine-tuning + real GGUF conversion (incl. f16). Requires:
#   - NVIDIA GPU (or TRAINING_CPU_FALLBACK=true)
#   - Python deps installed: pip install -r trainer/requirements.txt
# GGUF downloads then run trainer/gguf.py and produce full-size model files.
# MODEL_CACHE_DIR (optional) shares the HF base-model cache between training
# and GGUF conversion so weights are downloaded only once.
run-local: build
	TRAINING_BACKEND=local MODEL_CACHE_DIR=./data/model-cache ./distillery

docker-build:
	docker build -t distillery:latest .

docker-run:
	docker run --rm -p 8080:8080 -v distillery-data:/data distillery:latest

test:
	go vet ./...
	go build ./...
