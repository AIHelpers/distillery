.PHONY: run build docker-build docker-run test

build:
	go build -o distillery ./cmd/server

run: build
	./distillery

docker-build:
	docker build -t distillery:latest .

docker-run:
	docker run --rm -p 8080:8080 -v distillery-data:/data distillery:latest

test:
	go vet ./...
	go build ./...
