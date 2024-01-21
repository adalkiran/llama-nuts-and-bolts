#!/bin/bash

mkdir -p output

echo "$PWD/cmd/main.go"

for GOOS in darwin linux windows; do
  for GOARCH in 386 amd64 arm64; do
    echo "Building $GOOS-$GOARCH"
    export GOOS=$GOOS
    export GOARCH=$GOARCH
    go build -o $PWD/output/llama-nb-$GOOS-$GOARCH $PWD/cmd/main.go
  done
done