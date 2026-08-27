#!/usr/bin/env bash
# Build release wheels locally.
#
# Thanks to abi3 there is one wheel per platform rather than one per Python version: a
# `cp39-abi3` wheel runs on CPython 3.9 and everything after it.
#
#   ./scripts/build-wheels.sh              # host wheel + linux x86_64
#   ./scripts/build-wheels.sh linux        # linux x86_64 only
#   ./scripts/build-wheels.sh linux-arm    # linux aarch64 (emulated, slow)
#   ./scripts/build-wheels.sh host         # this machine only
#
# Linux wheels are built in the official maturin container, which links against an old glibc
# so the result is manylinux-compatible. Docker must be running.
#
# CI (.github/workflows/ci.yml) builds the same set on every push; this script is for trying
# things out before pushing.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
IMAGE="ghcr.io/pyo3/maturin:latest"
what="${1:-all}"

# Docker needs a Windows-style path for the bind mount when run from Git Bash.
host_path() {
    if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

# The container and the host compile for different targets but would share ./target, where
# each writes over the other's build of the host artifacts. Keep them apart.
docker_build() {
    local platform=$1 target_dir=$2 label=$3
    echo "==> $label"
    # Git Bash rewrites arguments that look like Unix paths into Windows ones, which would
    # mangle the container-side paths below. Both variables are ignored elsewhere.
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
    docker run --rm \
        --platform "$platform" \
        -v "$(host_path "$REPO"):/io" \
        -w /io \
        -e CARGO_TARGET_DIR="/io/$target_dir" \
        "$IMAGE" build --release --out dist
}

build_host() {
    echo "==> host ($(uname -s))"
    # `maturin develop` leaves a compiled extension in the source tree; packaging it alongside
    # the freshly built one would put two copies in the wheel.
    rm -f python/bpe_continue/*.pyd python/bpe_continue/*.so python/bpe_continue/*.dylib
    maturin build --release --out dist
}

case "$what" in
    host)      build_host ;;
    linux)     docker_build linux/amd64 target-linux-amd64 "linux x86_64 (manylinux)" ;;
    linux-arm) docker_build linux/arm64 target-linux-arm64 "linux aarch64 (manylinux, emulated)" ;;
    all)
        build_host
        docker_build linux/amd64 target-linux-amd64 "linux x86_64 (manylinux)"
        ;;
    *)
        echo "usage: $0 [all|host|linux|linux-arm]" >&2
        exit 2
        ;;
esac

echo
echo "==> dist/"
ls -lh dist/
