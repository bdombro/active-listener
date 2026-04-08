#!/usr/bin/env bash
# Create GitHub release with cross-compiled binary tarballs (requires gh, run after build-cross).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.cargo/bin:${PATH}"

VERSION=$(cargo metadata --no-deps --format-version 1 | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['packages'][0]['version'])")
echo "Releasing v${VERSION}..."

TARGETS=(
  "aarch64-apple-darwin"
  "x86_64-apple-darwin"
  "x86_64-unknown-linux-gnu"
)

ARTIFACTS=()
for TARGET in "${TARGETS[@]}"; do
  TARBALL="active-listener-${VERSION}-${TARGET}.tar.gz"
  tar -czf "${TARBALL}" -C "target/${TARGET}/release" active-listener
  ARTIFACTS+=("${TARBALL}")
  echo "Packaged: ${TARBALL}"
done

gh release create "v${VERSION}" \
  --title "v${VERSION}" \
  --generate-notes \
  "${ARTIFACTS[@]}"

rm -f "${ARTIFACTS[@]}"
echo "Released v${VERSION}"
