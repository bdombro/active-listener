#!/usr/bin/env bash
# Build release binaries for aarch64-apple-darwin, x86_64-apple-darwin, x86_64-unknown-linux-gnu,
# and x86_64-pc-windows-gnu (CPU-only for non-macOS targets; see just build-cross).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ensure_cargo_path() {
  export PATH="${HOME}/.cargo/bin:${PATH}"
}

require_rustup() {
  ensure_cargo_path
  if ! command -v rustup &>/dev/null; then
    echo "error: rustup not in PATH (expected at ~/.cargo/bin/rustup). Install: https://rustup.rs"
    exit 1
  fi
}

require_cargo() {
  ensure_cargo_path
  if ! command -v cargo &>/dev/null; then
    echo "error: cargo not in PATH (expected with rustup)."
    exit 1
  fi
}

# crates.io cross 0.2.5 breaks Linux-from-Apple-Silicon; fixed on git main.
# https://github.com/cross-rs/cross/issues/1649
ensure_cross() {
  ensure_cargo_path
  local cross_line=""
  if command -v cross &>/dev/null; then
    cross_line=$(cross --version 2>/dev/null | head -1 || true)
  fi
  if ! command -v cross &>/dev/null; then
    echo "Installing cross from git (fixes Apple Silicon -> Linux; cross-rs/cross#1649)..."
    cargo install cross --git https://github.com/cross-rs/cross --force
  elif [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 && "${cross_line}" == "cross 0.2.5" ]]; then
    echo "Replacing crates.io cross 0.2.5 with git (required for Linux targets on Apple Silicon; cross-rs/cross#1649)..."
    cargo install cross --git https://github.com/cross-rs/cross --force
  fi
}

precheck() {
  require_cargo
  if ! command -v docker &>/dev/null; then
    echo "error: docker not found. Install Docker Desktop (macOS) or your distro's docker package."
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "error: Docker is not running. Start Docker Desktop (or \`sudo service docker start\` on Linux), then retry."
    exit 1
  fi
  ensure_cross
}

build_macos_aarch64() {
  require_rustup
  rustup target add aarch64-apple-darwin
  echo "Building aarch64-apple-darwin..."
  cargo build --release --target aarch64-apple-darwin
  echo "Done: target/aarch64-apple-darwin/release/active-listener"
}

build_macos_x86_64() {
  require_rustup
  rustup target add x86_64-apple-darwin
  echo "Building x86_64-apple-darwin..."
  cargo build --release --target x86_64-apple-darwin
  echo "Done: target/x86_64-apple-darwin/release/active-listener"
}

build_linux() {
  ensure_cargo_path
  rustup target add x86_64-unknown-linux-gnu
  echo "Building x86_64-unknown-linux-gnu..."
  cross build --release --target x86_64-unknown-linux-gnu --no-default-features
  echo "Done: target/x86_64-unknown-linux-gnu/release/active-listener"
}

build_windows() {
  ensure_cargo_path
  rustup target add x86_64-pc-windows-gnu
  echo "Building x86_64-pc-windows-gnu..."
  cross build --release --target x86_64-pc-windows-gnu --no-default-features
  echo "Done: target/x86_64-pc-windows-gnu/release/active-listener.exe"
}

precheck
build_macos_aarch64
build_macos_x86_64
build_linux
build_windows
