# Active Listener — project tasks
# Install just: https://github.com/casey/just

# List available tasks
_:
    @just --list

# Build binary without cross compilation (output: target/release/active-listener)
build:
    cargo build --release

# macOS aarch64/x86_64 (Metal), Linux x86_64, Windows x86_64 GNU (CPU via cross for non-macOS). Needs Docker; see scripts/build-cross.sh.
build-cross:
    ./scripts/build-cross.sh

# List cargo lock files under target (remove manually if a build is stuck)
build-unblock:
    find target -name '.cargo-lock' 2>/dev/null

# Build and install to ~/.local/bin; configure shell integration
install: build
    ./target/release/active-listener install


# Create a GitHub release with cross-compiled binary tarballs (requires gh)
release: build-cross
    ./scripts/release.sh

# Smoke test the release binary with a sample WAV file
smoke-test: build
    ./target/release/active-listener process ./tests/2026-04-08_165818.wav --dir .