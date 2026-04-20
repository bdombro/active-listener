# Active Listener — project tasks
# Install just: https://github.com/casey/just

# List available tasks
_:
    @just --list

# Bump crate version: `just version-bump patch` (or major | minor)
version-bump kind:
    ./scripts/version-bump.sh {{kind}}

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


# Bump version, cross-build, then create a GitHub release with binary tarballs (requires gh)
release kind: (version-bump kind) build-cross
    ./scripts/release.sh

# Smoke test the release binary with a sample WAV file
test-smoke: build
    ./target/release/active-listener process --diarize ./tests/2026-04-08_165818.wav --dir tests
    ./target/release/active-listener process --diarize ./tests/2026-04-20_102511.wav --dir tests