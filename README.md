# Active Listener

Record and transcribe meetings using a CLI, fully local. Press Ctrl+C to stop — you get a WAV and a Whisper-transcribed markdown file. No cloud, no subscriptions.

## Quick start

Download the binary for your platform from the [releases page](https://github.com/bdombro/active-listener/releases):

| Platform | File |
|---|---|
| macOS Apple Silicon | `active-listener-*-aarch64-apple-darwin.tar.gz` |
| macOS Intel | `active-listener-*-x86_64-apple-darwin.tar.gz` |
| Linux x86_64 | `active-listener-*-x86_64-unknown-linux-gnu.tar.gz` |
| Windows x86_64 | `active-listener-*-x86_64-pc-windows-gnu.tar.gz` |

Extract and install (macOS / Linux):

```bash
tar -xzf active-listener-*.tar.gz
./active-listener install
source ~/.zshrc
```

`install` copies the binary to `~/.local/bin`, adds zsh completions, and pre-downloads the default Whisper weights (`small`) from Hugging Face. Diarization models are also pre-fetched so your first `--diarize` run starts immediately.

> **Windows**: `install` is not supported — place `active-listener.exe` on your `PATH` manually and run it directly. Note that the Windows release binary is built without diarization support, so `--diarize` has no effect.

**To build from source** (requires Rust 1.74+ and [just](https://github.com/casey/just)):

```bash
just install
```

## Usage

```bash
# Interactive: choose mic, system audio, or both (TTY only)
active-listener start

# Record mic only; write to current directory
active-listener start --mic

# Record mic + desktop audio (requires Screen Recording permission on macOS)
active-listener start --mic --system-audio

# Custom output folder and filename stem
active-listener start --mic --dir ~/notes --name standup

# Speaker labels in the output (adds ~10–30s post-processing)
active-listener start --mic --diarize

# Transcribe an existing WAV (no re-record)
active-listener process recording.wav --dir ~/notes
active-listener process recording.wav --diarize
```

## Output

Every session produces two files with the same basename:

- **`YYYY-MM-DD_HHMMSS.wav`** — raw 16 kHz mono recording
- **`YYYY-MM-DD_HHMMSS.md`** — frontmatter + transcript

### Default (no `--diarize`)

```markdown
---
date: 2026-04-08T14:30:00
duration: 32m 15s
whisper_model: small
---

# Meeting notes — 2026-04-08 14:30

## Transcript

**[00:00]** Welcome everyone, let's get started with today's standup.

**[00:12]** I finished the auth refactor yesterday and pushed it for review.

**[00:28]** Looks good to me, I'll take a look after this call.

**[01:04]** Any blockers? Nothing from my side, shipping the feature branch today.
```

### With `--diarize`

Whisper and the speaker diarizer run in parallel; labels are merged into the transcript by timestamp overlap.

```markdown
---
date: 2026-04-08T14:30:00
duration: 32m 15s
whisper_model: small
---

# Meeting notes — 2026-04-08 14:30

## Transcript

**Speaker 1**

**[00:00]** Welcome everyone, let's get started with today's standup.

**Speaker 2**

**[00:12]** I finished the auth refactor yesterday and pushed it for review.

**Speaker 1**

**[00:28]** Looks good to me, I'll take a look after this call.

**Speaker 2**

**[01:04]** No blockers, shipping the feature branch today.
```

Labels are `Speaker 1`, `Speaker 2`, … — not real names.

## Options

All flags work on both `start` and `process` unless noted.

| Flag | Default | Description |
|---|---|---|
| `--mic` | off | Capture microphone (`start` only; at least one source required) |
| `--system-audio` | off | Mix in desktop audio (`start` only) |
| `--dir PATH` | `.` | Output directory |
| `--name STEM` | datetime | Output filename without extension |
| `--whisper-model` | `small` | `tiny` / `base` / `small` / `medium` / `large` |
| `--diarize` | off | Add speaker labels via sherpa-onnx |
| `--num-speakers N` | auto | Fix speaker count (more reliable than threshold mode) |
| `--diarize-threshold` | `0.55` | Cluster merge/split — higher → fewer speakers |
| `--diarize-embedding PATH` | auto | Custom speaker embedding ONNX (see below) |
| `--duration SECS` | unlimited | Auto-stop after N seconds (`start` only) |
| `--device NAME` | default | Mic device name — see `--list-devices` |
| `--cpu` | off | Force CPU; skip Metal |
| `--verbose` | off | Print device and model details |

`ACTIVE_LISTENER_DIARIZE_EMBEDDING` is the env-var equivalent of `--diarize-embedding`.

## Speaker diarization

Powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (pyannote segmentation + NeMo Titanet small embedding). Models are downloaded once to `~/.cache/active-listener/diarize/` on first use (or pre-fetched by `install`).

Offline diarization is fuzzy — one threshold cannot perfectly split every conversation. Practical tips:

- **`--num-speakers N`** is the most reliable option when you know the count (two-person call, panel of three, etc.).
- **`--diarize-threshold`** (default `0.55`) is the fallback knob: higher merges more aggressively (fewer speakers), lower splits more.
- **`--diarize-embedding`** — swap in another sherpa-onnx embedding ONNX if you want to experiment (e.g. [Titanet large](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) for better accuracy at the cost of size and speed).

Expect imperfect results in difficult audio (crosstalk, TV in the background, single mixed channel).

## Privacy

- Everything runs locally. No audio, transcript, or metadata leaves your machine.
- Whisper weights are downloaded from Hugging Face once (model files only; no audio uploaded).
- Diarization models are downloaded from GitHub releases once (no audio uploaded).
- `active-listener uninstall` removes the binary, shell snippets from `~/.zshrc`, and all cached `openai/whisper-*` Hugging Face hub folders.
- The `.wav` and `.md` files contain raw audio and full transcripts — treat them accordingly.
- macOS requires **Microphone** permission for `--mic` and **Screen Recording** for `--system-audio`.

## Platform notes

| Platform | Audio capture | Diarization |
|---|---|---|
| macOS | Mic (cpal) + ScreenCaptureKit system audio | Yes (`diarize` feature, default) |
| Linux | Mic (cpal) + PipeWire (`linux-system-audio` feature, default) | Yes (native build); No (cross-compiled `--no-default-features` build) |
| Windows | Mic (cpal) + WASAPI loopback | Yes (native build); No (cross-compiled build) |

Linux native builds need `libpipewire-0.3-dev` and `libspa-0.2-dev`.

## Development

Requires [just](https://github.com/casey/just) (`brew install just`) and Rust 1.74+.

```bash
just build        # cargo build --release (native host)
just install      # build + install binary + shell config
just build-cross  # cross-compile: macOS aarch64/x86_64, Linux x86_64, Windows x86_64 (needs Docker)
just release      # build-cross + GitHub release (needs gh)
```

Quick dev loop: `cargo run -- start --mic` — records, transcribes, writes WAV + markdown.

To strip the diarization dependency for a smaller binary:
```bash
cargo build --release --no-default-features --features metal,linux-system-audio
```

## Requirements

- Rust 1.74+
- macOS: Metal is the default GPU backend (`metal` feature). On other platforms build with `--no-default-features` and re-add what you need.

## License

MIT
