//! Active Listener — record meetings; Whisper transcription after each recording.

use active_listener::audio::{
    list_input_devices, read_wav_16k_mono, record_until_stop, write_wav_16k_mono, Pcm16kMono,
    WHISPER_SAMPLE_RATE,
};
use active_listener::DiarizeParams;
use active_listener::markdown::{write_meeting_markdown, MeetingDoc};
use active_listener::system_audio::system_audio_supported;
use active_listener::transcribe::{
    delete_all_openai_whisper_hub_caches, ensure_whisper_artifacts, pick_device,
    transcribe_pcm_samples, WhichModel,
};
#[cfg(feature = "diarize")]
use active_listener::diarize;

use anstyle::{AnsiColor, Color, Style};
use anyhow::{Context, Result};
use clap::{builder::Styles, Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use console::style;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

type DiarizeLabelsJoinHandle = thread::JoinHandle<anyhow::Result<Vec<(f64, f64, String)>>>;

fn clap_styles() -> Styles {
    Styles::styled()
        .header(
            Style::new()
                .bold()
                .fg_color(Some(Color::Ansi(AnsiColor::Yellow))),
        )
        .usage(
            Style::new()
                .bold()
                .fg_color(Some(Color::Ansi(AnsiColor::Yellow))),
        )
        .literal(Style::new().fg_color(Some(Color::Ansi(AnsiColor::Green))))
        .placeholder(Style::new().fg_color(Some(Color::Ansi(AnsiColor::Cyan))))
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum WhisperSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl From<WhisperSize> for WhichModel {
    fn from(s: WhisperSize) -> Self {
        match s {
            WhisperSize::Tiny => WhichModel::Tiny,
            WhisperSize::Base => WhichModel::Base,
            WhisperSize::Small => WhichModel::Small,
            WhisperSize::Medium => WhichModel::Medium,
            WhisperSize::Large => WhichModel::Large,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "active-listener",
    version,
    about = "Record meetings; save WAV, then Whisper markdown.",
    long_about = "Captures microphone and/or system audio. If you omit `--mic` and `--system-audio`, you are prompted interactively (TTY only; otherwise pass flags explicitly). After each recording, a 16 kHz mono WAV is written, then Whisper runs and markdown is saved. Speaker diarization is off unless you pass `--diarize` (requires a build with the `diarize` feature, which is enabled by default).",
    styles = clap_styles(),
    after_help = "EXAMPLES:\n  active-listener start  # interactive source pick (TTY)\n  active-listener start --mic --system-audio --dir .\n  active-listener start --mic --dir ~/notes --name standup\n  active-listener start --mic --diarize  # speaker labels in markdown\n  active-listener process recording.wav --dir .\n  active-listener install\n  active-listener uninstall\n  active-listener completions zsh",
    subcommand_required = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, Args)]
#[command(next_help_heading = "Diarization (sherpa-onnx)")]
struct DiarizeCliArgs {
    /// Expected speaker count when you know it (e.g. `2` for a two-person call). Uses fixed clustering instead of threshold heuristics—often the most stable option.
    #[arg(long)]
    num_speakers: Option<u32>,

    /// When speaker count is unknown: **lower** → more clusters, **higher** → fewer (see sherpa-onnx clustering docs). Ignored if `--num-speakers` is set.
    #[arg(
        long = "diarize-threshold",
        default_value_t = active_listener::DEFAULT_DIARIZE_CLUSTER_THRESHOLD
    )]
    cluster_threshold: f32,

    /// Drop speech shorter than this many seconds before clustering. Slightly **higher** can remove spurious splits; too high clips short words.
    #[arg(
        long = "diarize-min-duration-on",
        default_value_t = active_listener::DEFAULT_DIARIZE_MIN_DURATION_ON
    )]
    min_duration_on: f32,

    /// Merge same-speaker regions when separated by gaps shorter than this many seconds. **Higher** reduces rapid speaker alternation in the output.
    #[arg(
        long = "diarize-min-duration-off",
        default_value_t = active_listener::DEFAULT_DIARIZE_MIN_DURATION_OFF
    )]
    min_duration_off: f32,

    /// Speaker embedding ONNX (16 kHz). Default: auto-download NeMo Titanet small. Override with another sherpa-onnx release model if needed (see README).
    #[arg(long = "diarize-embedding", env = "ACTIVE_LISTENER_DIARIZE_EMBEDDING")]
    embedding_model: Option<PathBuf>,
}

fn validate_diarize_cli(d: &DiarizeCliArgs) -> Result<()> {
    let thr = d.cluster_threshold;
    if !(0.01..=0.99).contains(&thr) {
        anyhow::bail!("--diarize-threshold must be between 0.01 and 0.99 (got {thr})");
    }
    let on = d.min_duration_on;
    if !(0.0..=5.0).contains(&on) {
        anyhow::bail!("--diarize-min-duration-on must be between 0 and 5 seconds (got {on})");
    }
    let off = d.min_duration_off;
    if !(0.0..=10.0).contains(&off) {
        anyhow::bail!("--diarize-min-duration-off must be between 0 and 10 seconds (got {off})");
    }
    Ok(())
}

fn diarize_params_from_cli(d: &DiarizeCliArgs) -> DiarizeParams {
    DiarizeParams {
        num_speakers: d.num_speakers,
        cluster_threshold: d.cluster_threshold,
        min_duration_on: d.min_duration_on,
        min_duration_off: d.min_duration_off,
        embedding_model: d.embedding_model.clone(),
    }
}

/// Record until Ctrl+C; writes WAV then Whisper markdown.
#[derive(Args)]
struct StartArgs {
    /// Output directory (default: current working directory).
    #[arg(long, default_value = ".")]
    dir: PathBuf,

    /// Output filename without extension (default: current datetime).
    #[arg(long)]
    name: Option<String>,

    #[arg(long, value_enum, default_value_t = WhisperSize::Small)]
    whisper_model: WhisperSize,

    /// Stop recording after this many seconds.
    #[arg(long)]
    duration: Option<u64>,

    /// Capture microphone input (combine with `--system-audio` for both).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    mic: bool,

    /// Mix in system/desktop audio when supported (requires Screen Recording on macOS, etc.).
    #[arg(long = "system-audio", action = clap::ArgAction::SetTrue)]
    system_audio: bool,

    #[arg(long, default_value_t = false)]
    list_devices: bool,

    /// Microphone device name (see `--list-devices`; only used with `--mic`).
    #[arg(long)]
    device: Option<String>,

    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Force CPU (no Metal).
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Run speaker diarization in parallel with Whisper (needs a build with the `diarize` feature).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    diarize: bool,

    #[command(flatten)]
    diarize_args: DiarizeCliArgs,
}

/// Transcribe an existing WAV from `start` (16 kHz mono PCM16); writes markdown only.
#[derive(Args)]
struct ProcessArgs {
    /// Path to a WAV file produced by this app's `start` command (16 kHz mono PCM16 LE).
    wav: PathBuf,

    /// Output directory for the markdown file (default: current working directory).
    #[arg(long, default_value = ".")]
    dir: PathBuf,

    /// Output markdown basename without extension (default: input WAV stem).
    #[arg(long)]
    name: Option<String>,

    #[arg(long, value_enum, default_value_t = WhisperSize::Small)]
    whisper_model: WhisperSize,

    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Force CPU (no Metal).
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Run speaker diarization in parallel with Whisper (needs a build with the `diarize` feature).
    #[arg(long, action = clap::ArgAction::SetTrue)]
    diarize: bool,

    #[command(flatten)]
    diarize_args: DiarizeCliArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Record audio until Ctrl+C (WAV + Whisper markdown after each session).
    Start(StartArgs),
    /// Transcribe an existing `start`-format WAV; writes markdown (does not rewrite the WAV).
    Process(ProcessArgs),
    /// Print shell completion script (zsh, bash, or fish).
    Completions {
        #[arg(value_enum)]
        shell: ShellArg,
    },
    /// Install the binary to ~/.local/bin, configure zsh, and pre-download Whisper weights (and sherpa-onnx diarization models when built with `diarize`).
    Install {
        /// Whisper checkpoint to cache from Hugging Face (same as `start --whisper-model`).
        #[arg(long, value_enum, default_value_t = WhisperSize::Small)]
        whisper_model: WhisperSize,
    },
    /// Remove `~/.local/bin/active-listener`, undo install-time `~/.zshrc` snippets, and delete all cached `openai/whisper-*` Hugging Face hub folders.
    Uninstall,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ShellArg {
    Zsh,
    Bash,
    Fish,
}

fn cmd_install(whisper_model: WhisperSize) -> Result<()> {
    use std::io::Write;

    let exe = std::env::current_exe().context("could not determine current executable path")?;
    let home = home_dir().context("could not determine home directory")?;

    let bin_dir = home.join(".local/bin");
    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("create {}", bin_dir.display()))?;

    let dest = bin_dir.join("active-listener");
    std::fs::copy(&exe, &dest)
        .with_context(|| format!("copy {} -> {}", exe.display(), dest.display()))?;
    println!("Installed: {}", dest.display());

    // Ensure ~/.local/bin is in PATH in .zshrc
    let zshrc_path = home.join(".zshrc");
    let path_line = r#"export PATH="$HOME/.local/bin:$PATH""#.to_string();
    let bin_dir_str = bin_dir.to_string_lossy();

    let in_current_path = std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .any(|p| p == bin_dir_str.as_ref());

    let zshrc_content = if zshrc_path.is_file() {
        std::fs::read_to_string(&zshrc_path).unwrap_or_default()
    } else {
        String::new()
    };

    if !in_current_path && !zshrc_content.contains(&path_line) {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&zshrc_path)
            .with_context(|| format!("open {}", zshrc_path.display()))?;
        writeln!(f, "\n# active-listener PATH")?;
        writeln!(f, "{path_line}")?;
        println!("Added PATH export to {}", zshrc_path.display());
    } else {
        println!("PATH already configured.");
    }

    // Add completions to .zshrc if not present
    let completion_line = "source <(active-listener completions zsh)";
    let zshrc_content = if zshrc_path.is_file() {
        std::fs::read_to_string(&zshrc_path).unwrap_or_default()
    } else {
        String::new()
    };
    if !zshrc_content.contains(completion_line) {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&zshrc_path)
            .with_context(|| format!("open {}", zshrc_path.display()))?;
        writeln!(f, "\n# active-listener completions")?;
        writeln!(f, "{completion_line}")?;
        println!("Added completions to {}", zshrc_path.display());
    } else {
        println!("Completions already configured.");
    }

    let which: WhichModel = whisper_model.into();
    println!(
        "\nDownloading Whisper model weights ({whisper_model:?}) from Hugging Face…"
    );
    ensure_whisper_artifacts(which).context("download Whisper model")?;
    println!("Whisper model cached.");

    #[cfg(feature = "diarize")]
    {
        println!(
            "\nDownloading speaker diarization models (sherpa-onnx: segmentation + NeMo Titanet small)…"
        );
        active_listener::diarize::ensure_diarize_models(
            true,
            &DiarizeParams::default(),
        )
        .context("download diarization models")?;
        println!("Diarization models cached.");
    }

    println!("\nDone. Run: source ~/.zshrc");
    Ok(())
}

/// Snippets appended by `install` (must match exactly for a clean undo).
const INSTALL_ZSHRC_PATH_SNIPPET: &str = "\n# active-listener PATH\nexport PATH=\"$HOME/.local/bin:$PATH\"\n";
const INSTALL_ZSHRC_COMPLETIONS_SNIPPET: &str =
    "\n# active-listener completions\nsource <(active-listener completions zsh)\n";

fn cmd_uninstall() -> Result<()> {
    use std::fs;

    let home = home_dir().context("could not determine home directory")?;
    let dest = home.join(".local/bin/active-listener");
    if dest.is_file() {
        fs::remove_file(&dest)
            .with_context(|| format!("remove {}", dest.display()))?;
        println!("Removed {}", dest.display());
    } else {
        println!(
            "Binary not found at {} (already removed or never installed).",
            dest.display()
        );
    }

    let zshrc_path = home.join(".zshrc");
    if zshrc_path.is_file() {
        let content = fs::read_to_string(&zshrc_path)
            .with_context(|| format!("read {}", zshrc_path.display()))?;
        let mut new_content = content.clone();
        new_content = new_content.replace(INSTALL_ZSHRC_PATH_SNIPPET, "");
        new_content = new_content.replace(INSTALL_ZSHRC_COMPLETIONS_SNIPPET, "");
        if new_content != content {
            fs::write(&zshrc_path, new_content)
                .with_context(|| format!("write {}", zshrc_path.display()))?;
            println!(
                "Removed active-listener PATH/completions snippets from {}",
                zshrc_path.display()
            );
        } else {
            println!(
                "No matching active-listener snippets in {} (nothing to change).",
                zshrc_path.display()
            );
        }
    } else {
        println!(
            "{} not found (skipping shell snippet cleanup).",
            zshrc_path.display()
        );
    }

    let n = delete_all_openai_whisper_hub_caches().context("remove OpenAI Whisper hub caches")?;
    if n == 0 {
        println!("No openai/whisper-* Hugging Face hub folders found (nothing to remove).");
    } else {
        println!(
            "Removed {n} openai/whisper-* {} from Hugging Face hub.",
            if n == 1 {
                "cache directory"
            } else {
                "cache directories"
            },
        );
    }

    Ok(())
}

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// When neither `--mic` nor `--system-audio` was passed, ask on an interactive terminal.
fn prompt_audio_sources(cli: &mut StartArgs) -> Result<()> {
    if cli.mic || cli.system_audio {
        return Ok(());
    }

    let stdin = io::stdin();
    if !stdin.is_terminal() {
        anyhow::bail!(
            "Select an audio source: pass `--mic`, `--system-audio`, or both (non-interactive stdin)."
        );
    }

    let sys_ok = system_audio_supported();
    println!(
        "{}",
        style("No audio source given. Choose what to record:").cyan().bold()
    );
    println!("  1) Microphone only");
    if sys_ok {
        println!("  2) System audio only");
        println!("  3) Microphone + system audio (default)");
    } else {
        println!(
            "  2) System audio only (unavailable in this build — will record mic-only if you pick 3)"
        );
        println!("  3) Microphone + system audio attempt (default, mic only)");
    }
    print!("Enter 1, 2, or 3 [3]: ");
    io::stdout().flush().context("flush stdout")?;

    let mut line = String::new();
    stdin
        .lock()
        .read_line(&mut line)
        .context("read choice from stdin")?;

    let choice = line.trim();
    let n = if choice.is_empty() { "3" } else { choice };

    match n {
        "1" => cli.mic = true,
        "2" => cli.system_audio = true,
        "3" => {
            cli.mic = true;
            cli.system_audio = true;
        }
        _ => {
            anyhow::bail!("Invalid choice {n:?}: expected 1, 2, or 3");
        }
    }

    Ok(())
}

fn output_path(args: &StartArgs) -> PathBuf {
    let stem = args
        .name
        .clone()
        .unwrap_or_else(|| chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string());
    args.dir.join(format!("{stem}.md"))
}

fn process_output_path(args: &ProcessArgs) -> PathBuf {
    let stem = args.name.clone().unwrap_or_else(|| {
        args.wav
            .file_stem()
            .and_then(|s| s.to_str())
            .map(String::from)
            .unwrap_or_else(|| chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string())
    });
    args.dir.join(format!("{stem}.md"))
}

fn eprint_num_speakers_notes(diarize_enabled: bool, num_speakers: Option<u32>) {
    if num_speakers.is_some() {
        if !diarize_enabled {
            eprintln!(
                "{}",
                style("Note: `--num-speakers` is ignored without `--diarize`.")
                    .yellow()
            );
        } else if !cfg!(feature = "diarize") {
            eprintln!(
                "{}",
                style(
                    "Note: `--num-speakers` is ignored (build without `--features diarize`; speaker labels disabled).",
                )
                .yellow()
            );
        }
    }
}

/// Whisper + optional diarization + markdown; used after capture or when loading a WAV.
struct MeetingFromSamplesOpts {
    out_md: PathBuf,
    wav_path_for_success_line: PathBuf,
    whisper_model: WhisperSize,
    verbose: bool,
    cpu: bool,
    diarize_enabled: bool,
    /// Diarization settings; only consumed when built with `--features diarize`.
    #[cfg_attr(not(feature = "diarize"), allow(dead_code))]
    diarize_params: DiarizeParams,
    /// Wall time for `start`, or audio length for `process`.
    meeting_duration: Option<Duration>,
}

fn transcribe_samples_to_markdown(samples: Vec<f32>, opts: MeetingFromSamplesOpts) -> Result<()> {
    let diarize_wanted = opts.diarize_enabled;
    let diarize_runs = diarize_wanted && cfg!(feature = "diarize");

    if diarize_wanted && !cfg!(feature = "diarize") {
        eprintln!(
            "{}",
            style(
                "Note: you passed `--diarize`, but this binary was built without the `diarize` feature; rebuild with `--features diarize` to enable speaker labels.",
            )
            .yellow()
        );
    }

    let device = pick_device(opts.cpu).context("device")?;
    if opts.verbose {
        eprintln!("Using device: {device:?}");
    }

    let which: WhichModel = opts.whisper_model.into();
    let whisper_label = format!("{:?}", opts.whisper_model).to_lowercase();

    let samples = Arc::new(samples);

    #[cfg(feature = "diarize")]
    let diarize_join: Option<DiarizeLabelsJoinHandle> = if diarize_runs {
        let s = Arc::clone(&samples);
        let cfg = opts.diarize_params.clone();
        let verb = opts.verbose;
        Some(thread::spawn(move || {
            diarize::diarize_samples(&s, &cfg, verb).map(|v| {
                v.into_iter()
                    .map(|d| (d.start_sec, d.end_sec, d.speaker))
                    .collect()
            })
        }))
    } else {
        None
    };
    #[cfg(not(feature = "diarize"))]
    let diarize_join: Option<DiarizeLabelsJoinHandle> = None;

    let audio_secs = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    let bar = indicatif::ProgressBar::new_spinner();
    if diarize_runs {
        bar.set_message(format!(
            "Whisper + speaker diarization ({audio_secs:.1}s audio)…"
        ));
    } else {
        bar.set_message(format!(
            "Transcribing {audio_secs:.1}s of audio with Whisper…"
        ));
    }
    bar.enable_steady_tick(Duration::from_millis(100));
    let segments = transcribe_pcm_samples(&samples, which, &device, 299_792_458, opts.verbose);
    bar.finish_and_clear();
    let segments = segments.context("transcribe")?;

    let speaker_labels: Vec<(f64, f64, String)> = if let Some(h) = diarize_join {
        let joined = h
            .join()
            .map_err(|_| anyhow::anyhow!("speaker diarization thread panicked"))?;
        joined.context("speaker diarization")?
    } else {
        Vec::new()
    };

    let title = format!(
        "# Meeting notes — {}",
        chrono::Local::now().format("%Y-%m-%d %H:%M")
    );
    let doc = MeetingDoc {
        title_line: title,
        whisper_model: whisper_label,
        duration: opts.meeting_duration,
        segments,
        speaker_labels,
    };
    write_meeting_markdown(&opts.out_md, &doc).context("write markdown")?;

    println!(
        "{}",
        style(format!(
            "Saved notes to {} (WAV: {})",
            opts.out_md.display(),
            opts.wav_path_for_success_line.display()
        ))
        .green()
        .bold()
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            let sh: Shell = match shell {
                ShellArg::Zsh => Shell::Zsh,
                ShellArg::Bash => Shell::Bash,
                ShellArg::Fish => Shell::Fish,
            };
            generate(sh, &mut cmd, "active-listener", &mut io::stdout());
        }
        Commands::Install { whisper_model } => {
            cmd_install(whisper_model)?;
        }
        Commands::Uninstall => {
            cmd_uninstall()?;
        }
        Commands::Start(args) => {
            run_start(args)?;
        }
        Commands::Process(args) => {
            run_process(args)?;
        }
    }

    Ok(())
}

fn run_start(mut cli: StartArgs) -> Result<()> {
    if cli.list_devices {
        for n in list_input_devices()? {
            println!("{n}");
        }
        return Ok(());
    }

    prompt_audio_sources(&mut cli).context("audio source selection")?;

    if !cli.mic && cli.device.is_some() {
        eprintln!(
            "{}",
            style("Note: `--device` is ignored without `--mic`.")
                .yellow()
        );
    }

    if cli.system_audio && !system_audio_supported() {
        eprintln!(
            "{}",
            style("System audio capture unavailable in this build; only microphone samples will be used.")
                .yellow()
        );
    }

    validate_diarize_cli(&cli.diarize_args)?;

    eprint_num_speakers_notes(cli.diarize, cli.diarize_args.num_speakers);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_c = stop.clone();
    if ctrlc::set_handler(move || {
        stop_c.store(true, Ordering::SeqCst);
    })
    .is_err()
    {
        eprintln!(
            "{}",
            style(
                "Warning: could not register Ctrl+C handler; you may need to kill the process to stop recording."
            )
            .yellow()
        );
    }

    let max_d = cli.duration.map(Duration::from_secs);
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_message("Recording… 00:00 — press Ctrl+C to stop");
    pb.enable_steady_tick(Duration::from_millis(120));
    let pb_rec = pb.clone();
    let stop_rec = stop.clone();
    let rec_tick = thread::spawn(move || {
        let t0 = Instant::now();
        while !stop_rec.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(500));
            let secs = t0.elapsed().as_secs();
            pb_rec.set_message(format!(
                "Recording… {:02}:{:02} — press Ctrl+C to stop",
                secs / 60,
                secs % 60
            ));
        }
    });

    let t0 = Instant::now();
    let pcm = record_until_stop(
        cli.device.as_deref(),
        cli.mic,
        cli.system_audio,
        stop.clone(),
        max_d,
        cli.verbose,
    )
    .context("record audio")?;
    pb.finish_and_clear();
    let _ = rec_tick.join();

    if pcm.samples.is_empty() {
        anyhow::bail!("No audio captured.");
    }

    let out_md = output_path(&cli);
    let out_wav = out_md.with_extension("wav");

    let Pcm16kMono { samples } = pcm;

    write_wav_16k_mono(&out_wav, &samples).context("write WAV")?;
    println!(
        "{}",
        style(format!("Saved recording to {}", out_wav.display()))
            .green()
            .bold()
    );

    transcribe_samples_to_markdown(
        samples,
        MeetingFromSamplesOpts {
            out_md,
            wav_path_for_success_line: out_wav,
            whisper_model: cli.whisper_model,
            verbose: cli.verbose,
            cpu: cli.cpu,
            diarize_enabled: cli.diarize,
            diarize_params: diarize_params_from_cli(&cli.diarize_args),
            meeting_duration: Some(t0.elapsed()),
        },
    )?;

    Ok(())
}

fn run_process(cli: ProcessArgs) -> Result<()> {
    validate_diarize_cli(&cli.diarize_args)?;

    eprint_num_speakers_notes(cli.diarize, cli.diarize_args.num_speakers);

    let Pcm16kMono { samples } =
        read_wav_16k_mono(&cli.wav).with_context(|| format!("read {}", cli.wav.display()))?;
    if samples.is_empty() {
        anyhow::bail!("Input WAV has no audio samples.");
    }

    let out_md = process_output_path(&cli);
    let audio_duration =
        Duration::from_secs_f64(samples.len() as f64 / WHISPER_SAMPLE_RATE as f64);

    transcribe_samples_to_markdown(
        samples,
        MeetingFromSamplesOpts {
            out_md,
            wav_path_for_success_line: cli.wav,
            whisper_model: cli.whisper_model,
            verbose: cli.verbose,
            cpu: cli.cpu,
            diarize_enabled: cli.diarize,
            diarize_params: diarize_params_from_cli(&cli.diarize_args),
            meeting_duration: Some(audio_duration),
        },
    )
}
