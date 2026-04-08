//! Active Listener — record meetings, transcribe with Whisper, optional local LLM notes.

mod audio;
mod markdown;
mod summarize;
mod transcribe;
mod whisper;

use anstyle::{AnsiColor, Color, Style};
use anyhow::{Context, Result};
use clap::{builder::Styles, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use console::style;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use audio::{list_input_devices, record_until_stop, system_audio_supported};
use markdown::{write_meeting_markdown, MeetingDoc};
use transcribe::{pick_device, transcribe_pcm_samples, WhichModel};

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

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Batch,
    Realtime,
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
    about = "Record meetings, get markdown notes (local Whisper + optional GGUF LLM).",
    long_about = "Captures microphone audio, transcribes with OpenAI Whisper via Candle, and optionally summarizes with a local Llama-compatible GGUF you provide.",
    styles = clap_styles(),
    after_help = "EXAMPLES:\n  active-listener\n  active-listener --dir ~/notes --name standup\n  active-listener --llm-model ~/models/mistral-q4.gguf\n  active-listener completions zsh",
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long, value_enum, default_value_t = Mode::Batch)]
    mode: Mode,

    /// Output directory (default: current working directory).
    #[arg(long, default_value = ".")]
    dir: PathBuf,

    /// Output filename without extension (default: current datetime).
    #[arg(long)]
    name: Option<String>,

    #[arg(long, value_enum, default_value_t = WhisperSize::Medium)]
    whisper_model: WhisperSize,

    /// Path to a GGUF model for summarization (`ACTIVE_LISTENER_LLM_MODEL` if unset).
    #[arg(long, env = "ACTIVE_LISTENER_LLM_MODEL")]
    llm_model: Option<PathBuf>,

    /// Stop recording after this many seconds.
    #[arg(long)]
    duration: Option<u64>,

    #[arg(long, default_value_t = false)]
    no_system_audio: bool,

    #[arg(long, default_value_t = false)]
    no_mic: bool,

    #[arg(long, default_value_t = false)]
    list_devices: bool,

    /// Microphone device name (see --list-devices).
    #[arg(long)]
    device: Option<String>,

    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Force CPU (no Metal).
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Print shell completion script (zsh, bash, or fish).
    Completions {
        #[arg(value_enum)]
        shell: ShellArg,
    },
    /// Install the binary to ~/.local/bin and configure zsh shell integration.
    Install,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ShellArg {
    Zsh,
    Bash,
    Fish,
}

fn cmd_install() -> Result<()> {
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

    println!("\nDone. Run: source ~/.zshrc");
    Ok(())
}

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

fn output_path(cli: &Cli) -> PathBuf {
    let stem = cli
        .name
        .clone()
        .unwrap_or_else(|| chrono::Local::now().format("%Y-%m-%d_%H%M%S").to_string());
    cli.dir.join(format!("{stem}.md"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(Commands::Completions { shell }) = cli.command {
        let mut cmd = Cli::command();
        let sh: Shell = match shell {
            ShellArg::Zsh => Shell::Zsh,
            ShellArg::Bash => Shell::Bash,
            ShellArg::Fish => Shell::Fish,
        };
        generate(sh, &mut cmd, "active-listener", &mut io::stdout());
        return Ok(());
    }

    if let Some(Commands::Install) = cli.command {
        cmd_install()?;
        return Ok(());
    }

    if cli.list_devices {
        for n in list_input_devices()? {
            println!("{n}");
        }
        return Ok(());
    }

    if cli.no_mic {
        anyhow::bail!("--no-mic is not supported yet (microphone required).");
    }

    let capture_system = !cli.no_system_audio;
    if capture_system && !system_audio_supported() {
        eprintln!(
            "{}",
            style("System audio capture unavailable in this build; recording microphone only.")
                .yellow()
        );
    }

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
        capture_system && system_audio_supported(),
        stop.clone(),
        max_d,
    )
    .context("record audio")?;
    pb.finish_and_clear();
    let _ = rec_tick.join();

    if cli.mode == Mode::Realtime {
        eprintln!(
            "{}",
            style("Note: realtime segment streaming is not implemented; full-file transcription runs after recording.")
                .yellow()
        );
    }

    if pcm.samples.is_empty() {
        anyhow::bail!("No audio captured.");
    }

    let device = pick_device(cli.cpu).context("device")?;
    if cli.verbose {
        eprintln!("Using device: {device:?}");
    }

    let which: WhichModel = cli.whisper_model.into();
    let whisper_label = format!("{:?}", cli.whisper_model).to_lowercase();

    let audio_secs = pcm.samples.len() as f64 / audio::WHISPER_SAMPLE_RATE as f64;
    let bar = indicatif::ProgressBar::new_spinner();
    bar.set_message(format!(
        "Transcribing {audio_secs:.1}s of audio with Whisper…"
    ));
    bar.enable_steady_tick(Duration::from_millis(100));
    let segments = transcribe_pcm_samples(&pcm.samples, which, &device, 299_792_458, cli.verbose)
        .context("transcribe")?;
    bar.finish_and_clear();

    let transcript: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let llm_md = if let Some(ref p) = cli.llm_model {
        let b = indicatif::ProgressBar::new_spinner();
        b.set_message("Summarizing with local LLM…");
        b.enable_steady_tick(Duration::from_millis(100));
        let r = summarize::summarize_with_gguf(p, &transcript, &device);
        b.finish_and_clear();
        Some(r.context("LLM summarize")?)
    } else {
        None
    };

    let out = output_path(&cli);
    let title = format!(
        "# Meeting notes — {}",
        chrono::Local::now().format("%Y-%m-%d %H:%M")
    );
    let doc = MeetingDoc {
        title_line: title,
        whisper_model: whisper_label,
        duration: Some(t0.elapsed()),
        llm_markdown: llm_md,
        segments,
    };
    write_meeting_markdown(&out, &doc).context("write markdown")?;

    println!(
        "{}",
        style(format!("Saved notes to {}", out.display()))
            .green()
            .bold()
    );
    Ok(())
}
