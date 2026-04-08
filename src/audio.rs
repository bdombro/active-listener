//! Microphone capture (cpal), optional system-audio placeholder, resampling to 16 kHz mono.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use crossbeam_channel::unbounded;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Mixed mono f32 PCM at 16 kHz for Whisper.
#[derive(Debug, Clone)]
pub struct Pcm16kMono {
    pub samples: Vec<f32>,
}

/// List cpal input devices (microphones).
pub fn list_input_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let mut names = Vec::new();
    for d in host.input_devices().context("no input devices")? {
        if let Ok(n) = d.name() {
            names.push(n);
        }
    }
    Ok(names)
}

fn pick_input_device(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    if let Some(want) = name {
        for d in host.input_devices().context("enumerate input devices")? {
            if d.name().ok().as_deref() == Some(want) {
                return Ok(d);
            }
        }
        anyhow::bail!("microphone not found: {want}");
    }
    host.default_input_device()
        .context("no default input device; try --list-devices")
}

/// System audio is not bundled here (ScreenCaptureKit needs full Xcode toolchain to link).
/// Callers should pass `capture_system: false` or accept mic-only until a native backend is added.
pub fn system_audio_supported() -> bool {
    false
}

pub fn start_system_audio_capture() -> Result<()> {
    anyhow::bail!(
        "System audio capture is not available in this build (use --no-system-audio or a virtual loopback device)."
    )
}

struct MicCapture {
    _stream: cpal::Stream,
}

impl MicCapture {
    fn run(
        device: cpal::Device,
        stop: Arc<AtomicBool>,
        tx: crossbeam_channel::Sender<f32>,
    ) -> Result<Self> {
        let supported = device
            .default_input_config()
            .context("default_input_config")?;
        let config: StreamConfig = supported.clone().into();
        let channels = config.channels as usize;
        let sample_format = supported.sample_format();

        let stream = match sample_format {
            SampleFormat::F32 => {
                let stop_c = stop.clone();
                let tx_c = tx.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _| {
                        if stop_c.load(Ordering::SeqCst) {
                            return;
                        }
                        if channels == 0 {
                            return;
                        }
                        for frame in data.chunks(channels) {
                            let v: f32 = frame.iter().copied().sum::<f32>() / channels as f32;
                            if tx_c.send(v).is_err() {
                                return;
                            }
                        }
                    },
                    |e| eprintln!("microphone stream error: {e}"),
                    None,
                )
            }
            SampleFormat::I16 => {
                let stop_c = stop.clone();
                let tx_c = tx.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        if stop_c.load(Ordering::SeqCst) {
                            return;
                        }
                        if channels == 0 {
                            return;
                        }
                        for frame in data.chunks(channels) {
                            let v: f32 = frame.iter().map(|&s| s as f32 / 32768.0).sum::<f32>()
                                / channels as f32;
                            if tx_c.send(v).is_err() {
                                return;
                            }
                        }
                    },
                    |e| eprintln!("microphone stream error: {e}"),
                    None,
                )
            }
            SampleFormat::U16 => {
                let stop_c = stop.clone();
                let tx_c = tx.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        if stop_c.load(Ordering::SeqCst) {
                            return;
                        }
                        if channels == 0 {
                            return;
                        }
                        for frame in data.chunks(channels) {
                            let v: f32 = frame
                                .iter()
                                .map(|&s| (s as f32 / 32768.0) - 1.0)
                                .sum::<f32>()
                                / channels as f32;
                            if tx_c.send(v).is_err() {
                                return;
                            }
                        }
                    },
                    |e| eprintln!("microphone stream error: {e}"),
                    None,
                )
            }
            _ => anyhow::bail!("unsupported microphone sample format: {sample_format:?}"),
        }
        .context("build_input_stream")?;
        stream.play().context("play mic stream")?;
        Ok(MicCapture { _stream: stream })
    }
}

/// Linear interpolation resampling (good enough for speech; avoids rubato fixed-block state).
fn resample_mono_linear(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = input_rate as f64 / output_rate as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len.max(1));
    for i in 0..out_len.max(1) {
        let src_pos = i as f64 * ratio;
        let j = src_pos.floor() as usize;
        let frac = (src_pos - j as f64) as f32;
        let a = input.get(j).copied().unwrap_or(0.0);
        let b = input.get(j.saturating_add(1)).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

/// Record from the microphone until `stop` is set or `max_duration` elapses.
/// When `capture_system` is true, attempts system audio (currently always fails — use mic-only).
pub fn record_until_stop(
    mic_name: Option<&str>,
    capture_system: bool,
    stop: Arc<AtomicBool>,
    max_duration: Option<Duration>,
) -> Result<Pcm16kMono> {
    if capture_system {
        let _ = start_system_audio_capture();
        // continue with mic only
    }

    let device = pick_input_device(mic_name)?;
    let supported = device.default_input_config()?;
    let in_rate = supported.sample_rate().0 as usize;

    let (tx, rx) = unbounded::<f32>();

    let _mic = MicCapture::run(device, stop.clone(), tx)?;

    let start = Instant::now();
    let mut raw_mono = Vec::<f32>::new();
    while !stop.load(Ordering::SeqCst) {
        if let Some(max) = max_duration {
            if start.elapsed() >= max {
                break;
            }
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(s) => {
                raw_mono.push(s);
                while let Ok(more) = rx.try_recv() {
                    raw_mono.push(more);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
    stop.store(true, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(50));
    while let Ok(s) = rx.try_recv() {
        raw_mono.push(s);
    }
    drop(_mic);

    let samples = resample_mono_linear(&raw_mono, in_rate as u32, WHISPER_SAMPLE_RATE);

    Ok(Pcm16kMono { samples })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_rate_identity() {
        let chunk = vec![0.0_f32, 0.5, -0.5];
        let out = resample_mono_linear(&chunk, WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE);
        assert_eq!(out, chunk);
    }
}
