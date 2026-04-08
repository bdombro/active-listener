//! Microphone capture (cpal), optional system audio, mixing, resampling to 16 kHz mono.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use crossbeam_channel::unbounded;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::system_audio::{start_system_capture, system_audio_supported, SystemCaptureHandle};

pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Mixed mono f32 PCM at 16 kHz for Whisper.
#[derive(Debug, Clone)]
pub struct Pcm16kMono {
    pub samples: Vec<f32>,
}

/// Sum two mono streams sample-by-sample, soft-clipping to [-1, 1].
/// Pads the shorter stream with silence.
pub fn mix_streams(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            let sa = a.get(i).copied().unwrap_or(0.0);
            let sb = b.get(i).copied().unwrap_or(0.0);
            (sa + sb).clamp(-1.0, 1.0)
        })
        .collect()
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

/// Write mono f32 samples (16 kHz, clamped to ±1) as a PCM16 little-endian WAV file.
pub fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> Result<()> {
    let sr = WHISPER_SAMPLE_RATE;
    let n = samples.len();
    let data_bytes = n
        .checked_mul(2)
        .context("WAV sample count overflow")? as u32;
    let riff_chunk_size = 36u32
        .checked_add(data_bytes)
        .context("WAV size overflow")?;

    let mut pcm = Vec::with_capacity(n.saturating_mul(2));
    for &x in samples {
        let v = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }

    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(b"RIFF")?;
    f.write_all(&riff_chunk_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sr.to_le_bytes())?;
    let byte_rate = sr
        .checked_mul(2)
        .context("WAV byte rate overflow")?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_bytes.to_le_bytes())?;
    f.write_all(&pcm)
        .with_context(|| format!("write WAV samples {}", path.display()))?;
    Ok(())
}

/// Read a PCM16 little-endian mono 16 kHz WAV file as written by [`write_wav_16k_mono`].
pub fn read_wav_16k_mono(path: &Path) -> Result<Pcm16kMono> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {}", path.display()))?;

    if bytes.len() < 12 {
        anyhow::bail!("WAV file too small");
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("not a RIFF/WAVE file");
    }

    let mut i = 12usize;
    let mut audio_format: Option<u16> = None;
    let mut num_channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut pcm_payload: Option<&[u8]> = None;

    while i + 8 <= bytes.len() {
        let chunk_id = &bytes[i..i + 4];
        let size = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        i += 8;
        let end = i
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .context("truncated WAV chunk")?;
        let payload = &bytes[i..end];
        i = end;
        if size % 2 == 1 {
            i = i
                .checked_add(1)
                .filter(|&j| j <= bytes.len())
                .context("truncated WAV padding byte")?;
        }

        match chunk_id {
            b"fmt " => {
                if payload.len() < 16 {
                    anyhow::bail!("WAV fmt chunk too small");
                }
                audio_format = Some(u16::from_le_bytes(payload[0..2].try_into().unwrap()));
                num_channels = Some(u16::from_le_bytes(payload[2..4].try_into().unwrap()));
                sample_rate = Some(u32::from_le_bytes(payload[4..8].try_into().unwrap()));
                bits_per_sample = Some(u16::from_le_bytes(payload[14..16].try_into().unwrap()));
            }
            b"data" => pcm_payload = Some(payload),
            _ => {}
        }
    }

    let af = audio_format.context("WAV missing fmt chunk")?;
    let nc = num_channels.context("WAV missing fmt chunk")?;
    let sr = sample_rate.context("WAV missing fmt chunk")?;
    let bps = bits_per_sample.context("WAV missing fmt chunk")?;
    let data = pcm_payload.context("WAV missing data chunk")?;

    if af != 1 {
        anyhow::bail!("expected PCM WAV (format 1), got format {af}");
    }
    if nc != 1 {
        anyhow::bail!("expected mono WAV, got {nc} channels");
    }
    if sr != WHISPER_SAMPLE_RATE {
        anyhow::bail!("expected {} Hz sample rate, got {sr}", WHISPER_SAMPLE_RATE);
    }
    if bps != 16 {
        anyhow::bail!("expected 16-bit PCM, got {bps} bits per sample");
    }
    if data.len() % 2 != 0 {
        anyhow::bail!("WAV data chunk size is not a multiple of 2");
    }

    let n = data.len() / 2;
    let mut samples = Vec::with_capacity(n);
    for chunk in data.chunks_exact(2) {
        let v = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(v as f32 / 32768.0);
    }

    Ok(Pcm16kMono { samples })
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

/// Record until `stop` is set or `max_duration` elapses.
/// At least one of `capture_mic` or `capture_system` must be true.
/// When both are true, streams are resampled separately then mixed (soft-clipped).
pub fn record_until_stop(
    mic_name: Option<&str>,
    capture_mic: bool,
    capture_system: bool,
    stop: Arc<AtomicBool>,
    max_duration: Option<Duration>,
    verbose: bool,
) -> Result<Pcm16kMono> {
    if !capture_mic && !capture_system {
        anyhow::bail!("record_until_stop: enable at least one of capture_mic or capture_system");
    }

    let mut mic_stream: Option<(crossbeam_channel::Receiver<f32>, usize, MicCapture)> = None;
    if capture_mic {
        let device = pick_input_device(mic_name)?;
        let supported = device.default_input_config()?;
        let in_rate = supported.sample_rate().0 as usize;
        let (mic_tx, mic_rx) = unbounded::<f32>();
        let _mic = MicCapture::run(device, stop.clone(), mic_tx)?;
        mic_stream = Some((mic_rx, in_rate, _mic));
    }

    let mut sys_capture: Option<(crossbeam_channel::Receiver<f32>, u32, SystemCaptureHandle)> = None;
    if capture_system && system_audio_supported() {
        let (sys_tx, sys_rx) = unbounded::<f32>();
        let (handle, info) =
            start_system_capture(stop.clone(), sys_tx).context("start system audio capture")?;
        if verbose {
            eprintln!(
                "System audio backend: {} ({} Hz)",
                info.backend_name, info.sample_rate
            );
        }
        sys_capture = Some((sys_rx, info.sample_rate, handle));
    }

    let start = Instant::now();
    let mut raw_mic = Vec::<f32>::new();
    let mut raw_sys = Vec::<f32>::new();

    while !stop.load(Ordering::SeqCst) {
        if let Some(max) = max_duration {
            if start.elapsed() >= max {
                break;
            }
        }
        if let Some((ref mic_rx, _, _)) = mic_stream {
            match mic_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(s) => {
                    raw_mic.push(s);
                    while let Ok(more) = mic_rx.try_recv() {
                        raw_mic.push(more);
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            thread::sleep(Duration::from_millis(100));
        }
        if let Some((ref sys_rx, _, _)) = sys_capture {
            while let Ok(s) = sys_rx.try_recv() {
                raw_sys.push(s);
            }
        }
    }
    stop.store(true, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(50));
    if let Some((ref mic_rx, _, _)) = mic_stream {
        while let Ok(s) = mic_rx.try_recv() {
            raw_mic.push(s);
        }
    }
    if let Some((ref sys_rx, _, _)) = sys_capture {
        while let Ok(s) = sys_rx.try_recv() {
            raw_sys.push(s);
        }
    }

    let sys_rate = sys_capture.as_ref().map(|(_, r, _)| *r);
    let mic_in_rate = mic_stream.as_ref().map(|(_, r, _)| *r as u32);
    drop(mic_stream);
    drop(sys_capture);

    let mic_16k = if let Some(rate) = mic_in_rate {
        resample_mono_linear(&raw_mic, rate, WHISPER_SAMPLE_RATE)
    } else {
        Vec::new()
    };

    let samples = match (capture_mic, sys_rate) {
        (true, Some(sr)) => {
            let sys_16k = resample_mono_linear(&raw_sys, sr, WHISPER_SAMPLE_RATE);
            mix_streams(&mic_16k, &sys_16k)
        }
        (true, None) => mic_16k,
        (false, Some(sr)) => resample_mono_linear(&raw_sys, sr, WHISPER_SAMPLE_RATE),
        (false, None) => Vec::new(),
    };

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

    #[test]
    fn mix_streams_equal_length_sums() {
        let a = vec![0.25_f32, -0.25];
        let b = vec![0.25_f32, 0.25];
        let m = mix_streams(&a, &b);
        assert_eq!(m.len(), 2);
        assert!((m[0] - 0.5).abs() < 1e-6);
        assert!((m[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mix_streams_pads_shorter_with_silence() {
        let a = vec![0.25_f32, 0.25, 0.25];
        let b = vec![0.25_f32];
        let m = mix_streams(&a, &b);
        assert_eq!(m, vec![0.5, 0.25, 0.25]);
    }

    #[test]
    fn mix_streams_one_empty_equals_other() {
        let a = vec![0.3_f32, -0.1];
        let b: Vec<f32> = vec![];
        assert_eq!(mix_streams(&a, &b), a);
        assert_eq!(mix_streams(&b, &a), a);
    }

    #[test]
    fn mix_streams_both_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(mix_streams(&a, &b).is_empty());
    }

    #[test]
    fn mix_streams_soft_clips() {
        let a = vec![0.9_f32];
        let b = vec![0.9_f32];
        let m = mix_streams(&a, &b);
        assert_eq!(m, vec![1.0]);
    }

    #[test]
    fn wav_writer_round_trip_header() {
        let tmp = std::env::temp_dir().join("active-listener-wav-test.wav");
        let samples = vec![0.0_f32, 1.0, -1.0];
        write_wav_16k_mono(&tmp, &samples).unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn wav_read_round_trip_samples() {
        let tmp = std::env::temp_dir().join("active-listener-wav-roundtrip.wav");
        let samples = vec![0.0_f32, 0.5, -0.25, 1.0, -1.0];
        write_wav_16k_mono(&tmp, &samples).unwrap();
        let Pcm16kMono { samples: read_back } = read_wav_16k_mono(&tmp).unwrap();
        assert_eq!(read_back.len(), samples.len());
        for (a, b) in samples.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < 2e-4, "expected {a}, got {b}");
        }
        let _ = std::fs::remove_file(&tmp);
    }
}
