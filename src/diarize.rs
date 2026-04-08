//! Download sherpa-onnx diarization models and run offline speaker diarization on 16 kHz mono f32 PCM.

use crate::DiarizeParams;
use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// One diarized interval with a human-readable speaker label (`Speaker 1`, …).
#[derive(Debug, Clone)]
pub struct DiarizeSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub speaker: String,
}

const SEGMENTATION_TAR_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";

/// Upstream release tag uses this spelling (`speaker-recongition-models`).
const EMBEDDING_FILENAME: &str = "nemo_en_titanet_large.onnx";
const EMBEDDING_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx";

const EXPECTED_SAMPLE_RATE: i32 = 16_000;

fn diarize_cache_dir() -> Result<PathBuf> {
    if let Some(h) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(h).join(".cache/active-listener/diarize"));
    }
    #[cfg(windows)]
    if let Some(h) = std::env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(h).join(".cache/active-listener/diarize"));
    }
    anyhow::bail!(
        "could not determine home directory for diarization model cache (set HOME or USERPROFILE)"
    )
}

/// Download segmentation model if needed; resolve embedding ONNX path (downloaded NeMo Titanet large or `params.embedding_model`).
/// Returns `(pyannote_segmentation_onnx, speaker_embedding_onnx)`.
pub fn ensure_diarize_models(verbose: bool, params: &DiarizeParams) -> Result<(PathBuf, PathBuf)> {
    let base = diarize_cache_dir()?;
    fs::create_dir_all(&base).with_context(|| format!("create {}", base.display()))?;

    let seg_dir = base.join("sherpa-onnx-pyannote-segmentation-3-0");
    let seg_fp32 = seg_dir.join("model.onnx");
    let seg_int8 = seg_dir.join("model.int8.onnx");
    if !seg_fp32.is_file() && !seg_int8.is_file() {
        if verbose {
            eprintln!("Downloading diarization segmentation model…");
        }
        download_and_extract_segmentation_tar(&base).with_context(|| {
            format!(
                "download or extract segmentation model into {}",
                base.display()
            )
        })?;
    }
    let seg_model = if seg_fp32.is_file() {
        seg_fp32
    } else if seg_int8.is_file() {
        seg_int8
    } else {
        anyhow::bail!(
            "expected model.onnx or model.int8.onnx under {} after extract",
            seg_dir.display()
        );
    };

    let emb = if let Some(ref p) = params.embedding_model {
        if p.is_file() {
            p.clone()
        } else {
            anyhow::bail!(
                "diarization embedding model not found: {} (download e.g. from sherpa-onnx speaker-recongition-models)",
                p.display()
            );
        }
    } else {
        let emb = base.join(EMBEDDING_FILENAME);
        if !emb.is_file() {
            if verbose {
                eprintln!("Downloading speaker embedding model (NeMo Titanet large)…");
            }
            download_file(EMBEDDING_URL, &emb).with_context(|| {
                format!(
                    "download embedding model to {}",
                    emb.display()
                )
            })?;
        }
        emb
    };

    if verbose {
        eprintln!("Using speaker embedding model: {}", emb.display());
    }

    Ok((seg_model, emb))
}

fn download_and_extract_segmentation_tar(base: &Path) -> Result<()> {
    let tarball = base.join("sherpa-onnx-pyannote-segmentation-3-0.tar.bz2");
    download_file(SEGMENTATION_TAR_URL, &tarball)?;

    let file = File::open(&tarball)
        .with_context(|| format!("open tarball {}", tarball.display()))?;
    let dec = BzDecoder::new(file);
    let mut archive = tar::Archive::new(dec);
    archive
        .unpack(base)
        .with_context(|| format!("extract tarball into {}", base.display()))?;
    fs::remove_file(&tarball).ok();
    Ok(())
}

fn partial_download_path(dest: &Path) -> PathBuf {
    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    dest.with_file_name(format!("{file_name}.partial"))
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(p) = dest.parent() {
        fs::create_dir_all(p).with_context(|| format!("create_dir_all {}", p.display()))?;
    }
    let partial = partial_download_path(dest);
    let _ = fs::remove_file(&partial);

    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP GET {url}"))?;
    let mut reader = resp.into_reader();
    let mut file = BufWriter::new(
        File::create(&partial).with_context(|| format!("create {}", partial.display()))?,
    );
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("write {}", partial.display()))?;
    file.into_inner()
        .context("flush download buffer")?
        .sync_all()
        .with_context(|| format!("sync {}", partial.display()))?;
    match fs::rename(&partial, dest) {
        Ok(()) => {}
        Err(err) if dest.exists() => {
            fs::remove_file(dest)
                .with_context(|| format!("remove existing {}", dest.display()))?;
            fs::rename(&partial, dest).with_context(|| {
                format!(
                    "replace {} -> {} after rename error: {err}",
                    partial.display(),
                    dest.display()
                )
            })?;
        }
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "rename {} -> {}",
                    partial.display(),
                    dest.display()
                )
            });
        }
    }
    Ok(())
}

/// Run offline speaker diarization on 16 kHz mono f32 samples (same layout as Whisper input).
///
/// When `num_speakers` is unset, `cluster_threshold` controls merge vs split (see sherpa-onnx docs).
/// `min_duration_on` / `min_duration_off` tune segment cleanup (defaults match sherpa).
pub fn diarize_samples(
    samples: &[f32],
    params: &DiarizeParams,
    verbose: bool,
) -> Result<Vec<DiarizeSegment>> {
    let (seg_path, emb_path) = ensure_diarize_models(verbose, params)?;

    // Positive: fixed cluster count (`cutree_k`). Non-positive: threshold-based (`cutree_cdist`); -1 is sherpa's default.
    let num_clusters = match params.num_speakers {
        Some(n) if n > 0 => n as i32,
        _ => -1,
    };

    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(seg_path.to_string_lossy().into_owned()),
            },
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(emb_path.to_string_lossy().into_owned()),
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters,
            threshold: params.cluster_threshold,
        },
        min_duration_on: params.min_duration_on,
        min_duration_off: params.min_duration_off,
    };

    let sd = OfflineSpeakerDiarization::create(&config)
        .context("failed to create OfflineSpeakerDiarization (check model paths)")?;

    if sd.sample_rate() != EXPECTED_SAMPLE_RATE {
        anyhow::bail!(
            "diarization model expects {} Hz audio; active-listener records at {} Hz",
            sd.sample_rate(),
            EXPECTED_SAMPLE_RATE
        );
    }

    if verbose {
        eprintln!(
            "diarization: {} samples, {:.1}s, num_clusters={}, cluster_threshold={}, min_duration_on={}, min_duration_off={}",
            samples.len(),
            samples.len() as f64 / EXPECTED_SAMPLE_RATE as f64,
            num_clusters,
            params.cluster_threshold,
            params.min_duration_on,
            params.min_duration_off
        );
    }

    let result = sd
        .process(samples)
        .context("speaker diarization failed (process returned None)")?;

    let raw_segments = result.sort_by_start_time();

    if verbose {
        eprintln!(
            "diarization result: num_speakers={}, num_segments={}",
            result.num_speakers(),
            result.num_segments()
        );
        for seg in &raw_segments {
            eprintln!(
                "  {:.3}–{:.3}s cluster {}",
                seg.start, seg.end, seg.speaker
            );
        }
    }

    let mut out = Vec::new();
    for seg in raw_segments {
        let label = format!("Speaker {}", seg.speaker.saturating_add(1));
        out.push(DiarizeSegment {
            start_sec: f64::from(seg.start),
            end_sec: f64::from(seg.end),
            speaker: label,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarize_cache_dir_respects_home() {
        // Only run when HOME is set (normal on Unix CI).
        if std::env::var("HOME").is_ok() {
            let p = diarize_cache_dir().unwrap();
            assert!(p.to_string_lossy().contains("active-listener"));
        }
    }

    #[test]
    fn partial_download_path_appends_suffix_to_filename() {
        let path = partial_download_path(Path::new("/tmp/model.onnx"));
        assert_eq!(path, PathBuf::from("/tmp/model.onnx.partial"));
    }
}
