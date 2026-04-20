//! Library surface for the `active-listener` binary and integration tests.

use std::path::PathBuf;

/// Default sherpa-onnx clustering threshold when speaker count is unknown (`start --diarize-threshold`).
/// Higher values merge more aggressively (fewer speakers); lower values split more (more speakers).
pub const DEFAULT_DIARIZE_CLUSTER_THRESHOLD: f32 = 0.9;

/// Sherpa default: speech shorter than this (seconds) is dropped before clustering.
pub const DEFAULT_DIARIZE_MIN_DURATION_ON: f32 = 0.3;

/// Sherpa default: same-speaker gaps shorter than this (seconds) are merged.
pub const DEFAULT_DIARIZE_MIN_DURATION_OFF: f32 = 0.5;

/// Parameters for offline speaker diarization (`--diarize-*` flags).
#[derive(Clone, Debug)]
pub struct DiarizeParams {
    pub num_speakers: Option<u32>,
    pub cluster_threshold: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
    pub embedding_model: Option<PathBuf>,
}

impl Default for DiarizeParams {
    fn default() -> Self {
        Self {
            num_speakers: None,
            cluster_threshold: DEFAULT_DIARIZE_CLUSTER_THRESHOLD,
            min_duration_on: DEFAULT_DIARIZE_MIN_DURATION_ON,
            min_duration_off: DEFAULT_DIARIZE_MIN_DURATION_OFF,
            embedding_model: None,
        }
    }
}

pub mod audio;
#[cfg(feature = "diarize")]
pub mod diarize;
pub mod markdown;
pub mod transcribe;
pub mod whisper;
pub mod system_audio;
