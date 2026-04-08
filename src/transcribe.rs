//! Download OpenAI Whisper weights (safetensors) and run inference via the vendored `whisper` module.

use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

pub use crate::whisper::{TranscriptSegment, WhichModel};

use crate::whisper::{load_mel_filters, load_model_weights, pcm_to_mel_tensor, transcribe_mel};

/// Prefer Metal on Apple Silicon when `metal` feature is enabled; otherwise CPU.
pub fn pick_device(force_cpu: bool) -> Result<candle_core::Device> {
    if force_cpu {
        return Ok(candle_core::Device::Cpu);
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        match candle_core::Device::new_metal(0) {
            Ok(d) => return Ok(d),
            Err(e) => {
                eprintln!("Metal unavailable ({e}), using CPU");
            }
        }
    }
    Ok(candle_core::Device::Cpu)
}

fn hf_whisper_repo(which: WhichModel) -> (String, &'static str) {
    let (id, rev) = which.model_and_revision();
    (id.to_string(), rev)
}

/// Ensure `config.json`, `tokenizer.json`, and `model.safetensors` are cached; return paths.
pub fn ensure_whisper_artifacts(which: WhichModel) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let (model_id, revision) = hf_whisper_repo(which);
    let api = Api::new().context("hf-hub Api")?;
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        revision.to_string(),
    ));
    let config = repo.get("config.json").context("download config.json")?;
    let tokenizer = repo
        .get("tokenizer.json")
        .context("download tokenizer.json")?;
    let weights = repo
        .get("model.safetensors")
        .context("download model.safetensors")?;
    Ok((config, tokenizer, weights))
}

/// Full path: load weights, build mel, decode.
pub fn transcribe_pcm_samples(
    pcm_16k_mono: &[f32],
    which: WhichModel,
    device: &candle_core::Device,
    seed: u64,
    verbose: bool,
) -> Result<Vec<TranscriptSegment>> {
    let (cfg_p, tok_p, w_p) = ensure_whisper_artifacts(which)?;
    let (model, config, tokenizer) = load_model_weights(&cfg_p, &tok_p, &w_p, device)?;
    let mel_filters = load_mel_filters(&config)?;
    let mel = pcm_to_mel_tensor(&config, pcm_16k_mono, &mel_filters, device)?;
    transcribe_mel(model, tokenizer, &mel, device, seed, verbose)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn which_base_revision() {
        let (id, _) = hf_whisper_repo(WhichModel::Base);
        assert!(id.contains("whisper"));
    }
}
