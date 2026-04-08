//! Download OpenAI Whisper weights (safetensors) and run inference via the vendored `whisper` module.

use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use std::fs;
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

/// Hugging Face hub folder prefix for all `openai/whisper-*` model repos (see `hf_hub::Repo::folder_name`).
const OPENAI_WHISPER_HUB_DIR_PREFIX: &str = "models--openai--whisper-";

/// Remove every `models--openai--whisper-*` directory under the Hugging Face hub cache.
/// Returns how many directories were removed.
pub fn delete_all_openai_whisper_hub_caches() -> Result<usize> {
    let cache = Cache::from_env();
    let hub = cache.path();
    if !hub.is_dir() {
        return Ok(0);
    }
    let mut removed = 0usize;
    for entry in fs::read_dir(hub).with_context(|| format!("read_dir {}", hub.display()))? {
        let entry = entry.with_context(|| format!("read_dir entry under {}", hub.display()))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        if !name
            .to_string_lossy()
            .starts_with(OPENAI_WHISPER_HUB_DIR_PREFIX)
        {
            continue;
        }
        fs::remove_dir_all(&path)
            .with_context(|| format!("remove Whisper hub cache at {}", path.display()))?;
        removed += 1;
    }
    Ok(removed)
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
    use hf_hub::{Repo, RepoType};

    #[test]
    fn which_base_revision() {
        let (id, _) = hf_whisper_repo(WhichModel::Base);
        assert!(id.contains("whisper"));
    }

    #[test]
    fn whisper_medium_hub_folder_name_matches_hf_layout() {
        let repo = Repo::new("openai/whisper-medium".to_string(), RepoType::Model);
        assert_eq!(repo.folder_name(), "models--openai--whisper-medium");
    }

    #[test]
    fn all_whisper_sizes_use_openai_whisper_hub_prefix() {
        for which in [
            WhichModel::Tiny,
            WhichModel::Base,
            WhichModel::Small,
            WhichModel::Medium,
            WhichModel::Large,
        ] {
            let (id, _) = hf_whisper_repo(which);
            let repo = Repo::new(id, RepoType::Model);
            assert!(
                repo.folder_name().starts_with(OPENAI_WHISPER_HUB_DIR_PREFIX),
                "{}",
                repo.folder_name()
            );
        }
    }
}
