//! Optional local GGUF LLM summarization (Llama-family via `quantized_llama`).

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::fs::File;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn tokenizer_path_for_gguf(gguf: &Path) -> PathBuf {
    gguf.with_file_name("tokenizer.json")
}

/// Build a Llama/Mistral-style instruct prompt (works for many GGUF chat models).
fn meeting_notes_prompt(transcript: &str) -> String {
    format!(
        "<s>[INST] You are a meeting notes assistant. Given the transcript below, write structured markdown with sections: Summary, Key discussion points, Action items, Decisions made. Use bullet lists where appropriate.\n\nTranscript:\n{transcript}\n [/INST]"
    )
}

/// Run quantized Llama-compatible GGUF + `tokenizer.json` beside the file. Returns markdown text.
pub fn summarize_with_gguf(gguf_path: &Path, transcript: &str, device: &Device) -> Result<String> {
    let tok_path = tokenizer_path_for_gguf(gguf_path);
    if !tok_path.is_file() {
        anyhow::bail!(
            "tokenizer.json not found next to {} (copy it from the model repo)",
            gguf_path.display()
        );
    }
    let tokenizer =
        Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let mut file =
        File::open(gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
    let ct = gguf_file::Content::read(&mut file).context("read gguf")?;
    let mut model = ModelWeights::from_gguf(ct, &mut file, device).context("load ModelWeights")?;

    const MAX_SEQ: usize = 2048;
    const MAX_NEW: usize = 512;
    let max_prompt_tokens = MAX_SEQ - MAX_NEW;

    let mut body = transcript.to_string();
    let prompt_tokens: Vec<u32> = loop {
        let p = meeting_notes_prompt(&body);
        let enc = tokenizer
            .encode(p.as_str(), true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        if enc.len() <= max_prompt_tokens {
            break enc.get_ids().to_vec();
        }
        if body.len() < 400 {
            anyhow::bail!(
                "transcript is too long for the model context even after truncation; try a shorter recording"
            );
        }
        let mut new_len = body.len() * 4 / 5;
        while new_len > 0 && !body.is_char_boundary(new_len) {
            new_len -= 1;
        }
        body.truncate(new_len);
        body.push_str("\n\n[... end of transcript omitted for context limit ...]");
    };

    let mut logits_processor = LogitsProcessor::from_sampling(
        299_792_458,
        Sampling::TopP {
            p: 0.9,
            temperature: 0.7,
        },
    );

    let mut all_tokens: Vec<u32> = vec![];

    let eos_ids: Vec<u32> = [
        "</s>",
        "<|end_of_text|>",
        "<|endoftext|>",
        "<|im_end|>",
    ]
    .iter()
    .filter_map(|t| tokenizer.token_to_id(t))
    .collect();

    let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;

    for index in 0..MAX_NEW {
        if eos_ids.contains(&next_token) {
            break;
        }
        all_tokens.push(next_token);
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;
    }

    let text = tokenizer
        .decode(&all_tokens, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    Ok(text.trim().to_string())
}
