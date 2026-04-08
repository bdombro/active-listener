// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).
//
// Vendored from huggingface/candle `candle-examples` Whisper example; keep upstream shape.
#![allow(dead_code)]

use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{
    ops::{log_softmax, softmax},
    VarBuilder,
};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::SeedableRng;
use tokenizers::Tokenizer;

mod multilingual;

use candle_transformers::models::whisper::{self as m, audio, Config};

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    max_initial_timestamp_index: Option<u32>,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        max_initial_timestamp_index: Option<u32>,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            max_initial_timestamp_index,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let audio_features = self.model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = self.model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = self
                .model
                .decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = self.model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            // Apply timestamp rules when timestamps are enabled
            let logits = if self.timestamps {
                self.apply_timestamp_rules(&logits, &tokens)?
            } else {
                logits
            };

            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token
                || tokens.len() > self.model.config().max_target_positions
            {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    eprintln!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn apply_timestamp_rules(&self, input_logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let device = input_logits.device().clone();
        let timestamp_begin = self.no_timestamps_token + 1;
        let vocab_size = self.model.config().vocab_size as u32;

        // ========== SETUP: Extract sampled tokens for analysis ==========
        let sample_begin = if self.language_token.is_some() { 3 } else { 2 };
        let sampled_tokens = if tokens.len() > sample_begin {
            &tokens[sample_begin..]
        } else {
            &[]
        };

        let mut masks = Vec::new();
        // Pre-allocate reusable mask buffer to avoid repeated allocations
        let mut mask_buffer = vec![0.0f32; vocab_size as usize];

        // ========== RULE 1: Timestamp pairing constraints ==========
        // Timestamps must come in pairs, except directly before EOT
        if !sampled_tokens.is_empty() {
            let last_was_timestamp = sampled_tokens
                .last()
                .map(|&t| t >= timestamp_begin)
                .unwrap_or(false);

            let penultimate_was_timestamp = if sampled_tokens.len() >= 2 {
                sampled_tokens[sampled_tokens.len() - 2] >= timestamp_begin
            } else {
                false
            };

            if last_was_timestamp {
                if penultimate_was_timestamp {
                    // Has to be non-timestamp - suppress timestamp tokens
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i >= timestamp_begin {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
                } else {
                    // Cannot be normal text tokens - suppress everything before EOT
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i < self.eot_token {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
                }
            }

            // ========== RULE 2: Non-decreasing timestamp constraint ==========
            // Timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
            let timestamp_tokens: Vec<u32> = sampled_tokens
                .iter()
                .filter(|&&t| t >= timestamp_begin)
                .cloned()
                .collect();

            if !timestamp_tokens.is_empty() {
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    *timestamp_tokens.last().unwrap()
                } else {
                    timestamp_tokens.last().unwrap() + 1
                };

                for i in 0..vocab_size {
                    mask_buffer[i as usize] = if i >= timestamp_begin && i < timestamp_last {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    };
                }
                masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
            }
        }

        // ========== RULE 3: Force initial timestamp ==========
        // At the beginning, suppress generating non-timestamp tokens
        if tokens.len() == sample_begin {
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);

            // Apply the max_initial_timestamp constraint
            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = timestamp_begin + max_initial_timestamp_index;
                if last_allowed < vocab_size {
                    for i in 0..vocab_size {
                        mask_buffer[i as usize] = if i > last_allowed {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        };
                    }
                    masks.push(Tensor::new(mask_buffer.as_slice(), &device)?);
                }
            }
        }

        // ========== APPLY MASKS: Apply all constraint masks ==========
        let mut logits = input_logits.clone();
        for mask in masks {
            logits = logits.broadcast_add(&mask)?;
        }

        // ========== RULE 4: Probability-based timestamp preference ==========
        // If sum of probability over timestamps is above any other token, sample timestamp
        let log_probs = log_softmax(&logits, 0)?;

        // Extract timestamp and text log probabilities
        let timestamp_log_probs = log_probs.narrow(
            0,
            timestamp_begin as usize,
            vocab_size as usize - timestamp_begin as usize,
        )?;

        let text_log_probs = log_probs.narrow(0, 0, timestamp_begin as usize)?;

        // Implement logsumexp for timestamp tokens (numerically stable)
        let timestamp_logprob = {
            let max_val = timestamp_log_probs.max(0)?;
            let shifted = timestamp_log_probs.broadcast_sub(&max_val)?;
            let exp_shifted = shifted.exp()?;
            let sum_exp = exp_shifted.sum(0)?;
            let log_sum = sum_exp.log()?;
            max_val.broadcast_add(&log_sum)?.to_scalar::<f32>()?
        };

        // Get max text token log probability
        let max_text_token_logprob: f32 = text_log_probs.max(0)?.to_scalar::<f32>()?;

        // Compare in log space
        if timestamp_logprob > max_text_token_logprob {
            // Only consider timestamp tokens
            for i in 0..vocab_size {
                mask_buffer[i as usize] = if i < timestamp_begin {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            let mask_tensor = Tensor::new(mask_buffer.as_slice(), &device)?;
            logits = logits.broadcast_add(&mask_tensor)?;
        }

        Ok(logits)
    }

    fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                if self.verbose {
                    println!("no speech detected, skipping {seek} {dr:?}");
                }
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                if self.verbose {
                    println!(
                        "{:.1}s -- {:.1}s",
                        segment.start,
                        segment.start + segment.duration,
                    );
                }
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            if self.verbose {
                                println!(
                                    "  {:.1}s-{:.1}s: {}",
                                    prev_timestamp_s, timestamp_s, text
                                );
                            }
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if self.verbose && !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else if self.verbose {
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Task {
    Transcribe,
    Translate,
}

/// Maps to Hugging Face `openai/whisper-*` checkpoints (multilingual variants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl WhichModel {
    pub fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
        }
    }
}

/// One decoded time range and text.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

/// Build mel filters from bundled bytes (Whisper 80- or 128-bin banks).
pub fn load_mel_filters(config: &Config) -> anyhow::Result<Vec<f32>> {
    let mel_bytes: &[u8] = match config.num_mel_bins {
        80 => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/melfilters.bytes"
        )),
        128 => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/melfilters128.bytes"
        )),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    Ok(mel_filters)
}

/// PCM at 16 kHz mono -> mel tensor `(1, n_mels, frames)`.
pub fn pcm_to_mel_tensor(
    config: &Config,
    pcm: &[f32],
    mel_filters: &[f32],
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mel = audio::pcm_to_mel(config, pcm, mel_filters);
    let mel_len = mel.len();
    Ok(Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        device,
    )?)
}

/// Load Whisper weights from safetensors on disk.
pub fn load_model_weights(
    config_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    weights_path: &std::path::Path,
    device: &Device,
) -> anyhow::Result<(Model, Config, Tokenizer)> {
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, device)? };
    let model = Model::Normal(m::model::Whisper::load(&vb, config.clone())?);
    Ok((model, config, tokenizer))
}

/// Run decoder and collect segments.
pub fn transcribe_mel(
    mut model: Model,
    tokenizer: Tokenizer,
    mel: &Tensor,
    device: &Device,
    seed: u64,
    verbose: bool,
) -> anyhow::Result<Vec<TranscriptSegment>> {
    let language_token = Some(multilingual::detect_language(
        &mut model,
        &tokenizer,
        mel,
        verbose,
    )?);
    let mut dc = Decoder::new(
        model,
        tokenizer,
        seed,
        device,
        language_token,
        None,
        false,
        None,
        verbose,
    )?;
    let internal = dc.run(mel)?;
    let mut out = Vec::with_capacity(internal.len());
    for s in internal {
        out.push(TranscriptSegment {
            start_sec: s.start,
            end_sec: s.start + s.duration,
            text: s.dr.text.trim().to_string(),
        });
    }
    Ok(out)
}
