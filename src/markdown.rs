//! Write meeting notes markdown (frontmatter + transcript).

use crate::transcribe::TranscriptSegment;
use anyhow::{Context, Result};
use chrono::{DateTime, Local};
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Default)]
pub struct MeetingDoc {
    pub title_line: String,
    pub whisper_model: String,
    pub duration: Option<Duration>,
    pub segments: Vec<TranscriptSegment>,
    /// `(start_sec, end_sec, speaker_label)` from diarization; empty if not used.
    pub speaker_labels: Vec<(f64, f64, String)>,
}

pub fn write_meeting_markdown(path: &Path, doc: &MeetingDoc) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all {}", parent.display()))?;
    }
    let now: DateTime<Local> = Local::now();
    let dur = doc.duration.map(format_duration).unwrap_or_default();
    let mut s = String::new();
    s.push_str("---\n");
    s.push_str(&format!("date: {}\n", now.format("%Y-%m-%dT%H:%M:%S")));
    if !dur.is_empty() {
        s.push_str(&format!("duration: {dur}\n"));
    }
    s.push_str(&format!("whisper_model: {}\n", doc.whisper_model));
    s.push_str("---\n\n");
    s.push_str(&doc.title_line);
    s.push_str("\n\n");
    s.push_str("## Transcript\n\n");
    let mut prev_speaker: Option<String> = None;
    for seg in &doc.segments {
        if !doc.speaker_labels.is_empty() {
            let slices = diarization_slices_within_asr(&doc.speaker_labels, seg.start_sec, seg.end_sec);
            if slices.len() > 1 {
                let weights: Vec<f64> = slices.iter().map(|(a, b, _)| b - a).collect();
                let parts = split_text_by_weights(&seg.text, &weights);
                for (t0, _t1, spk, part) in coalesce_same_speaker_runs(&slices, &parts) {
                    let part = part.trim();
                    if part.is_empty() {
                        continue;
                    }
                    if prev_speaker.as_deref() != Some(spk.as_str()) {
                        s.push_str(&format!("**{spk}**\n\n"));
                        prev_speaker = Some(spk.clone());
                    }
                    append_timestamp_line(&mut s, t0, part);
                }
                continue;
            }
            if slices.len() == 1 {
                let (t0, _t1, spk) = &slices[0];
                if prev_speaker.as_deref() != Some(spk.as_str()) {
                    s.push_str(&format!("**{spk}**\n\n"));
                    prev_speaker = Some(spk.clone());
                }
                append_timestamp_line(&mut s, *t0, seg.text.trim());
                continue;
            }
            if let Some(spk) = speaker_for_interval(&doc.speaker_labels, seg.start_sec, seg.end_sec)
            {
                if prev_speaker.as_deref() != Some(spk) {
                    s.push_str(&format!("**{spk}**\n\n"));
                    prev_speaker = Some(spk.to_string());
                }
            } else {
                prev_speaker = None;
            }
        }
        append_timestamp_line(&mut s, seg.start_sec, seg.text.trim());
    }
    fs::write(path, s).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let m = secs / 60;
    let s = secs % 60;
    format!("{m}m {s}s")
}

fn append_timestamp_line(out: &mut String, start_sec: f64, text: &str) {
    let sm = (start_sec / 60.0).floor() as u32;
    let ss = (start_sec % 60.0).floor() as u32;
    out.push_str(&format!("**[{:02}:{:02}]** {}\n\n", sm, ss, text));
}

/// Join consecutive diarization slices that share a speaker into one timestamp line.
fn coalesce_same_speaker_runs(
    slices: &[(f64, f64, String)],
    parts: &[String],
) -> Vec<(f64, f64, String, String)> {
    let mut out: Vec<(f64, f64, String, String)> = Vec::new();
    for (i, (t0, t1, spk)) in slices.iter().enumerate() {
        let p = parts.get(i).map(String::as_str).unwrap_or("").trim();
        if p.is_empty() {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last.2 == *spk {
                last.1 = *t1;
                last.3.push(' ');
                last.3.push_str(p);
                continue;
            }
        }
        out.push((*t0, *t1, spk.clone(), p.to_string()));
    }
    out
}

/// Split `text` into `weights.len()` word runs with lengths proportional to positive `weights`.
fn split_text_by_weights(text: &str, weights: &[f64]) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = weights.len();
    if n == 0 {
        return vec![];
    }
    let total: f64 = weights.iter().copied().filter(|w| *w > 0.0).sum();
    let nw = words.len();
    if total <= 0.0 || nw == 0 {
        return vec![String::new(); n];
    }
    let mut out = Vec::with_capacity(n);
    let mut wi = 0usize;
    for (i, w) in weights.iter().enumerate() {
        let is_last = i + 1 == n;
        let count = if is_last {
            nw.saturating_sub(wi)
        } else if *w <= 0.0 {
            0
        } else {
            ((nw as f64) * w / total).floor() as usize
        };
        let end = (wi + count).min(nw);
        out.push(words[wi..end].join(" "));
        wi = end;
    }
    out
}

/// Collects sorted time boundaries from `[a0, a1]` and diarization clips, then assigns each
/// sub-interval a speaker (shortest overlapping diarization span wins at the midpoint).
fn diarization_slices_within_asr(
    labels: &[(f64, f64, String)],
    a0: f64,
    a1: f64,
) -> Vec<(f64, f64, String)> {
    if !(a1 > a0) {
        return vec![];
    }
    let mut cuts: Vec<f64> = vec![a0, a1];
    for (s, e, _) in labels {
        let s2 = s.clamp(a0, a1);
        let e2 = e.clamp(a0, a1);
        if e2 > s2 {
            cuts.push(s2);
            cuts.push(e2);
        }
    }
    cuts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    cuts.dedup_by(|a, b| (*b - *a).abs() < 1e-4);

    let mut raw: Vec<(f64, f64, String)> = Vec::new();
    let mut i = 0usize;
    while i + 1 < cuts.len() {
        let t0 = cuts[i];
        let t1 = cuts[i + 1];
        i += 1;
        if t1 - t0 < 1e-4 {
            continue;
        }
        let mid = (t0 + t1) * 0.5;
        if let Some(spk) = speaker_at_time(labels, mid) {
            raw.push((t0, t1, spk.to_string()));
        }
    }

    // Merge adjacent slices that share the same speaker.
    let mut merged: Vec<(f64, f64, String)> = Vec::new();
    for (t0, t1, spk) in raw {
        if let Some(last) = merged.last_mut() {
            if last.2 == spk && (t0 - last.1).abs() < 1e-3 {
                last.1 = t1;
                continue;
            }
        }
        merged.push((t0, t1, spk));
    }
    merged
}

fn speaker_at_time<'a>(labels: &'a [(f64, f64, String)], t: f64) -> Option<&'a str> {
    labels
        .iter()
        .filter(|(s, e, _)| t >= *s && t < *e)
        .min_by(|a, b| {
            let da = a.1 - a.0;
            let db = b.1 - b.0;
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, _, n)| n.as_str())
}

fn overlap_seconds(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    let s = a0.max(b0);
    let e = a1.min(b1);
    (e - s).max(0.0)
}

/// Picks the diarization label with the largest time overlap with `[start_sec, end_sec]`.
fn speaker_for_interval(
    labels: &[(f64, f64, String)],
    start_sec: f64,
    end_sec: f64,
) -> Option<&str> {
    labels
        .iter()
        .map(|(s, e, name)| {
            (
                overlap_seconds(start_sec, end_sec, *s, *e),
                name.as_str(),
            )
        })
        .filter(|(o, _)| *o > 0.0)
        .max_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, n)| n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_transcript_section() {
        let doc = MeetingDoc {
            title_line: "# Test".into(),
            whisper_model: "base".into(),
            duration: Some(Duration::from_secs(125)),
            segments: vec![TranscriptSegment {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "hello".into(),
            }],
            speaker_labels: vec![],
        };
        let tmp = std::env::temp_dir().join("active-listener-test.md");
        write_meeting_markdown(&tmp, &doc).unwrap();
        let t = std::fs::read_to_string(&tmp).unwrap();
        assert!(t.contains("whisper_model: base"));
        assert!(t.contains("**[00:00]** hello"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn splits_long_asr_segment_when_diarization_has_multiple_speakers() {
        let doc = MeetingDoc {
            title_line: "# Test".into(),
            whisper_model: "base".into(),
            duration: None,
            speaker_labels: vec![
                (0.0, 5.0, "Speaker 1".into()),
                (5.0, 10.0, "Speaker 2".into()),
            ],
            segments: vec![TranscriptSegment {
                start_sec: 0.0,
                end_sec: 10.0,
                text: "one two three four five six seven eight".into(),
            }],
        };
        let tmp = std::env::temp_dir().join("active-listener-test-diarize-split.md");
        write_meeting_markdown(&tmp, &doc).unwrap();
        let t = std::fs::read_to_string(&tmp).unwrap();
        assert!(t.contains("**Speaker 1**"));
        assert!(t.contains("**Speaker 2**"));
        assert!(t.contains("one two three four"));
        assert!(t.contains("five six seven eight"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn coalesce_joins_adjacent_runs_for_same_speaker() {
        let slices = vec![
            (0.0, 1.0, "Speaker 1".into()),
            (1.0, 2.0, "Speaker 1".into()),
        ];
        let parts = vec!["hello".to_string(), "world".to_string()];
        let r = coalesce_same_speaker_runs(&slices, &parts);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].3, "hello world");
    }

    #[test]
    fn prepends_speaker_heading_when_labels_overlap() {
        let doc = MeetingDoc {
            title_line: "# Test".into(),
            whisper_model: "base".into(),
            duration: None,
            speaker_labels: vec![
                (0.0, 2.0, "Speaker 1".into()),
                (2.0, 10.0, "Speaker 2".into()),
            ],
            segments: vec![
                TranscriptSegment {
                    start_sec: 0.5,
                    end_sec: 1.0,
                    text: "first".into(),
                },
                TranscriptSegment {
                    start_sec: 2.5,
                    end_sec: 3.0,
                    text: "second".into(),
                },
                TranscriptSegment {
                    start_sec: 3.1,
                    end_sec: 3.5,
                    text: "third".into(),
                },
            ],
        };
        let tmp = std::env::temp_dir().join("active-listener-test-diarize.md");
        write_meeting_markdown(&tmp, &doc).unwrap();
        let t = std::fs::read_to_string(&tmp).unwrap();
        assert!(t.contains("**Speaker 1**"));
        assert!(t.contains("**Speaker 2**"));
        let speaker2_count = t.matches("**Speaker 2**").count();
        assert_eq!(speaker2_count, 1, "same speaker: single heading");
        let _ = std::fs::remove_file(&tmp);
    }
}
