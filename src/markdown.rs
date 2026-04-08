//! Write meeting notes markdown (frontmatter + optional LLM body + transcript).

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
    pub llm_markdown: Option<String>,
    pub segments: Vec<TranscriptSegment>,
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
    if let Some(ref body) = doc.llm_markdown {
        s.push_str(body.trim());
        if !body.ends_with('\n') {
            s.push('\n');
        }
        s.push_str("\n\n");
    }
    s.push_str("## Transcript\n\n");
    for seg in &doc.segments {
        let sm = (seg.start_sec / 60.0).floor() as u32;
        let ss = (seg.start_sec % 60.0).floor() as u32;
        let em = (seg.end_sec / 60.0).floor() as u32;
        let es = (seg.end_sec % 60.0).floor() as u32;
        s.push_str(&format!(
            "**[{:02}:{:02}–{:02}:{:02}]** {}\n\n",
            sm, ss, em, es, seg.text
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_transcript_section() {
        let doc = MeetingDoc {
            title_line: "# Test".into(),
            whisper_model: "base".into(),
            duration: Some(Duration::from_secs(125)),
            llm_markdown: None,
            segments: vec![TranscriptSegment {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "hello".into(),
            }],
        };
        let tmp = std::env::temp_dir().join("active-listener-test.md");
        write_meeting_markdown(&tmp, &doc).unwrap();
        let t = std::fs::read_to_string(&tmp).unwrap();
        assert!(t.contains("whisper_model: base"));
        assert!(t.contains("**[00:00–00:01]** hello"));
        let _ = std::fs::remove_file(&tmp);
    }
}
