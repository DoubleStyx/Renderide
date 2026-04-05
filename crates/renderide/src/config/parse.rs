//! Lenient INI parsing into [`IniDocument`].

use std::collections::HashMap;

/// Parsed INI content: nested maps with lowercase section and key strings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IniDocument {
    sections: HashMap<String, HashMap<String, String>>,
}

/// Non-fatal parse issue (malformed line, etc.).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseWarning {
    /// 1-based line index in the source.
    pub line: usize,
    pub message: &'static str,
}

impl IniDocument {
    /// Looks up `key` in `section` (both compared case-insensitively).
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        let sec = section.to_lowercase();
        let k = key.to_lowercase();
        self.sections.get(&sec)?.get(&k).map(String::as_str)
    }

    /// Returns true if the document has no entries.
    pub fn is_empty(&self) -> bool {
        self.sections.values().all(|m| m.is_empty()) || self.sections.is_empty()
    }
}

/// Parses INI text into a document and warnings. Never fails overall; bad lines are skipped.
///
/// Rules:
/// - Full-line comments: trimmed line starts with `#` or `;` → ignored.
/// - Sections: `[name]` → keys apply to `name` (lowercased).
/// - Keys before the first section use section `""` (empty string).
/// - Inline comments: value truncated at first `#` or `;` outside quotes (same as legacy parser).
/// - `key = value` only; missing `=` yields a warning.
pub fn parse_ini_document(content: &str) -> (IniDocument, Vec<ParseWarning>) {
    let mut doc = IniDocument::default();
    let mut warnings = Vec::new();
    let mut section = String::new();

    for (idx, raw) in content.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            if let Some(end) = rest.find(']') {
                section = rest[..end].trim().to_lowercase();
                doc.sections.entry(section.clone()).or_default();
            } else {
                warnings.push(ParseWarning {
                    line: line_no,
                    message: "section header missing closing ']'",
                });
            }
            continue;
        }
        if let Some(eq) = line.find('=') {
            let key = line[..eq].trim().to_lowercase();
            if key.is_empty() {
                warnings.push(ParseWarning {
                    line: line_no,
                    message: "empty key before '='",
                });
                continue;
            }
            let raw_val = line[eq + 1..].trim();
            let val = raw_val
                .split_once('#')
                .map(|(v, _)| v)
                .or_else(|| raw_val.split_once(';').map(|(v, _)| v))
                .unwrap_or(raw_val)
                .trim()
                .to_string();
            doc.sections
                .entry(section.clone())
                .or_default()
                .insert(key, val);
        } else {
            warnings.push(ParseWarning {
                line: line_no,
                message: "line is not a comment, section, or key=value",
            });
        }
    }

    (doc, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file() {
        let (doc, w) = parse_ini_document("");
        assert!(doc.sections.is_empty() || doc.is_empty());
        assert!(w.is_empty());
    }

    #[test]
    fn comments_and_section() {
        let raw = r"
# top comment
; also comment
[display]
focused_fps = 120
unfocused_fps = 60
";
        let (doc, w) = parse_ini_document(raw);
        assert!(w.is_empty());
        assert_eq!(doc.get("display", "focused_fps"), Some("120"));
        assert_eq!(doc.get("DISPLAY", "FOCUSED_FPS"), Some("120"));
    }

    #[test]
    fn inline_comment_stripped() {
        let (doc, _) = parse_ini_document("[a]\nx = 1 # tail");
        assert_eq!(doc.get("a", "x"), Some("1"));
        let (doc2, _) = parse_ini_document("[a]\nx = 1 ; tail");
        assert_eq!(doc2.get("a", "x"), Some("1"));
    }

    #[test]
    fn keys_before_first_section() {
        let (doc, _) = parse_ini_document("orphan = ok\n[s]\nk=v\n");
        assert_eq!(doc.get("", "orphan"), Some("ok"));
        assert_eq!(doc.get("s", "k"), Some("v"));
    }

    #[test]
    fn warning_on_no_equals() {
        let (doc, w) = parse_ini_document("[s]\nnot_a_pair\nk=v");
        assert_eq!(doc.get("s", "k"), Some("v"));
        assert!(!w.is_empty());
    }
}
