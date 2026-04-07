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

    /// Inserts or replaces `key` in `section` (stored as lowercase section and key).
    pub fn set(&mut self, section: &str, key: &str, value: impl Into<String>) {
        let sec = section.to_lowercase();
        let k = key.to_lowercase();
        self.sections
            .entry(sec)
            .or_default()
            .insert(k, value.into());
    }

    /// Removes `key` from `section` if present.
    pub fn remove(&mut self, section: &str, key: &str) -> Option<String> {
        let sec = section.to_lowercase();
        let k = key.to_lowercase();
        let removed = self.sections.get_mut(&sec)?.remove(&k);
        if self.sections.get(&sec).is_none_or(|m| m.is_empty()) {
            self.sections.remove(&sec);
        }
        removed
    }

    /// Returns true if the document has no entries.
    pub fn is_empty(&self) -> bool {
        self.sections.values().all(|m| m.is_empty()) || self.sections.is_empty()
    }

    /// Serializes to INI text with deterministic ordering: orphan keys (`""` section) first, then
    /// `[section]` headers in sorted order; keys within each section sorted lexicographically.
    pub fn serialize(&self) -> String {
        let mut out = String::new();

        if let Some(orphan) = self.sections.get("") {
            if !orphan.is_empty() {
                let mut keys: Vec<_> = orphan.keys().cloned().collect();
                keys.sort();
                for k in keys {
                    if let Some(v) = orphan.get(&k) {
                        out.push_str(&format!("{k} = {v}\n"));
                    }
                }
                out.push('\n');
            }
        }

        let mut section_names: Vec<_> = self
            .sections
            .keys()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        section_names.sort();

        for sec in section_names {
            let Some(map) = self.sections.get(&sec) else {
                continue;
            };
            if map.is_empty() {
                continue;
            }
            out.push('[');
            out.push_str(&sec);
            out.push_str("]\n");
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            for k in keys {
                if let Some(v) = map.get(&k) {
                    out.push_str(&format!("{k} = {v}\n"));
                }
            }
            out.push('\n');
        }

        while out.ends_with('\n') && out.len() > 1 {
            out.pop();
        }
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out
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

    #[test]
    fn set_and_serialize_roundtrip() {
        let mut doc = IniDocument::default();
        doc.set("display", "focused_fps", "240");
        doc.set("display", "unfocused_fps", "60");
        doc.set("rendering", "vsync", "true");
        let s = doc.serialize();
        let (doc2, w) = parse_ini_document(&s);
        assert!(w.is_empty());
        assert_eq!(doc2.get("display", "focused_fps"), Some("240"));
        assert_eq!(doc2.get("rendering", "vsync"), Some("true"));
    }

    #[test]
    fn serialize_deterministic_order() {
        let mut doc = IniDocument::default();
        doc.set("zsec", "b", "2");
        doc.set("asec", "c", "3");
        doc.set("zsec", "a", "1");
        let s = doc.serialize();
        let pos_asec = s.find("[asec]").unwrap();
        let pos_zsec = s.find("[zsec]").unwrap();
        assert!(pos_asec < pos_zsec);
        let pos_a = s.find("a = 1").unwrap();
        let pos_b = s.find("b = 2").unwrap();
        assert!(pos_a < pos_b);
    }

    #[test]
    fn roundtrip_parse_merge_serialize() {
        let raw = "[display]\na=1\nb=2\n";
        let (mut doc, _) = parse_ini_document(raw);
        doc.set("display", "c", "3");
        let s = doc.serialize();
        let (doc3, w) = parse_ini_document(&s);
        assert!(w.is_empty());
        assert_eq!(doc3.get("display", "a"), Some("1"));
        assert_eq!(doc3.get("display", "c"), Some("3"));
    }
}
