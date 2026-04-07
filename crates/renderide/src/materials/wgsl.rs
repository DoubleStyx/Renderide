//! Ordered, deterministic WGSL source patching (permutation / material overrides).

/// One transform applied in sequence to a WGSL template.
#[derive(Clone, Copy, Debug)]
pub enum WgslPatch {
    /// Replaces the first occurrence of `needle` with `replacement` (exact substring).
    ReplaceFirst {
        /// Unique marker in the template (for example `// @MATERIAL_FRAG_BODY`).
        needle: &'static str,
        /// Replacement WGSL source (no automatic newline insertion).
        replacement: &'static str,
    },
}

/// Applies `patches` to `base` in order. Missing needles leave the source unchanged for that patch.
pub fn compose_wgsl(base: &str, patches: &[WgslPatch]) -> String {
    let mut s = base.to_string();
    for patch in patches {
        match *patch {
            WgslPatch::ReplaceFirst {
                needle,
                replacement,
            } => {
                if let Some(i) = s.find(needle) {
                    s.replace_range(i..i + needle.len(), replacement);
                }
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::{compose_wgsl, WgslPatch};

    #[test]
    fn replace_first_applies_in_order() {
        let base = "A /*M1*/ B /*M2*/";
        let out = compose_wgsl(
            base,
            &[
                WgslPatch::ReplaceFirst {
                    needle: "/*M1*/",
                    replacement: "one",
                },
                WgslPatch::ReplaceFirst {
                    needle: "/*M2*/",
                    replacement: "two",
                },
            ],
        );
        assert_eq!(out, "A one B two");
    }

    #[test]
    fn missing_needle_skips_patch() {
        let base = "hello";
        let out = compose_wgsl(
            base,
            &[WgslPatch::ReplaceFirst {
                needle: "nope",
                replacement: "x",
            }],
        );
        assert_eq!(out, "hello");
    }
}
