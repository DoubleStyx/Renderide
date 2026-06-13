//! Source audit: every renderer config setting must be editable in the renderer-config HUD.
//!
//! The project contract is that a config setting lives in both the config schema and the ImGui
//! renderer-config window. This audit serializes [`RendererSettings::default`] to enumerate every
//! persisted key, subtracts the deliberate HUD-UI-state exemptions, and requires each remaining
//! field name to appear in the `diagnostics/hud/windows/renderer_config/` sources.

use std::fs;
use std::path::{Path, PathBuf};

use hashbrown::HashMap;

use super::types::RendererSettings;

/// Dotted config key prefixes that are intentionally not edited through the renderer-config HUD.
///
/// Extend this list only for keys that are genuinely HUD/UI state persisted alongside settings;
/// a real renderer setting belongs in a `renderer_config` window instead.
const HUD_STATE_EXEMPT_PREFIXES: &[&str] = &[
    // Schema version stamp written by the renderer itself.
    "config_version",
    // Debug HUD window/tab state, edited by interacting with the HUD chrome directly.
    "debug.hud.renderer_config_open",
    "debug.hud.scene_transforms_open",
    "debug.hud.texture_debug_open",
    "debug.hud.texture_debug_current_view_only",
    "debug.hud.draw_state_ui_only",
    "debug.hud.draw_state_only_overrides",
    "debug.hud.shader_routes_only_fallback",
    "debug.hud.main_tab",
    "debug.hud.main_tabs",
    "debug.hud.renderer_config_tab",
    "debug.hud.renderer_config_tabs",
    "debug.hud.scene_transforms_space_id",
];

/// Returns the renderide crate directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Flattens a TOML document into dotted leaf key paths.
fn flatten_keys(prefix: &str, value: &toml::Value, out: &mut Vec<String>) {
    match value {
        toml::Value::Table(table) => {
            for (key, child) in table {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_keys(&path, child, out);
            }
        }
        _ => out.push(prefix.to_owned()),
    }
}

/// Maps serialized TOML key names back to Rust field names for `#[serde(rename = "...")]` fields.
fn serde_rename_map() -> HashMap<String, String> {
    let mut sources = Vec::new();
    collect_rust_sources(&manifest_dir().join("src/config/types"), &mut sources);
    sources.push(manifest_dir().join("src/config/types.rs"));

    let mut renames = HashMap::new();
    for path in sources {
        let text = fs::read_to_string(&path).expect("read config type source");
        let mut pending_rename: Option<String> = None;
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("#[serde(rename = \"") {
                if let Some(end) = rest.find('"') {
                    pending_rename = Some(rest[..end].to_owned());
                }
                continue;
            }
            if let Some(toml_name) = pending_rename.take() {
                if let Some(rest) = trimmed.strip_prefix("pub ") {
                    let field: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if !field.is_empty() {
                        renames.insert(toml_name, field);
                    }
                } else if trimmed.starts_with("#[") || trimmed.starts_with("///") {
                    // Attribute or doc line between the rename and the field; keep waiting.
                    pending_rename = Some(toml_name);
                }
            }
        }
    }
    renames
}

/// Recursively collects `.rs` files below `dir`.
fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Concatenates every renderer-config HUD window source.
fn renderer_config_hud_sources() -> String {
    let dir = manifest_dir().join("src/diagnostics/hud/windows/renderer_config");
    let mut sources = Vec::new();
    collect_rust_sources(&dir, &mut sources);
    assert!(
        !sources.is_empty(),
        "no HUD sources found under {}",
        dir.display()
    );
    sources.sort();
    sources
        .iter()
        .map(|path| fs::read_to_string(path).expect("read HUD source"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns whether `needle` appears in `haystack` as a whole identifier.
fn identifier_present(haystack: &str, needle: &str) -> bool {
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let bytes = haystack.as_bytes();
    let mut search_from = 0;
    while let Some(found) = haystack[search_from..].find(needle) {
        let start = search_from + found;
        let end = start + needle.len();
        let boundary_before = start == 0 || !is_ident(bytes[start - 1]);
        let boundary_after = end == bytes.len() || !is_ident(bytes[end]);
        if boundary_before && boundary_after {
            return true;
        }
        search_from = start + 1;
    }
    false
}

#[test]
fn every_config_setting_is_exposed_in_the_renderer_config_hud() {
    let document =
        toml::Value::try_from(RendererSettings::default()).expect("serialize default settings");
    let mut keys = Vec::new();
    flatten_keys("", &document, &mut keys);
    assert!(!keys.is_empty(), "no config keys serialized");

    let renames = serde_rename_map();
    let hud = renderer_config_hud_sources();
    let mut missing = Vec::new();
    for key in &keys {
        if HUD_STATE_EXEMPT_PREFIXES
            .iter()
            .any(|prefix| key == prefix || key.starts_with(&format!("{prefix}.")))
        {
            continue;
        }
        let leaf = key.rsplit('.').next().unwrap_or(key);
        let field = renames.get(leaf).map(String::as_str).unwrap_or(leaf);
        if !identifier_present(&hud, field) {
            missing.push(format!("{key} (field `{field}`)"));
        }
    }

    assert!(
        missing.is_empty(),
        "config settings missing from the renderer-config HUD: {missing:#?}\n\
         Add each field to a window under src/diagnostics/hud/windows/renderer_config/ or, for \
         genuine HUD UI state, extend HUD_STATE_EXEMPT_PREFIXES with a justification."
    );
}

#[test]
fn serde_rename_map_resolves_display_fps_fields() {
    let renames = serde_rename_map();

    assert_eq!(
        renames.get("focused_fps").map(String::as_str),
        Some("focused_fps_cap")
    );
    assert_eq!(
        renames.get("unfocused_fps").map(String::as_str),
        Some("unfocused_fps_cap")
    );
}
