//! TOML compatibility and one-shot schema migrations applied before figment extraction.
//!
//! [`run_pipeline_tolerating_toml`] is the entry point: it parses the on-disk TOML into a
//! [`toml_edit::DocumentMut`], applies versioned migrations, then runs the canonical pipeline.
//! When figment rejects a stray key, the offending path is dropped and the pipeline retries up
//! to [`MAX_COMPATIBILITY_DROPS`] times so renderer downgrades and partial-key forwards work.

use std::path::Path;

use toml_edit::{DocumentMut, Item, value};

use crate::config::types::RendererSettings;

use super::pipeline::run_pipeline;

pub(super) const MAX_COMPATIBILITY_DROPS: usize = 64;

#[derive(Debug)]
pub(super) struct ConfigCompatibilityDrop {
    pub path: String,
    pub value: String,
    pub error: String,
}

#[derive(Debug)]
pub(super) struct ToleratedTomlLoad {
    pub settings: RendererSettings,
    pub drops: Vec<ConfigCompatibilityDrop>,
    pub migrated_toml: Option<String>,
}

pub(super) fn run_pipeline_tolerating_toml(
    toml_content: &str,
) -> Result<ToleratedTomlLoad, Box<figment::Error>> {
    let Ok(mut document) = toml_content.parse::<DocumentMut>() else {
        return run_pipeline(Some(toml_content.to_string())).map(|settings| ToleratedTomlLoad {
            settings,
            drops: vec![],
            migrated_toml: None,
        });
    };

    let migrated_toml = migrate_unversioned_config(&mut document).then(|| document.to_string());

    let mut drops = Vec::new();
    for _ in 0..MAX_COMPATIBILITY_DROPS {
        match run_pipeline(Some(document.to_string())) {
            Ok(settings) => {
                return Ok(ToleratedTomlLoad {
                    settings,
                    drops,
                    migrated_toml,
                });
            }
            Err(e) => {
                let Some(path) = compatibility_error_path(&e) else {
                    return Err(e);
                };
                let Some(removed) = remove_document_path(&mut document, &path) else {
                    return Err(e);
                };
                drops.push(ConfigCompatibilityDrop {
                    path: path.join("."),
                    value: summarize_removed_item(&removed),
                    error: e.to_string(),
                });
            }
        }
    }

    run_pipeline(Some(document.to_string())).map(|settings| ToleratedTomlLoad {
        settings,
        drops,
        migrated_toml,
    })
}

fn migrate_unversioned_config(document: &mut DocumentMut) -> bool {
    if document.get("config_version").is_some() {
        return false;
    }

    document.as_table_mut().insert(
        "config_version",
        value(RendererSettings::CURRENT_CONFIG_VERSION),
    );
    true
}

fn compatibility_error_path(error: &figment::Error) -> Option<Vec<String>> {
    let path = error
        .path
        .iter()
        .filter(|segment| segment.as_str() != "default")
        .cloned()
        .collect::<Vec<_>>();
    if path.is_empty() { None } else { Some(path) }
}

fn remove_document_path(document: &mut DocumentMut, path: &[String]) -> Option<Item> {
    let (last, parents) = path.split_last()?;
    let mut table = document.as_table_mut();
    for segment in parents {
        table = table.get_mut(segment)?.as_table_mut()?;
    }
    table.remove(last)
}

fn summarize_removed_item(item: &Item) -> String {
    const MAX_LEN: usize = 160;
    let mut text = item.to_string().replace(['\n', '\r'], " ");
    text = text.trim().to_string();
    if text.len() > MAX_LEN {
        text.truncate(MAX_LEN);
        text.push_str("...");
    }
    text
}

pub(super) fn log_compatibility_drops(path: &Path, drops: &[ConfigCompatibilityDrop]) {
    for drop in drops {
        logger::warn!(
            "Ignoring incompatible renderer config key {} in {}: {} ({})",
            drop.path,
            path.display(),
            drop.value,
            drop.error
        );
    }
}
