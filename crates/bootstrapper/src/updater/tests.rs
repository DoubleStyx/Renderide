//! Unit tests for release filtering, manifests, and skip state.

use self_update::update::{Release, ReleaseAsset};

use super::*;

fn metadata() -> ReleaseBuildMetadata {
    ReleaseBuildMetadata {
        channel: RELEASE_CHANNEL.to_owned(),
        tag: "nightly-2026-05-26-1111111".to_owned(),
        commit: "1111111111111111111111111111111111111111".to_owned(),
        platform: "linux-x86_64".to_owned(),
    }
}

fn release(tag: &str, commit: &str, asset_name: &str) -> Release {
    Release {
        name: tag.to_owned(),
        version: tag.to_owned(),
        date: "2026-05-27T00:00:00Z".to_owned(),
        body: Some(format!("Commit: {commit}\n")),
        assets: vec![ReleaseAsset {
            name: asset_name.to_owned(),
            download_url: "https://api.github.com/repos/DoubleStyx/Renderide/releases/assets/1"
                .to_owned(),
        }],
    }
}

#[test]
fn full_sha_validation_requires_forty_hex_chars() {
    assert!(is_full_sha("0123456789abcdef0123456789ABCDEF01234567"));
    assert!(!is_full_sha("0123456789abcdef"));
    assert!(!is_full_sha("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"));
}

#[test]
fn release_commit_parses_first_line_shape() {
    let commit = "2222222222222222222222222222222222222222";
    assert_eq!(
        release_commit(Some(&format!("Commit: {commit}\n\nbody"))),
        Some(commit.to_owned())
    );
    assert_eq!(release_commit(Some("Commit: bad")), None);
    assert_eq!(release_commit(None), None);
}

#[test]
fn candidate_selection_uses_nightly_tag_and_exact_platform_asset() {
    let metadata = metadata();
    let commit = "2222222222222222222222222222222222222222";
    let releases = vec![
        release("v1.0.0", commit, "renderide-linux-x86_64-v1.0.0.zip"),
        release(
            "nightly-2026-05-27-2222222",
            commit,
            "renderide-linux-x86_64-nightly-2026-05-27-2222222.zip",
        ),
    ];

    let candidate = select_update_candidate(&releases, &metadata);

    match candidate {
        Some(candidate) => {
            assert_eq!(candidate.tag, "nightly-2026-05-27-2222222");
            assert_eq!(candidate.commit, commit);
        }
        None => panic!("expected update candidate"),
    }
}

#[test]
fn candidate_selection_ignores_current_tag() {
    let metadata = metadata();
    let releases = vec![release(
        &metadata.tag,
        "2222222222222222222222222222222222222222",
        &asset_name_for(&metadata.platform, &metadata.tag),
    )];

    assert!(select_update_candidate(&releases, &metadata).is_none());
}

#[test]
fn skipped_release_state_parses_tag() {
    assert_eq!(
        skipped_release_tag_from_contents("skip_release_tag=nightly-1\n"),
        Some("nightly-1".to_owned())
    );
    assert_eq!(skipped_release_tag_from_contents("other=value\n"), None);
}

#[test]
fn unsafe_zip_paths_are_rejected() {
    for name in ["/abs", "../up", "root/../up", "root\\file", "C:/file"] {
        assert!(
            validate_zip_entry_name(name).is_err(),
            "{name} should be rejected"
        );
    }
    assert!(validate_zip_entry_name("renderide-linux/foo").is_ok());
    assert!(validate_zip_entry_name("renderide-linux/xr/").is_ok());
}

#[test]
fn manifest_validation_requires_expected_fields() {
    let metadata = metadata();
    let candidate = UpdateCandidate {
        tag: "nightly-2026-05-27-2222222".to_owned(),
        commit: "2222222222222222222222222222222222222222".to_owned(),
        asset: ReleaseAsset {
            name: asset_name_for(&metadata.platform, "nightly-2026-05-27-2222222"),
            download_url: String::new(),
        },
    };
    let text = serde_json::json!({
        "schema": 1,
        "channel": RELEASE_CHANNEL,
        "tag": candidate.tag,
        "commit": candidate.commit,
        "platform": metadata.platform,
        "required_files": ["renderide", "renderide-renderer", "xr"]
    })
    .to_string();

    assert!(validate_manifest(&text, &metadata, &candidate).is_ok());
}
