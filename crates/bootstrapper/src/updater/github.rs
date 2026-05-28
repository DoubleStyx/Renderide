//! GitHub release selection for update candidates.

use self_update::backends::github::ReleaseList;
use self_update::update::Release;

use super::release_metadata::is_full_sha;
use super::{
    NIGHTLY_PREFIX, REPO_NAME, REPO_OWNER, ReleaseBuildMetadata, UpdateCandidate, UpdateError,
    to_self_update_error,
};

/// Fetches GitHub releases and selects the newest eligible update candidate.
pub(super) fn fetch_latest_candidate(
    metadata: &ReleaseBuildMetadata,
) -> Result<Option<UpdateCandidate>, UpdateError> {
    let releases = ReleaseList::configure()
        .repo_owner(REPO_OWNER)
        .repo_name(REPO_NAME)
        .build()
        .map_err(to_self_update_error)?
        .fetch()
        .map_err(to_self_update_error)?;
    Ok(select_update_candidate(&releases, metadata))
}

/// Selects the first eligible release asset that is not the running release tag.
pub(super) fn select_update_candidate(
    releases: &[Release],
    metadata: &ReleaseBuildMetadata,
) -> Option<UpdateCandidate> {
    releases
        .iter()
        .filter(|release| release.version.starts_with(NIGHTLY_PREFIX))
        .filter(|release| release.version != metadata.tag)
        .find_map(|release| candidate_from_release(release, metadata))
}

/// Converts a GitHub release into an update candidate when it has this platform's asset.
fn candidate_from_release(
    release: &Release,
    metadata: &ReleaseBuildMetadata,
) -> Option<UpdateCandidate> {
    let commit = release_commit(release.body.as_deref())?;
    let asset_name = asset_name_for(&metadata.platform, &release.version);
    let asset = release
        .assets
        .iter()
        .find(|asset| asset.name == asset_name)
        .cloned()?;
    Some(UpdateCandidate {
        tag: release.version.clone(),
        commit,
        asset,
    })
}

/// Builds the exact GitHub release asset name for a platform and tag.
pub(super) fn asset_name_for(platform: &str, tag: &str) -> String {
    format!("renderide-{platform}-{tag}.zip")
}

/// Parses the commit SHA recorded in a GitHub release body.
pub(super) fn release_commit(body: Option<&str>) -> Option<String> {
    body.and_then(|body| {
        body.lines().find_map(|line| {
            let commit = line.strip_prefix("Commit: ")?;
            is_full_sha(commit).then(|| commit.to_owned())
        })
    })
}

#[cfg(test)]
mod tests {
    use self_update::update::{Release, ReleaseAsset};

    use super::*;
    use crate::updater::RELEASE_CHANNEL;

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
}
