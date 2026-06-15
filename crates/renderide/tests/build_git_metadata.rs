//! Exercises the build-time commit metadata helpers under `cargo test`.

#![allow(
    dead_code,
    reason = "the path-included build-script module exposes helpers outside these focused tests"
)]
#![allow(
    clippy::print_stdout,
    reason = "the included build-script module emits Cargo directives through println!"
)]

#[path = "../build_support/git.rs"]
mod git;

#[test]
fn full_commit_to_short_accepts_full_hex_sha() {
    assert_eq!(
        git::full_commit_to_short("03B605ADE0000000000000000000000000000000"),
        Some("03b605ad".to_string())
    );
}

#[test]
fn full_commit_to_short_rejects_invalid_sha() {
    assert_eq!(git::full_commit_to_short("03b605ad"), None);
    assert_eq!(
        git::full_commit_to_short("03b605adzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"),
        None
    );
    assert_eq!(git::full_commit_to_short(""), None);
}
