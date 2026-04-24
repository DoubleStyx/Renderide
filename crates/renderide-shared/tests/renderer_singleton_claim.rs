//! Integration: [`renderide_shared::ipc::connection::try_claim_renderer_singleton`] process guard.

use renderide_shared::ipc::connection::{try_claim_renderer_singleton, InitError};

#[test]
fn second_try_claim_returns_singleton_already_exists() {
    try_claim_renderer_singleton().expect("first claim");
    let err = try_claim_renderer_singleton().expect_err("second claim");
    assert!(matches!(err, InitError::SingletonAlreadyExists));
}
