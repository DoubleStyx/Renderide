//! Verifies [`renderide_test::cli::parse_resolution`] at the public lib boundary.

use renderide_test::cli::parse_resolution;

#[test]
fn parse_resolution_round_trips_common_inputs() {
    assert_eq!(parse_resolution("1920x1080"), (1920, 1080));
    assert_eq!(parse_resolution("64x32"), (64, 32));
    assert_eq!(parse_resolution("800x600"), (800, 600));
}

#[test]
fn parse_resolution_accepts_uppercase_x() {
    assert_eq!(parse_resolution("1920X1080"), (1920, 1080));
}

#[test]
fn parse_resolution_invalid_falls_back_to_default() {
    assert_eq!(parse_resolution("abc"), (256, 256));
    assert_eq!(parse_resolution("1x"), (256, 256));
    assert_eq!(parse_resolution("1xxx2"), (256, 256));
    assert_eq!(parse_resolution(""), (256, 256));
}

#[test]
fn parse_resolution_clamps_zero_dimensions_to_one() {
    assert_eq!(parse_resolution("0x0"), (1, 1));
    assert_eq!(parse_resolution("0x64"), (1, 64));
    assert_eq!(parse_resolution("64x0"), (64, 1));
}
