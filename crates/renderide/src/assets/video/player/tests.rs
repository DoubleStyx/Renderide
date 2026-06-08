//! Tests for video player helper logic.

use super::*;

use super::super::clock::max_seek_drift_seconds;
use super::super::source::is_uri_source;
fn update(position: f64, play: bool) -> VideoTextureUpdate {
    VideoTextureUpdate {
        position,
        play,
        ..VideoTextureUpdate::default()
    }
}

fn audio_track(index: i32) -> VideoAudioTrack {
    VideoAudioTrack {
        index,
        channel_count: 2,
        sample_rate: DEFAULT_AUDIO_SAMPLE_RATE,
        name: None,
        language_code: None,
    }
}

fn ready_with_tracks(audio_tracks: Vec<VideoAudioTrack>) -> VideoTextureReady {
    VideoTextureReady {
        length: 12.5,
        size: IVec2::new(640, 480),
        has_alpha: false,
        playback_engine: Some(String::from("GStreamer test")),
        instance_changed: true,
        audio_tracks,
        asset_id: 42,
    }
}

#[test]
fn invalid_audio_sample_rate_uses_default() {
    assert_eq!(normalized_audio_sample_rate(0), DEFAULT_AUDIO_SAMPLE_RATE);
    assert_eq!(
        normalized_audio_sample_rate(-44_100),
        DEFAULT_AUDIO_SAMPLE_RATE
    );
    assert_eq!(normalized_audio_sample_rate(44_100), 44_100);
}

#[test]
fn audio_track_index_rejects_negative_values() {
    assert_eq!(validated_audio_track_index(-1), None);
    assert_eq!(validated_audio_track_index(0), Some(0));
    assert_eq!(validated_audio_track_index(2), Some(2));
}

#[test]
fn ready_message_comparison_rejects_different_track_counts() {
    let first = ready_with_tracks(vec![audio_track(0)]);
    let second = ready_with_tracks(vec![audio_track(0), audio_track(1)]);

    assert!(!video_texture_ready_eq(&first, &second));
}

#[test]
fn ready_message_comparison_rejects_playback_engine_changes() {
    let first = ready_with_tracks(vec![audio_track(0)]);
    let mut second = ready_with_tracks(vec![audio_track(0)]);
    second.playback_engine = Some(String::from("Other engine"));

    assert!(!video_texture_ready_eq(&first, &second));
}

#[test]
fn seek_threshold_is_tighter_when_paused() {
    assert!(!should_seek_to_host_position(10.5, &update(10.0, true)));
    assert!(should_seek_to_host_position(10.5, &update(10.0, false)));
}

#[test]
fn max_seek_drift_tracks_play_state() {
    assert_eq!(max_seek_drift_seconds(&update(0.0, true)), 1.0);
    assert_eq!(max_seek_drift_seconds(&update(0.0, false)), 0.01);
}

#[test]
fn target_state_tracks_play_state() {
    assert_eq!(
        target_state_for_update(&update(0.0, true)),
        gstreamer::State::Playing
    );
    assert_eq!(
        target_state_for_update(&update(0.0, false)),
        gstreamer::State::Paused
    );
}

#[test]
fn clock_time_from_seconds_clamps_large_values() {
    assert_eq!(clock_time_from_seconds(f64::MAX), gstreamer::ClockTime::MAX);
}

#[test]
fn clock_time_from_seconds_clamps_invalid_values_to_zero() {
    assert_eq!(
        clock_time_from_seconds(f64::NAN),
        gstreamer::ClockTime::ZERO
    );
    assert_eq!(clock_time_from_seconds(-1.0), gstreamer::ClockTime::ZERO);
    assert_eq!(
        clock_time_from_seconds(1.25),
        gstreamer::ClockTime::from_nseconds(1_250_000_000)
    );
}

#[test]
fn finite_media_duration_reports_ready_length_and_clock_error() {
    let duration =
        MediaDuration::from_clock_time(Some(gstreamer::ClockTime::from_nseconds(3_000_000_000)));

    assert_eq!(duration.ready_length_seconds(), 3.0);
    assert!(duration.supports_timeline_seeking());
    assert!(duration.reports_clock_error());
}

#[test]
fn ready_message_comparison_accepts_identical_tracks() {
    let first = ready_with_tracks(vec![audio_track(0), audio_track(1)]);
    let second = ready_with_tracks(vec![audio_track(0), audio_track(1)]);

    assert!(video_texture_ready_eq(&first, &second));
}

#[test]
fn ready_message_comparison_uses_float_bits_for_length() {
    let first = VideoTextureReady {
        length: 0.0,
        ..ready_with_tracks(Vec::new())
    };
    let second = VideoTextureReady {
        length: -0.0,
        ..ready_with_tracks(Vec::new())
    };

    assert!(!video_texture_ready_eq(&first, &second));
}

#[test]
fn missing_media_duration_is_stream_length_without_clock_error() {
    let duration = MediaDuration::from_clock_time(None);

    assert!(duration.ready_length_seconds().is_infinite());
    assert!(!duration.supports_timeline_seeking());
    assert!(!duration.reports_clock_error());
}

#[test]
fn zero_media_duration_is_stream_length_without_clock_error() {
    let duration = MediaDuration::from_clock_time(Some(gstreamer::ClockTime::ZERO));

    assert!(duration.ready_length_seconds().is_infinite());
    assert!(!duration.supports_timeline_seeking());
    assert!(!duration.reports_clock_error());
}

#[test]
fn uri_sources_pass_through_without_file_conversion() {
    assert!(is_uri_source("https://example.invalid/video.mp4"));
    assert!(is_uri_source("file:///tmp/video.mp4"));
    assert!(!is_uri_source("/tmp/video.mp4"));
}

fn update_decoded_at(position: f64, play: bool, decoded_nanos: i128) -> VideoTextureUpdate {
    VideoTextureUpdate {
        position,
        play,
        decoded_time: decoded_nanos,
        ..VideoTextureUpdate::default()
    }
}

const HALF_SECOND_NS: i128 = 500_000_000;
const ONE_SECOND_NS: i128 = 1_000_000_000;

#[test]
fn adjusted_host_position_advances_unconditionally_when_playing() {
    let u = update_decoded_at(10.0, true, 0);
    assert!((adjusted_host_position(&u, HALF_SECOND_NS) - 10.5).abs() < 1e-9);
}

#[test]
fn adjusted_host_position_advances_even_when_paused() {
    // Mirrors C# `VideoTextureUpdate.AdjustedPosition`, which has no play-state guard. The
    // host re-sends paused updates frequently so elapsed-since-decoded stays bounded.
    let u = update_decoded_at(10.0, false, 0);
    assert!((adjusted_host_position(&u, HALF_SECOND_NS) - 10.5).abs() < 1e-9);
}

#[test]
fn adjusted_host_position_zero_elapsed_returns_position() {
    let u = update_decoded_at(7.25, true, ONE_SECOND_NS);
    assert_eq!(adjusted_host_position(&u, ONE_SECOND_NS), 7.25);
}

#[test]
fn adjusted_host_position_handles_negative_elapsed() {
    // If wall-clock goes backwards, elapsed becomes negative and the adjusted position retreats,
    // matching the host tick contract.
    let u = update_decoded_at(4.0, true, ONE_SECOND_NS);
    let earlier = ONE_SECOND_NS - HALF_SECOND_NS;
    assert!((adjusted_host_position(&u, earlier) - 3.5).abs() < 1e-9);
}
