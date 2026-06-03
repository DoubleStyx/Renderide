//! Unit tests for the driver thread's ring and payload wiring.
//!
//! The driver thread itself is not exercised here because spawning it requires a real
//! `wgpu::Queue`; integration tests in `renderide-test` cover the full path.

use std::time::Duration;

use super::ring::BoundedRing;
use super::submit_batch::SubmitWait;
use super::{
    DESKTOP_SUBMIT_BATCHES_PER_VISIBLE_FRAME, DRIVER_VISIBLE_FRAMES_IN_FLIGHT, RING_CAPACITY,
};

/// The driver ring is sized in visible desktop frames, not raw submit batches.
#[test]
fn ring_capacity_covers_two_desktop_visible_frames() {
    assert_eq!(DESKTOP_SUBMIT_BATCHES_PER_VISIBLE_FRAME, 2);
    assert_eq!(DRIVER_VISIBLE_FRAMES_IN_FLIGHT, 2);
    assert_eq!(
        RING_CAPACITY,
        DESKTOP_SUBMIT_BATCHES_PER_VISIBLE_FRAME * DRIVER_VISIBLE_FRAMES_IN_FLIGHT
    );
}

/// Push blocks when the ring is full and wakes once a pop makes space available.
#[test]
fn ring_blocks_producer_when_full_and_wakes_on_pop() {
    let ring: BoundedRing<u32> = BoundedRing::new(2);
    ring.push(1).expect("push succeeds while consumer is alive");
    ring.push(2).expect("push succeeds while consumer is alive");
    // Ring is now full; consume on a worker so the producer can proceed.
    let ring_arc = std::sync::Arc::new(ring);
    let ring_for_consumer = std::sync::Arc::clone(&ring_arc);

    let handle = std::thread::spawn(move || {
        // Give the main thread time to block on push(3) before popping.
        std::thread::sleep(Duration::from_millis(50));
        let popped = ring_for_consumer.pop();
        assert_eq!(popped, 1);
    });

    ring_arc
        .push(3)
        .expect("push succeeds after consumer drains slot");
    handle.join().expect("consumer thread joined");
    assert_eq!(ring_arc.pop(), 2);
    assert_eq!(ring_arc.pop(), 3);
}

/// Pop blocks when the ring is empty and wakes once a push puts an item in.
#[test]
fn ring_blocks_consumer_when_empty_and_wakes_on_push() {
    let ring: std::sync::Arc<BoundedRing<u32>> = std::sync::Arc::new(BoundedRing::new(2));
    let ring_for_producer = std::sync::Arc::clone(&ring);

    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        ring_for_producer
            .push(42)
            .expect("push succeeds while consumer is alive");
    });

    // Blocks briefly, then returns 42.
    assert_eq!(ring.pop(), 42);
    handle.join().expect("producer thread joined");
}

/// Capacity 1 still functions correctly (edge case).
#[test]
fn ring_capacity_one_round_trip() {
    let ring: BoundedRing<&'static str> = BoundedRing::new(1);
    ring.push("hello").expect("push succeeds on empty ring");
    assert_eq!(ring.pop(), "hello");
    ring.push("world").expect("push succeeds on empty ring");
    assert_eq!(ring.pop(), "world");
}

/// Once the consumer is marked dead, a producer waiting on a full ring returns the
/// pushed item rather than blocking forever.
#[test]
fn ring_push_returns_err_when_consumer_dies_while_full() {
    let ring: std::sync::Arc<BoundedRing<u32>> = std::sync::Arc::new(BoundedRing::new(1));
    ring.push(1).expect("push succeeds on empty ring");
    // Ring is full; spawn a thread that flips the liveness flag after a short delay.
    let ring_for_killer = std::sync::Arc::clone(&ring);
    let killer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        ring_for_killer.mark_consumer_dead();
    });
    // This push would block forever without the liveness wake-up.
    let pushed_item = 99u32;
    let result = ring.push(pushed_item);
    assert_eq!(result, Err(pushed_item));
    killer.join().expect("killer thread joined");
}

/// Capacity zero is rejected by `BoundedRing::new`.
#[test]
#[should_panic(expected = "capacity")]
fn ring_capacity_zero_panics() {
    let _ring: BoundedRing<u32> = BoundedRing::new(0);
}

/// `SubmitWait::signal` fires the oneshot exactly once; the receiver sees one value and
/// then observes the sender disconnecting (consuming `signal` drops the `SyncSender`).
#[test]
fn submit_wait_oneshot_fires_once() {
    let (wait, rx) = SubmitWait::new();
    wait.signal();
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_ok());
    // The sender was consumed by `signal`, so the second recv sees a disconnected channel.
    assert!(rx.recv_timeout(Duration::from_millis(50)).is_err());
}

/// Dropping the receiver before signaling must not panic; `signal` silently discards.
#[test]
fn submit_wait_signal_tolerates_dropped_receiver() {
    let (wait, rx) = SubmitWait::new();
    drop(rx);
    wait.signal(); // must not panic
}
