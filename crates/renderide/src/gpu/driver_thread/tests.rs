//! Unit tests for the driver thread's ring and payload wiring.
//!
//! The driver thread itself is not exercised here because spawning it requires a real
//! `wgpu::Queue`; integration tests in `renderide-test` cover the full path.

use std::time::Duration;

use super::ring::BoundedRing;
use super::submit_batch::SubmitWait;

/// Push blocks when the ring is full and wakes once a pop makes space available.
#[test]
fn ring_blocks_producer_when_full_and_wakes_on_pop() {
    let ring: BoundedRing<u32> = BoundedRing::new(2);
    ring.push(1);
    ring.push(2);
    // Ring is now full; consume on a worker so the producer can proceed.
    let ring_arc = std::sync::Arc::new(ring);
    let ring_for_consumer = std::sync::Arc::clone(&ring_arc);

    let handle = std::thread::spawn(move || {
        // Give the main thread time to block on push(3) before popping.
        std::thread::sleep(Duration::from_millis(50));
        let popped = ring_for_consumer.pop();
        assert_eq!(popped, 1);
    });

    ring_arc.push(3); // must block briefly, then succeed after the consumer pops.
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
        ring_for_producer.push(42);
    });

    // Blocks briefly, then returns 42.
    assert_eq!(ring.pop(), 42);
    handle.join().expect("producer thread joined");
}

/// Capacity 1 still functions correctly (edge case).
#[test]
fn ring_capacity_one_round_trip() {
    let ring: BoundedRing<&'static str> = BoundedRing::new(1);
    ring.push("hello");
    assert_eq!(ring.pop(), "hello");
    ring.push("world");
    assert_eq!(ring.pop(), "world");
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
