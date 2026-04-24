//! Integration tests using only the public `interprocess` API.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use interprocess::{Publisher, QueueOptions, Subscriber};

#[test]
fn enqueue_and_dequeue_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let opts = QueueOptions::with_path("e2e_roundtrip", dir.path(), 4096).expect("valid options");
    let mut publisher = Publisher::new(opts.clone()).expect("publisher");
    let mut subscriber = Subscriber::new(opts).expect("subscriber");

    assert!(publisher.try_enqueue(b"hello"));
    assert_eq!(
        subscriber.try_dequeue().as_deref(),
        Some(b"hello".as_slice())
    );
}

#[test]
fn subscriber_opens_before_publisher_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let name = format!("e2e_sub_first_{}", std::process::id());
    let opts = QueueOptions::with_path(&name, dir.path(), 4096).expect("valid options");
    let mut subscriber = Subscriber::new(opts.clone()).expect("subscriber");
    let mut publisher = Publisher::new(opts).expect("publisher");

    assert!(publisher.try_enqueue(b"hello"));
    assert_eq!(
        subscriber.try_dequeue().as_deref(),
        Some(b"hello".as_slice())
    );
}

#[test]
fn dequeue_unblocks_after_subscriber_first() {
    let dir = tempfile::tempdir().expect("tempdir");
    let name = format!("e2e_deq_block_{}", std::process::id());
    let opts = QueueOptions::with_path(&name, dir.path(), 4096).expect("valid");
    let opts_pub = opts.clone();
    let barrier = Arc::new(Barrier::new(2));
    let barrier_consumer = Arc::clone(&barrier);

    let consumer = thread::spawn(move || {
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        barrier_consumer.wait();
        let cancel = AtomicBool::new(false);
        subscriber.dequeue(&cancel)
    });

    barrier.wait();
    let mut publisher = Publisher::new(opts_pub).expect("publisher");
    assert!(publisher.try_enqueue(b"wake"));

    let got = consumer.join().expect("consumer join");
    assert_eq!(got, b"wake");
}

#[test]
fn multi_message_fifo_varied_sizes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let opts = QueueOptions::with_path("e2e_fifo", dir.path(), 8192).expect("valid");
    let mut publisher = Publisher::new(opts.clone()).expect("publisher");
    let mut subscriber = Subscriber::new(opts).expect("subscriber");

    let payloads: Vec<Vec<u8>> = vec![
        vec![],
        vec![1],
        vec![0u8; 7],
        vec![0u8; 8],
        vec![0u8; 9],
        vec![0xAB; 4000],
    ];

    for p in &payloads {
        assert!(publisher.try_enqueue(p.as_slice()));
    }
    for p in &payloads {
        assert_eq!(subscriber.try_dequeue().as_deref(), Some(p.as_slice()));
    }
}

#[cfg(unix)]
#[test]
fn destroy_on_dispose_unlinks_qu_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let opts =
        QueueOptions::with_path_and_destroy("e2e_destroy", dir.path(), 4096, true).expect("valid");
    let path = opts.file_path();
    let _publisher = Publisher::new(opts).expect("publisher");
    assert!(path.exists());
    drop(_publisher);
    assert!(!path.exists());
}

#[test]
fn concurrent_producer_consumer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let opts = QueueOptions::with_path("e2e_concurrent", dir.path(), 65_536).expect("valid");
    let opts_pub = opts.clone();
    let n = 50u32;
    let cancel = Arc::new(AtomicBool::new(false));

    let cancel_clone = Arc::clone(&cancel);
    let consumer = thread::spawn(move || {
        let mut subscriber = Subscriber::new(opts).expect("subscriber");
        let mut got = 0u32;
        while got < n {
            if let Some(msg) = subscriber.try_dequeue() {
                assert_eq!(msg, format!("m{got}").into_bytes());
                got += 1;
            } else if cancel_clone.load(Ordering::Relaxed) {
                break;
            } else {
                thread::sleep(Duration::from_millis(1));
            }
        }
        assert_eq!(got, n);
    });

    let mut publisher = Publisher::new(opts_pub).expect("publisher");
    for i in 0..n {
        let payload = format!("m{i}");
        while !publisher.try_enqueue(payload.as_bytes()) {
            thread::sleep(Duration::from_millis(1));
        }
    }

    consumer.join().expect("consumer join");
    cancel.store(true, Ordering::Relaxed);
}
