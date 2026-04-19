//! Integration: [`bootstrapper::ipc::BootstrapQueues`] open/dispose against an isolated backing directory.

use std::sync::Mutex;

use bootstrapper::ipc::{
    bootstrap_queue_base_names, BootstrapQueues, RENDERIDE_INTERPROCESS_DIR_ENV,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Opens the bootstrapper queue pair with `RENDERIDE_INTERPROCESS_DIR` set to a temp folder, then
/// drops the handles so `destroy_on_dispose` removes backing storage where applicable.
#[test]
fn bootstrap_queues_open_drop_succeeds_with_env_override() {
    let _guard = ENV_LOCK.lock().expect("env lock");

    let tmp = std::env::temp_dir().join(format!("bootstrapper_it_ipc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");

    std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);

    let prefix = format!("it{}", std::process::id());
    let (in_name, out_name) = bootstrap_queue_base_names(&prefix);

    {
        let _queues = BootstrapQueues::open(&prefix).expect("open bootstrap queues");
        let in_qu = tmp.join(format!("{in_name}.qu"));
        let out_qu = tmp.join(format!("{out_name}.qu"));
        #[cfg(unix)]
        {
            assert!(
                in_qu.exists() || out_qu.exists(),
                "expected at least one .qu under {:?}",
                tmp
            );
        }
        #[cfg(windows)]
        {
            let _ = (in_qu, out_qu);
        }
    }

    #[cfg(unix)]
    {
        assert!(
            !tmp.join(format!("{in_name}.qu")).exists(),
            "incoming .qu should be removed on drop"
        );
        assert!(
            !tmp.join(format!("{out_name}.qu")).exists(),
            "outgoing .qu should be removed on drop"
        );
    }

    std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
    let _ = std::fs::remove_dir_all(&tmp);
}

/// [`bootstrap_queue_base_names`] is stable for Host alignment; exercise it from outside the crate.
#[test]
fn queue_base_names_match_expected_suffixes() {
    let (a, b) = bootstrap_queue_base_names("Z9");
    assert_eq!(a, "Z9.bootstrapper_in");
    assert_eq!(b, "Z9.bootstrapper_out");
}

/// `incoming` and `outgoing` are different queue names; still verify the publisher accepts a write.
#[test]
fn outgoing_try_enqueue_succeeds_on_open_pair() {
    let _guard = ENV_LOCK.lock().expect("env lock");

    let tmp = std::env::temp_dir().join(format!("bootstrapper_it_rt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");
    std::env::set_var(RENDERIDE_INTERPROCESS_DIR_ENV, &tmp);

    let prefix = format!("rt{}", std::process::id());
    let mut queues = BootstrapQueues::open(&prefix).expect("open");

    assert!(
        queues.outgoing.try_enqueue(b"integration"),
        "enqueue should succeed on fresh outgoing queue"
    );

    std::env::remove_var(RENDERIDE_INTERPROCESS_DIR_ENV);
    let _ = std::fs::remove_dir_all(&tmp);
}
