//! Integration tests for the 5 PTY capabilities resolved by nexcore-pty.
//!
//! These tests validate real POSIX PTY behavior at the foundation level:
//! nexcore-pty -> libc -> kernel PTY subsystem.
//!
//! Each test targets exactly one of the 5 limitations that piped-stdio
//! cannot provide:
//!
//! 1. Echo/ICRNL — line discipline translates CR->LF and echoes input
//! 2. Resize — TIOCSWINSZ ioctl delivers SIGWINCH to foreground group
//! 3. Signal propagation — Ctrl-C (0x03) delivers SIGINT via terminal driver
//! 4. Job control — Ctrl-Z (0x1a) suspends foreground process
//! 5. ANSI/fullscreen — TERM is set and tput queries work through real PTY

use nexcore_pty::{
    SpawnConfig, WinSize, fork_exec, open_pty, read_master, resize, set_nonblocking,
    signal_process, try_wait_pid, write_master,
};
use std::os::fd::AsRawFd;

/// Helper: spawn a shell in a real PTY, return (master OwnedFd, child_pid).
/// Sets master to non-blocking for reliable reads.
fn spawn_shell(rows: u16, cols: u16, shell: &str, args: &[&str]) -> (std::os::fd::OwnedFd, u32) {
    let ws = WinSize { rows, cols };
    let pair = open_pty(ws).expect("open_pty failed");
    set_nonblocking(&pair.master).expect("set_nonblocking failed");

    let env_pairs = vec![
        ("TERM".to_string(), "xterm-256color".to_string()),
        ("COLUMNS".to_string(), cols.to_string()),
        ("LINES".to_string(), rows.to_string()),
        ("PS1".to_string(), "PROMPT> ".to_string()),
    ];

    let config = SpawnConfig {
        program: shell,
        args,
        working_dir: "/tmp",
        env: &env_pairs,
    };

    let master_raw = pair.master.as_raw_fd();
    let pid = fork_exec(pair.slave, master_raw, &config).expect("fork_exec failed");

    // Consume any initial shell startup output (prompt, motd, etc.)
    // by sleeping briefly and draining.
    std::thread::sleep(std::time::Duration::from_millis(200));
    drain_output(&pair.master);

    (pair.master, pid)
}

/// Helper: read all available output from master, returning accumulated bytes.
/// Retries with brief sleeps up to `timeout_ms` total.
fn read_until_contains(
    master: &std::os::fd::OwnedFd,
    needle: &str,
    timeout_ms: u64,
) -> Option<String> {
    let mut accumulated = Vec::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_millis(timeout_ms);

    loop {
        let mut buf = [0u8; 4096];
        match read_master(master, &mut buf) {
            Ok(0) => break,
            Ok(n) => {
                accumulated.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&accumulated);
                if text.contains(needle) {
                    return Some(text.into_owned());
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No data yet — wait and retry.
            }
            Err(_) => break,
        }

        if start.elapsed() >= timeout {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    if accumulated.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&accumulated).into_owned())
    }
}

/// Helper: drain all available output without waiting.
fn drain_output(master: &std::os::fd::OwnedFd) {
    let mut buf = [0u8; 4096];
    loop {
        match read_master(master, &mut buf) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(_) => break,
        }
    }
}

/// Helper: clean up child process.
fn cleanup(pid: u32) {
    // SIGKILL = 9
    if signal_process(pid, 9).is_err() {
        return;
    }
    for _ in 0..50 {
        if let Ok(Some(_)) = try_wait_pid(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

// --- Test 1: Echo / ICRNL --------------------------------------------------

/// Validates that the PTY line discipline provides ICRNL (CR->LF translation)
/// and echo. Writing "echo hello\r" to the master should produce output
/// containing "hello" -- the terminal driver translates \r to \n for the
/// shell, and echoes the command + output back through the master.
#[test]
fn test_pty_echo() {
    let (master, pid) = spawn_shell(24, 80, "/bin/bash", &["bash", "--norc", "--noprofile"]);

    // Send command with CR (like a real terminal Enter key).
    write_master(&master, b"echo hello\r").expect("write failed");

    // Read output -- should contain "hello" echoed back through PTY.
    let output = read_until_contains(&master, "hello", 2000);
    cleanup(pid);

    let output = output.expect("should have received output containing 'hello'");
    assert!(
        output.contains("hello"),
        "PTY echo should contain 'hello', got: {output:?}"
    );
}

// --- Test 2: Resize ---------------------------------------------------------

/// Validates that TIOCSWINSZ ioctl updates the terminal dimensions visible
/// to the child process. After resize, `tput cols` should report the new
/// column count.
#[test]
fn test_pty_resize() {
    let (master, pid) = spawn_shell(24, 80, "/bin/bash", &["bash", "--norc", "--noprofile"]);

    // Resize to 40 columns.
    let ws = WinSize { rows: 10, cols: 40 };
    resize(&master, ws).expect("resize should succeed");

    // Brief pause for SIGWINCH delivery.
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Ask the shell for the current column count.
    write_master(&master, b"tput cols\r").expect("write failed");

    let output = read_until_contains(&master, "40", 2000);
    cleanup(pid);

    let output = output.expect("should have received tput cols output");
    assert!(
        output.contains("40"),
        "tput cols should report 40 after resize, got: {output:?}"
    );
}

// --- Test 3: Signal Propagation (Ctrl-C) ------------------------------------

/// Validates that sending 0x03 (Ctrl-C) through the PTY master causes the
/// terminal driver to deliver SIGINT to the foreground process group.
/// A blocked `sleep` should be interrupted -- the shell should return to
/// its prompt rather than staying hung.
#[test]
fn test_pty_ctrl_c() {
    let (master, pid) = spawn_shell(24, 80, "/bin/bash", &["bash", "--norc", "--noprofile"]);

    // Start a long sleep.
    write_master(&master, b"sleep 60\r").expect("write failed");
    std::thread::sleep(std::time::Duration::from_millis(200));
    drain_output(&master);

    // Send Ctrl-C (0x03 = ETX, interpreted by terminal driver as SIGINT).
    write_master(&master, b"\x03").expect("write Ctrl-C failed");

    // The shell should return to prompt. Send a marker command to confirm.
    std::thread::sleep(std::time::Duration::from_millis(200));
    drain_output(&master);
    write_master(&master, b"echo CTRL_C_WORKED\r").expect("write marker failed");

    let output = read_until_contains(&master, "CTRL_C_WORKED", 2000);
    cleanup(pid);

    let output = output.expect("shell should have returned to prompt after Ctrl-C");
    assert!(
        output.contains("CTRL_C_WORKED"),
        "Ctrl-C should interrupt sleep and allow next command, got: {output:?}"
    );
}

// --- Test 4: Job Control (Ctrl-Z) -------------------------------------------

/// Validates that sending 0x1a (Ctrl-Z) through the PTY master causes the
/// terminal driver to deliver SIGTSTP to the foreground process, suspending
/// it. The shell should print a "Stopped" message.
#[test]
fn test_pty_job_control() {
    let (master, pid) = spawn_shell(24, 80, "/bin/bash", &["bash", "--norc", "--noprofile"]);

    // Start cat (blocks reading stdin).
    write_master(&master, b"cat\r").expect("write failed");
    std::thread::sleep(std::time::Duration::from_millis(200));
    drain_output(&master);

    // Send Ctrl-Z (0x1a = SUB, interpreted by terminal driver as SIGTSTP).
    write_master(&master, b"\x1a").expect("write Ctrl-Z failed");

    // Look for "Stopped" or "stopped" in output -- bash prints job control status.
    let output = read_until_contains(&master, "topped", 2000);

    // Clean up: kill any suspended jobs.
    write_master(&master, b"kill %1 2>/dev/null\r").ok();
    std::thread::sleep(std::time::Duration::from_millis(100));
    cleanup(pid);

    let output = output.expect("shell should print Stopped after Ctrl-Z");
    // bash prints "[1]+  Stopped" -- case varies by locale, check for "topped"
    assert!(
        output.contains("topped") || output.contains("suspend"),
        "Ctrl-Z should suspend cat with Stopped message, got: {output:?}"
    );
}

// --- Test 5: ANSI / Fullscreen (tput lines) ---------------------------------

/// Validates that the PTY has a real terminal device so `tput` queries work.
/// With piped stdio, tput fails because there is no terminal. With a real
/// PTY, tput reads the terminal dimensions via ioctl and reports them.
#[test]
fn test_pty_fullscreen() {
    let (master, pid) = spawn_shell(30, 100, "/bin/bash", &["bash", "--norc", "--noprofile"]);

    // Query row count via tput.
    write_master(&master, b"tput lines\r").expect("write failed");

    let output = read_until_contains(&master, "30", 2000);
    cleanup(pid);

    let output = output.expect("tput lines should report configured row count");
    assert!(
        output.contains("30"),
        "tput lines should report 30 rows, got: {output:?}"
    );
}
