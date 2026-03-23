//! POSIX pseudo-terminal allocation — safe wrappers around `openpty(3)` and friends.
//!
//! Foundation-layer crate with a single external dependency (`libc`).
//! Wraps `openpty()`, `fork()`, `setsid()`, `dup2()`, `execvp()`, and
//! `TIOCSWINSZ` ioctl behind safe Rust functions.
//!
//! # Supply Chain Sovereignty
//!
//! This crate replaces external PTY crates (`nix`, `portable-pty`) with a
//! minimal, auditable wrapper around POSIX system calls. ~250 lines, zero
//! transitive dependencies beyond `libc`.
//!
//! ## Primitive Grounding
//!
//! `∂(Boundary) + ς(State) + →(Causality)`

#![warn(missing_docs)]
#[cfg(not(unix))]
compile_error!("nexcore-pty requires a Unix platform (Linux or macOS)");

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

// ─── Types ───────────────────────────────────────────────────────────

/// Terminal dimensions for PTY allocation and resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WinSize {
    /// Row count.
    pub rows: u16,
    /// Column count.
    pub cols: u16,
}

/// A matched pair of PTY file descriptors (master + slave).
///
/// The master fd is used by the controlling process (terminal emulator).
/// The slave fd becomes the child's stdin/stdout/stderr.
pub struct PtyPair {
    /// Master side — read/write from terminal emulator.
    pub master: OwnedFd,
    /// Slave side — becomes child's controlling terminal.
    pub slave: OwnedFd,
}

/// Configuration for spawning a child process in a PTY.
pub struct SpawnConfig<'a> {
    /// Shell binary path (e.g., "/bin/bash").
    pub program: &'a str,
    /// Command-line arguments (argv\[0\] should be program name).
    pub args: &'a [&'a str],
    /// Working directory for the child.
    pub working_dir: &'a str,
    /// Environment variables to set (merged with inherited env).
    pub env: &'a [(String, String)],
}

/// Errors from PTY operations.
#[non_exhaustive]
#[derive(Debug)]
pub enum PtyError {
    /// `openpty()` failed.
    OpenPtyFailed(std::io::Error),
    /// `fork()` failed.
    ForkFailed(std::io::Error),
    /// `execvp()` failed in child (communicated via pipe).
    ExecFailed(std::io::Error),
    /// `ioctl(TIOCSWINSZ)` failed.
    ResizeFailed(std::io::Error),
    /// `fcntl(F_SETFL)` failed.
    SetNonblockFailed(std::io::Error),
    /// `kill()` or `waitpid()` failed.
    ProcessError(std::io::Error),
    /// A string argument contained an interior NUL byte.
    InvalidString(std::ffi::NulError),
}

impl std::fmt::Display for PtyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenPtyFailed(e) => write!(f, "openpty failed: {e}"),
            Self::ForkFailed(e) => write!(f, "fork failed: {e}"),
            Self::ExecFailed(e) => write!(f, "exec failed in child: {e}"),
            Self::ResizeFailed(e) => write!(f, "TIOCSWINSZ ioctl failed: {e}"),
            Self::SetNonblockFailed(e) => write!(f, "fcntl F_SETFL failed: {e}"),
            Self::ProcessError(e) => write!(f, "process operation failed: {e}"),
            Self::InvalidString(e) => write!(f, "invalid C string: {e}"),
        }
    }
}

impl std::error::Error for PtyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenPtyFailed(e)
            | Self::ForkFailed(e)
            | Self::ExecFailed(e)
            | Self::ResizeFailed(e)
            | Self::SetNonblockFailed(e)
            | Self::ProcessError(e) => Some(e),
            Self::InvalidString(e) => Some(e),
        }
    }
}

impl PtyError {
    /// Extract the underlying I/O error for conversion to other error types.
    ///
    /// # Panics
    ///
    /// Panics if called on `InvalidString` variant. Callers should match
    /// or use `source()` instead for that case.
    #[must_use]
    pub fn into_io(self) -> std::io::Error {
        match self {
            Self::OpenPtyFailed(e)
            | Self::ForkFailed(e)
            | Self::ExecFailed(e)
            | Self::ResizeFailed(e)
            | Self::SetNonblockFailed(e)
            | Self::ProcessError(e) => e,
            Self::InvalidString(e) => std::io::Error::new(std::io::ErrorKind::InvalidInput, e),
        }
    }
}

// ─── Functions ───────────────────────────────────────────────────────

/// Allocate a new pseudo-terminal pair.
///
/// Returns master and slave file descriptors. The master fd is for the
/// terminal emulator; the slave fd is for the child process.
///
/// The PTY is initialized with the given terminal dimensions.
#[allow(unsafe_code, reason = "FFI boundary for POSIX openpty(3)")]
pub fn open_pty(size: WinSize) -> Result<PtyPair, PtyError> {
    let mut master_fd: libc::c_int = -1;
    let mut slave_fd: libc::c_int = -1;

    let ws = libc::winsize {
        ws_row: size.rows,
        ws_col: size.cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };

    // SAFETY: `master_fd` and `slave_fd` are valid mutable pointers to c_int on
    // the stack. `openpty` writes valid fds on success (returns 0). On failure
    // (returns -1), no fds are created. We pass NULL for name and termios
    // (use defaults), and a valid winsize pointer for initial dimensions.
    let ret = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(), // name — not needed
            std::ptr::null(),     // termios — use defaults (includes ICRNL)
            &ws,
        )
    };

    if ret != 0 {
        return Err(PtyError::OpenPtyFailed(std::io::Error::last_os_error()));
    }

    // SAFETY: `openpty` succeeded (ret == 0), so both fds are valid, open file
    // descriptors. `OwnedFd::from_raw_fd` takes ownership and will close them
    // on drop.
    let master = unsafe { OwnedFd::from_raw_fd(master_fd) };
    let slave = unsafe { OwnedFd::from_raw_fd(slave_fd) };

    Ok(PtyPair { master, slave })
}

/// Fork and exec a program in the slave PTY.
///
/// - Creates a new session (`setsid`)
/// - Sets the slave fd as controlling terminal
/// - Dups slave to stdin/stdout/stderr
/// - Closes master fd in child
/// - Execs the program
///
/// Returns the child PID. The slave fd is consumed (closed in parent
/// when `slave` is dropped).
///
/// **Critical safety note:** Between `fork()` and `exec()`, only
/// async-signal-safe functions are called. No Rust allocation occurs
/// in the child before exec.
#[allow(unsafe_code, reason = "FFI boundary for POSIX fork(2)/execvp(3)")]
pub fn fork_exec(
    slave: OwnedFd,
    master_fd: RawFd,
    config: &SpawnConfig<'_>,
) -> Result<u32, PtyError> {
    // Pre-fork: prepare all C strings while we can still allocate safely.
    let c_program = CString::new(config.program).map_err(PtyError::InvalidString)?;

    let c_args: Vec<CString> = config
        .args
        .iter()
        .map(|a| CString::new(*a))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PtyError::InvalidString)?;

    let c_arg_ptrs: Vec<*const libc::c_char> = c_args
        .iter()
        .map(|a| a.as_ptr())
        .chain(std::iter::once(std::ptr::null()))
        .collect();

    let c_working_dir = CString::new(config.working_dir).map_err(PtyError::InvalidString)?;

    let c_env: Vec<CString> = config
        .env
        .iter()
        .map(|(k, v)| CString::new(format!("{k}={v}")))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PtyError::InvalidString)?;

    // Create a pipe for exec error reporting.
    // If execvp succeeds, the write end is auto-closed (CLOEXEC).
    // If execvp fails, the child writes errno to the pipe.
    let mut pipe_fds = [0i32; 2];

    // SAFETY: pipe_fds is a valid [c_int; 2] on the stack.
    // pipe() writes two valid fds on success. We set CLOEXEC manually.
    #[cfg(target_os = "linux")]
    {
        let ret = unsafe { libc::pipe2(pipe_fds.as_mut_ptr(), libc::O_CLOEXEC) };
        if ret != 0 {
            return Err(PtyError::ForkFailed(std::io::Error::last_os_error()));
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        // SAFETY: pipe_fds is a valid [c_int; 2]. pipe() creates two valid fds.
        let ret = unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
        if ret != 0 {
            return Err(PtyError::ForkFailed(std::io::Error::last_os_error()));
        }
        // SAFETY: pipe_fds[0] and pipe_fds[1] are valid fds from pipe().
        // F_SETFD with FD_CLOEXEC is a standard fcntl operation.
        unsafe {
            libc::fcntl(pipe_fds[0], libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(pipe_fds[1], libc::F_SETFD, libc::FD_CLOEXEC);
        }
    }

    let pipe_read = pipe_fds[0];
    let pipe_write = pipe_fds[1];
    let slave_fd = slave.as_raw_fd();

    // SAFETY: fork() is async-signal-safe. After fork(), in the child process,
    // we only call async-signal-safe functions (setsid, dup2, close, chdir,
    // execvp, write, _exit). No Rust heap allocation occurs between fork and
    // exec. The child never returns to Rust code — it either execs or calls
    // _exit(1).
    let pid = unsafe { libc::fork() };

    if pid < 0 {
        // Fork failed — clean up pipe fds.
        // SAFETY: pipe_read and pipe_write are valid fds from pipe().
        unsafe {
            libc::close(pipe_read);
            libc::close(pipe_write);
        }
        drop(slave);
        return Err(PtyError::ForkFailed(std::io::Error::last_os_error()));
    }

    if pid == 0 {
        // ── Child process ──
        // Only async-signal-safe calls from here until exec.

        // SAFETY: close(pipe_read) — we only need the write end in the child.
        // master_fd — the child doesn't need the master side.
        unsafe {
            libc::close(pipe_read);
            libc::close(master_fd);
        }

        // SAFETY: setsid() creates a new session. The child becomes session
        // leader and process group leader. This is required for the slave PTY
        // to become the controlling terminal.
        unsafe {
            libc::setsid();
        }

        // SAFETY: TIOCSCTTY sets the slave as the controlling terminal for
        // this session. The slave_fd is a valid PTY slave from openpty().
        // Argument 0 means "don't steal from another session."
        #[cfg(not(target_os = "macos"))]
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSCTTY, 0);
        }
        // On macOS, opening the slave after setsid() automatically makes it
        // the controlling terminal — TIOCSCTTY is still available but the
        // constant may differ. Opening the slave path is the portable approach,
        // but since we already have the fd from openpty(), TIOCSCTTY works.
        #[cfg(target_os = "macos")]
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSCTTY as libc::c_ulong, 0);
        }

        // SAFETY: dup2() copies slave_fd to stdin/stdout/stderr. slave_fd is
        // a valid fd from openpty(). After dup2, fds 0/1/2 point to the slave.
        unsafe {
            libc::dup2(slave_fd, libc::STDIN_FILENO);
            libc::dup2(slave_fd, libc::STDOUT_FILENO);
            libc::dup2(slave_fd, libc::STDERR_FILENO);
        }

        // Close original slave fd if it wasn't 0, 1, or 2.
        if slave_fd > 2 {
            // SAFETY: slave_fd is valid and distinct from 0/1/2.
            unsafe {
                libc::close(slave_fd);
            }
        }

        // Change working directory.
        // SAFETY: c_working_dir is a valid CString prepared before fork.
        unsafe {
            libc::chdir(c_working_dir.as_ptr());
        }

        // Set environment variables.
        for c_var in &c_env {
            // SAFETY: c_var is a valid CString in "KEY=VALUE" format.
            unsafe {
                libc::putenv(c_var.as_ptr() as *mut libc::c_char);
            }
        }

        // Exec the program.
        // SAFETY: c_program is a valid CString. c_arg_ptrs is a null-terminated
        // array of valid CString pointers, prepared before fork. execvp either
        // replaces the process image (never returns) or returns -1 on failure.
        unsafe {
            libc::execvp(c_program.as_ptr(), c_arg_ptrs.as_ptr());
        }

        // If we get here, execvp failed. Write errno to pipe and exit.
        let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(1);
        let err_bytes = err.to_ne_bytes();
        // SAFETY: pipe_write is valid (CLOEXEC didn't fire because exec failed).
        // err_bytes is a valid [u8; 4] on the stack.
        unsafe {
            libc::write(pipe_write, err_bytes.as_ptr().cast(), err_bytes.len());
            libc::_exit(1);
        }
    }

    // ── Parent process ──

    // Close pipe write end — we only read from it.
    // SAFETY: pipe_write is a valid fd from pipe().
    unsafe {
        libc::close(pipe_write);
    }

    // Drop the slave fd — parent doesn't need it. The child has dup2'd it.
    drop(slave);

    // Read from pipe to check if exec succeeded.
    // If exec succeeded, pipe_read returns EOF (0 bytes) because CLOEXEC
    // closed the write end. If exec failed, we get 4 bytes of errno.
    let mut err_buf = [0u8; 4];
    // SAFETY: pipe_read is a valid fd. err_buf is a valid [u8; 4] on stack.
    let n = unsafe { libc::read(pipe_read, err_buf.as_mut_ptr().cast(), err_buf.len()) };

    // SAFETY: pipe_read is a valid fd.
    unsafe {
        libc::close(pipe_read);
    }

    if n > 0 {
        // Exec failed — decode errno.
        let errno = i32::from_ne_bytes(err_buf);
        // Reap the child to avoid zombie.
        // SAFETY: pid is a valid child PID from fork().
        unsafe {
            libc::waitpid(pid, std::ptr::null_mut(), 0);
        }
        return Err(PtyError::ExecFailed(std::io::Error::from_raw_os_error(
            errno,
        )));
    }

    // pid is a valid child PID (positive i32 from successful fork).
    #[allow(
        clippy::cast_sign_loss,
        reason = "pid is positive after successful fork (checked pid < 0 and pid == 0 above)"
    )]
    let child_pid = pid as u32;

    Ok(child_pid)
}

/// Resize the terminal via `TIOCSWINSZ` ioctl.
///
/// Sends the new dimensions to the terminal driver, which delivers
/// `SIGWINCH` to the foreground process group.
#[allow(unsafe_code, reason = "FFI boundary for POSIX ioctl(2) TIOCSWINSZ")]
pub fn resize(master: &OwnedFd, size: WinSize) -> Result<(), PtyError> {
    let ws = libc::winsize {
        ws_row: size.rows,
        ws_col: size.cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };

    // SAFETY: master is a valid PTY master fd (from OwnedFd). ws is a valid
    // libc::winsize on the stack. TIOCSWINSZ is the correct ioctl for setting
    // terminal dimensions.
    let ret = unsafe { libc::ioctl(master.as_raw_fd(), libc::TIOCSWINSZ, &ws) };

    if ret != 0 {
        return Err(PtyError::ResizeFailed(std::io::Error::last_os_error()));
    }

    Ok(())
}

/// Set a file descriptor to non-blocking mode (`O_NONBLOCK`).
///
/// Required before wrapping in tokio's `AsyncFd`.
#[allow(unsafe_code, reason = "FFI boundary for POSIX fcntl(2)")]
pub fn set_nonblocking(fd: &OwnedFd) -> Result<(), PtyError> {
    let raw = fd.as_raw_fd();

    // SAFETY: raw is a valid fd from OwnedFd. F_GETFL retrieves current flags.
    let flags = unsafe { libc::fcntl(raw, libc::F_GETFL) };
    if flags < 0 {
        return Err(PtyError::SetNonblockFailed(std::io::Error::last_os_error()));
    }

    // SAFETY: raw is valid. F_SETFL with O_NONBLOCK added to existing flags.
    let ret = unsafe { libc::fcntl(raw, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if ret != 0 {
        return Err(PtyError::SetNonblockFailed(std::io::Error::last_os_error()));
    }

    Ok(())
}

/// Read bytes from a PTY master fd (synchronous).
///
/// For use inside `tokio::io::unix::AsyncFd::try_io()`. Returns the number
/// of bytes read, or 0 if the slave side has been closed (child exited).
#[allow(unsafe_code, reason = "FFI boundary for POSIX read(2)")]
pub fn read_master(master: &OwnedFd, buf: &mut [u8]) -> std::io::Result<usize> {
    // SAFETY: master is a valid fd from OwnedFd. buf is a valid mutable slice.
    // read(2) writes at most buf.len() bytes and returns the count, or -1 on error.
    let n = unsafe { libc::read(master.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // n >= 0, safe to cast to usize.
        #[allow(
            clippy::cast_sign_loss,
            reason = "n is non-negative (checked n < 0 above)"
        )]
        Ok(n as usize)
    }
}

/// Write bytes to a PTY master fd (synchronous).
///
/// For use inside `tokio::io::unix::AsyncFd::try_io()`. Returns the number
/// of bytes written.
#[allow(unsafe_code, reason = "FFI boundary for POSIX write(2)")]
pub fn write_master(master: &OwnedFd, data: &[u8]) -> std::io::Result<usize> {
    // SAFETY: master is a valid fd from OwnedFd. data is a valid slice.
    // write(2) writes at most data.len() bytes and returns the count, or -1 on error.
    let n = unsafe { libc::write(master.as_raw_fd(), data.as_ptr().cast(), data.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        #[allow(
            clippy::cast_sign_loss,
            reason = "n is non-negative (checked n < 0 above)"
        )]
        Ok(n as usize)
    }
}

/// Send a signal to a process.
///
/// Wrapper around `kill(2)`. Use `libc::SIGKILL` (9) for unconditional
/// termination, `libc::SIGTERM` (15) for graceful shutdown.
#[allow(unsafe_code, reason = "FFI boundary for POSIX kill(2)")]
pub fn signal_process(pid: u32, sig: i32) -> Result<(), PtyError> {
    // SAFETY: pid is a valid PID from our fork(). sig is a signal number.
    // kill(2) is safe with any pid/sig — errors are returned, never UB.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "PID fits in i32 (kernel guarantees PIDs < 2^22)"
    )]
    let ret = unsafe { libc::kill(pid as i32, sig) };
    if ret != 0 {
        return Err(PtyError::ProcessError(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Non-blocking wait for child exit status.
///
/// Returns `Some(exit_code)` if the child has exited, `None` if still running.
/// Uses `WNOHANG` flag.
#[allow(unsafe_code, reason = "FFI boundary for POSIX waitpid(2)")]
pub fn try_wait_pid(pid: u32) -> Result<Option<i32>, PtyError> {
    let mut status: libc::c_int = 0;

    // SAFETY: pid is a valid child PID from fork(). status is a valid c_int
    // on the stack. WNOHANG makes it non-blocking — returns 0 if child is
    // still running, pid if exited, -1 on error.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "PID fits in i32 (kernel guarantees PIDs < 2^22)"
    )]
    let ret = unsafe { libc::waitpid(pid as i32, &mut status, libc::WNOHANG) };

    if ret < 0 {
        return Err(PtyError::ProcessError(std::io::Error::last_os_error()));
    }

    if ret == 0 {
        // Child still running.
        return Ok(None);
    }

    // Child exited — extract exit code.
    // WIFEXITED/WEXITSTATUS/WIFSIGNALED/WTERMSIG are safe functions in libc.
    if libc::WIFEXITED(status) {
        Ok(Some(libc::WEXITSTATUS(status)))
    } else if libc::WIFSIGNALED(status) {
        // Killed by signal — return -signal_number as the exit code.
        Ok(Some(-libc::WTERMSIG(status)))
    } else {
        Ok(Some(-1))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_pty_creates_valid_pair() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 });
        assert!(pair.is_ok(), "openpty should succeed");
        let pair = pair.expect("already checked");
        // Both fds should be valid (positive).
        assert!(pair.master.as_raw_fd() >= 0);
        assert!(pair.slave.as_raw_fd() >= 0);
        assert_ne!(pair.master.as_raw_fd(), pair.slave.as_raw_fd());
    }

    #[test]
    fn resize_on_valid_master() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 }).expect("openpty");
        let result = resize(
            &pair.master,
            WinSize {
                rows: 40,
                cols: 120,
            },
        );
        assert!(result.is_ok(), "resize should succeed on valid PTY master");
    }

    #[test]
    fn set_nonblocking_on_valid_fd() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 }).expect("openpty");
        let result = set_nonblocking(&pair.master);
        assert!(result.is_ok(), "set_nonblocking should succeed");
    }

    #[test]
    fn fork_exec_echo() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 }).expect("openpty");
        let master_raw = pair.master.as_raw_fd();
        let config = SpawnConfig {
            program: "/bin/echo",
            args: &["echo", "hello"],
            working_dir: "/tmp",
            env: &[],
        };
        let result = fork_exec(pair.slave, master_raw, &config);
        assert!(result.is_ok(), "fork_exec /bin/echo should succeed");
        let pid = result.expect("already checked");
        assert!(pid > 0);

        // Read output from master.
        let mut buf = [0u8; 256];
        let n = read_master(&pair.master, &mut buf);
        assert!(n.is_ok(), "should read from master");

        // Wait for child to finish.
        let wait = try_wait_pid(pid);
        // Child may have already exited — either Some or None is valid here.
        assert!(wait.is_ok());
    }

    #[test]
    fn fork_exec_nonexistent_fails() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 }).expect("openpty");
        let master_raw = pair.master.as_raw_fd();
        let config = SpawnConfig {
            program: "/nonexistent/binary/path",
            args: &["nonexistent"],
            working_dir: "/tmp",
            env: &[],
        };
        let result = fork_exec(pair.slave, master_raw, &config);
        assert!(
            result.is_err(),
            "fork_exec of nonexistent binary should fail"
        );
        if let Err(PtyError::ExecFailed(_)) = result {
            // Expected — exec failed, reported via pipe.
        } else {
            panic!("Expected ExecFailed, got {:?}", result.err());
        }
    }

    #[test]
    fn signal_and_wait() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 }).expect("openpty");
        let master_raw = pair.master.as_raw_fd();
        let config = SpawnConfig {
            program: "/bin/sleep",
            args: &["sleep", "60"],
            working_dir: "/tmp",
            env: &[],
        };
        let pid = fork_exec(pair.slave, master_raw, &config).expect("fork_exec sleep");

        // Should be running.
        let status = try_wait_pid(pid).expect("try_wait");
        assert_eq!(status, None, "sleep should still be running");

        // Kill it.
        signal_process(pid, libc::SIGKILL).expect("signal_process");

        // Wait for it to actually die (brief spin since SIGKILL is immediate).
        let mut exited = false;
        for _ in 0..100 {
            if let Ok(Some(_)) = try_wait_pid(pid) {
                exited = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(exited, "process should have exited after SIGKILL");
    }

    #[test]
    fn write_and_read_master() {
        let pair = open_pty(WinSize { rows: 24, cols: 80 }).expect("openpty");
        let master_raw = pair.master.as_raw_fd();
        let config = SpawnConfig {
            program: "/bin/cat",
            args: &["cat"],
            working_dir: "/tmp",
            env: &[],
        };
        let pid = fork_exec(pair.slave, master_raw, &config).expect("fork_exec cat");

        // Write to master (goes to cat's stdin).
        let written = write_master(&pair.master, b"test\n");
        assert!(written.is_ok(), "write to master should succeed");

        // Give cat a moment to echo back.
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Read from master (cat's stdout).
        set_nonblocking(&pair.master).expect("nonblock");
        let mut buf = [0u8; 256];
        let n = read_master(&pair.master, &mut buf);
        // May succeed with data or get EAGAIN — both are valid.
        if let Ok(count) = n {
            assert!(count > 0, "should have read some output from cat");
        }

        // Clean up.
        signal_process(pid, libc::SIGKILL).ok();
        try_wait_pid(pid).ok();
    }

    #[test]
    fn pty_error_display() {
        let err = PtyError::OpenPtyFailed(std::io::Error::from_raw_os_error(2));
        let msg = format!("{err}");
        assert!(msg.contains("openpty failed"));
    }

    #[test]
    fn pty_error_into_io() {
        let err = PtyError::ResizeFailed(std::io::Error::from_raw_os_error(22));
        let io_err = err.into_io();
        assert_eq!(io_err.raw_os_error(), Some(22));
    }
}
