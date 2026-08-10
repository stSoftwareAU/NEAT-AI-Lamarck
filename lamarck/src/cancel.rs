//! Graceful cancellation for the optimisation loop (Issue #72).
//!
//! `SIGINT`/`SIGTERM` must not kill Lamarck mid-experiment: the run-summary
//! re-tag of `best.json` and the working-directory cleanup both live after the
//! loop. The signal handler therefore only sets the flag below; the loop polls
//! it, abandons the in-flight experiment and exits through the normal path.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shared "stop as soon as it is safe to" flag.
///
/// Cloning shares the flag, so a signal handler, a test, or any other thread
/// can cancel a run the optimisation loop is executing.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A token that has not been cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Safe to call from a signal handler.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// The underlying flag, for registering with a signal handler.
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag)
    }
}

/// Exit code used when a second `SIGINT` force-quits an unresponsive run.
///
/// `128 + SIGINT`, the shell convention for "terminated by signal 2".
#[cfg(unix)]
pub const FORCE_QUIT_EXIT_CODE: i32 = 130;

/// Route `SIGINT`/`SIGTERM` into `token` so the run stops gracefully.
///
/// A **second** signal of the same kind force-quits with
/// [`FORCE_QUIT_EXIT_CODE`], so a run wedged inside a long scorer batch can
/// still be killed without `SIGKILL`.
///
/// # Errors
///
/// Returns the OS error when a handler cannot be installed — an uninstallable
/// handler means cancellation would silently do nothing, so the caller must
/// fail rather than continue.
#[cfg(unix)]
pub fn install_cancel_signals(token: &CancelToken) -> Result<(), String> {
    use signal_hook::consts::{SIGINT, SIGTERM};

    for signal in [SIGINT, SIGTERM] {
        // Registered first so it runs first: it fires only once the flag is
        // already set, i.e. on the second signal.
        signal_hook::flag::register_conditional_shutdown(
            signal,
            FORCE_QUIT_EXIT_CODE,
            token.flag(),
        )
        .map_err(|e| format!("cannot install force-quit handler for signal {signal}: {e}"))?;
        signal_hook::flag::register(signal, token.flag())
            .map_err(|e| format!("cannot install cancellation handler for signal {signal}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_token_is_not_cancelled() {
        assert!(!CancelToken::new().is_cancelled());
    }

    #[test]
    fn cancel_sets_the_flag() {
        let token = CancelToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancelling_is_idempotent() {
        let token = CancelToken::new();
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn clones_share_one_flag() {
        let token = CancelToken::new();
        let clone = token.clone();
        clone.cancel();
        assert!(
            token.is_cancelled(),
            "a clone must cancel the original token"
        );
    }

    #[test]
    fn another_thread_can_cancel() {
        let token = CancelToken::new();
        let worker = token.clone();
        std::thread::spawn(move || worker.cancel()).join().unwrap();
        assert!(token.is_cancelled());
    }

    /// A real `SIGINT` must set the token instead of killing the process (#72).
    #[cfg(unix)]
    #[test]
    fn sigint_cancels_the_token_instead_of_terminating() {
        let token = CancelToken::new();
        install_cancel_signals(&token).expect("handlers install");
        assert!(!token.is_cancelled());

        signal_hook::low_level::raise(signal_hook::consts::SIGINT).expect("raise SIGINT");

        // Delivery is asynchronous; the process is still alive either way.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !token.is_cancelled() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            token.is_cancelled(),
            "SIGINT must set the cancellation flag"
        );
    }
}
