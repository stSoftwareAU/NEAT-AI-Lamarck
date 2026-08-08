//! Lightweight coloured stderr logging for interactive Lamarck runs.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

static COLOUR_ENABLED: AtomicU8 = AtomicU8::new(2); // 0=off 1=on 2=unset

fn colour_enabled() -> bool {
    match COLOUR_ENABLED.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let enabled = std::io::stderr().is_terminal()
                && std::env::var_os("NO_COLOR").is_none()
                && std::env::var_os("LAMARCK_NO_COLOR").is_none();
            COLOUR_ENABLED.store(u8::from(enabled), Ordering::Relaxed);
            enabled
        }
    }
}

fn paint(code: &str, text: &str) -> String {
    if colour_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

/// Log an informational banner line.
pub fn info(msg: &str) {
    eprintln!("{} {}", paint("1;36", "●"), msg);
}

/// Log a success / completed step.
pub fn ok(msg: &str) {
    eprintln!("{} {}", paint("1;32", "✓"), msg);
}

/// Log a warning.
pub fn warn(msg: &str) {
    eprintln!("{} {}", paint("1;33", "⚠"), msg);
}

/// Log a dim detail / progress line.
pub fn detail(msg: &str) {
    eprintln!("  {}", paint("2", msg));
}

/// Log a bright progress update with a spinner glyph.
pub fn progress(spinner: char, msg: &str) {
    eprint!(
        "\r{} {}                    ",
        paint("1;35", &spinner.to_string()),
        msg
    );
    let _ = std::io::stderr().flush();
}

/// Finish a progress line (newline after `\r` updates).
pub fn progress_done(msg: &str) {
    eprintln!("\r{} {}                    ", paint("1;32", "✓"), msg);
}

/// Spinner state for periodic progress.
#[derive(Debug)]
pub struct Spinner {
    frames: &'static [char],
    idx: usize,
    last: Instant,
    every: Duration,
}

impl Spinner {
    /// Create a spinner that advances at most every `every` duration.
    pub fn new(every: Duration) -> Self {
        Self {
            frames: &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
            idx: 0,
            last: Instant::now()
                .checked_sub(every)
                .unwrap_or_else(Instant::now),
            every,
        }
    }

    /// Advance and return the current glyph when enough time has elapsed.
    pub fn tick(&mut self) -> Option<char> {
        if self.last.elapsed() < self.every {
            return None;
        }
        self.last = Instant::now();
        let ch = self.frames[self.idx % self.frames.len()];
        self.idx = self.idx.wrapping_add(1);
        Some(ch)
    }
}
