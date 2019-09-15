//! Measurements of which parts of analysis take how long.

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
use std::sync::Mutex;
use std::time::{Instant, Duration};
use lazy_static::lazy_static;


lazy_static! {
    /// Manages all measurements.
    static ref TIMER: Mutex<Timer> = Mutex::new(Timer::new());
}

/// Try to get the timer or return early.
macro_rules! timer {
    () => { if let Ok(timer) = TIMER.lock() { timer } else { return; } };
    ($message:expr) => { TIMER.lock().expect($message) }
}

/// Retrieve the finished measurements.
pub fn get() -> Measurement {
    let timer = timer!("could not acquire timer");

    let parts = timer.finished.clone();
    let mut measurement = Measurement {
        name: "Measurement".to_string(),
        duration: parts.iter().map(|m| m.duration).sum(),
        parts
    };

    measurement.resort();
    measurement
}

/// Reset all measurements.
pub fn reset() {
    timer!().reset();
}

/// The singleton structure that holds all measurements.
struct Timer {
    stack: Vec<Timing>,
    finished: Vec<Measurement>,
}

/// A single timing created by `start` and finished into a `Measurement` by `stop`.
struct Timing {
    name: String,
    started: Instant,
    children: HashMap<String, Measurement>,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            stack: Vec::new(),
            finished: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.stack.clear();
        self.finished.clear();
    }
}

/// Run and measure `what` by executing the closure.
pub(crate) fn with<S: Into<String>, F, T>(what: S, f: F) -> T where F: FnOnce() -> T {
    start(what);
    let value = f();
    stop();
    value
}

/// Tell the timer, that `what` is now being executed.
pub(crate) fn start<S: Into<String>>(what: S) {
    let started = Instant::now();

    let mut timer = timer!();
    timer.stack.push(Timing {
        name: what.into(),
        started,
        children: HashMap::new(),
    });
}

/// Tell the timer, that the `what` from the last `start` is now finished.
pub(crate) fn stop() {
    let stopped = Instant::now();

    let mut timer = timer!();
    let timing = timer.stack.pop().expect("stop: stopped without previous start");

    let mut measurement = Measurement {
        name: timing.name.clone(),
        duration: stopped - timing.started,
        parts: timing.children.into_iter().map(|(_, v)| v).collect(),
    };

    if let Some(last) = timer.stack.last_mut() {
        last.children.entry(timing.name)
            .and_modify(|m| m.merge_with(measurement.clone()))
            .or_insert_with(|| {
                measurement.resort();
                measurement
            });
    } else {
        measurement.resort();
        timer.finished.push(measurement);
    }
}

/// A named part of execution with timed subparts.
#[derive(Debug, Clone)]
pub struct Measurement {
    pub name: String,
    pub duration: Duration,

    /// The submeasurements sorted by duration (high to low).
    pub parts: Vec<Measurement>,
}

impl Measurement {
    /// The total times of all named executions within all nested measurements.
    pub fn total(&self) -> HashMap<String, Duration> {
        let mut total = HashMap::new();
        for part in &self.parts {
            part.add_to_total(&mut total);
        }
        total
    }

    fn add_to_total(&self, total: &mut HashMap<String, Duration>) {
        total.entry(self.name.clone())
            .and_modify(|t| *t += self.duration)
            .or_insert(self.duration);

        for part in &self.parts {
            part.add_to_total(total);
        }
    }

    /// Combine two measurements by merging all executions with
    /// the same name recursively.
    pub fn merge_with(&mut self, other: Measurement) {
        self.duration += other.duration;
        for part in other.parts {
            if let Some(same) = self.parts.iter_mut().find(|m| m.name == part.name) {
                same.merge_with(part);
            } else {
                self.parts.push(part);
            }
        }
        self.resort();
    }

    /// Resort the parts by duration high to low.
    pub fn resort(&mut self) {
        self.parts.sort_by_key(|m| std::cmp::Reverse(m.duration));
    }

    fn display_with_indent(&self, f: &mut Formatter, indent: usize) -> fmt::Result {
        let ind = " ".repeat(indent);
        writeln!(f, "{}{}: {:?}", ind, self.name, self.duration)?;
        let mut first = true;
        let mut used_newline = false;
        for part in &self.parts {
            if !first && (!part.parts.is_empty() || used_newline) {
                writeln!(f)?;
                used_newline = true;
            }
            first = false;
            part.display_with_indent(f, indent + 4)?;
        }
        if !self.parts.is_empty() {
            let remaining = self.duration - self.parts.iter().map(|m| m.duration).sum();
            if remaining > Duration::from_nanos(0) {
                if used_newline { writeln!(f)?; }
                writeln!(f, "{}    other: {:?}", ind, remaining)?;
            }
        }
        Ok(())
    }
}

impl Display for Measurement {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.display_with_indent(f, 0)?;
        writeln!(f)?;
        let mut total: Vec<_> = self.total().into_iter().collect();
        total.sort_by_key(|p| std::cmp::Reverse(p.1));
        writeln!(f, "Total:")?;
        for (name, time) in &total {
            let ratio = (time.as_nanos() as f64) / (self.duration.as_nanos() as f64);
            writeln!(f, "    {}: {:?} ({:.2} %)", name, time, 100.0 * ratio)?;
        }
        Ok(())
    }
}
