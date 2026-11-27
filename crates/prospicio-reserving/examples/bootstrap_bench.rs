//! Time and allocation count of the ODP and Mack bootstraps, lifetime and
//! one-year, on a 40 by 40 quarterly triangle (#1126).
//!
//! `cargo run --release -p prospicio-reserving --example bootstrap_bench -- [n_sims] [odp|mack|1y|both]`
//! (10,000 and `both` by default). A counting allocator reports the
//! allocations and bytes of each fit; `RAYON_NUM_THREADS=1` gives the
//! single-thread times, and `valgrind --tool=callgrind` on a few hundred
//! simulations the allocator's share of the instructions
//! (`knowledge/findings/bootstrap-allocations.md`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use prospicio_reserving::{
    DevelopmentColumn, Grain, Long, MackBootstrap, Month, OdpBootstrap, Triangle,
};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn triangle(n: usize) -> Triangle {
    let mut origin = vec![];
    let mut dev = vec![];
    let mut paid = vec![];
    let mut state = 12345u64;
    let mut u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for o in 0..n {
        let month = Month::new(2010 + (o / 4) as i32, (o % 4 * 3 + 1) as u8).unwrap();
        let mut c = 1000.0 * (1.0 + 0.2 * u());
        for d in 0..(n - o) {
            origin.push(month);
            dev.push(3 * (d as u32 + 1));
            paid.push(c);
            let f = 1.0 + 1.5 / (1.0 + d as f64).powf(1.3) * (0.7 + 0.6 * u());
            c *= f;
        }
    }
    Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&dev),
        values: &[("paid", &paid)],
        origin_grain: Grain::Quarter,
        development_grain: Grain::Quarter,
        cumulative: true,
    })
    .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_sims: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(10_000);
    let which = args.get(2).map(String::as_str).unwrap_or("both");
    let tri = triangle(40);
    if which != "mack" {
        let b = OdpBootstrap {
            n_sims,
            seed: 1,
            ..Default::default()
        };
        let (a0, b0) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        let t = Instant::now();
        let fit = b.fit(&tri, "paid").unwrap();
        let el = t.elapsed();
        let (a1, b1) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        println!(
            "odp  {n_sims} sims: {el:?}, {} allocs ({:.1}/sim), {:.1} MB, mean {}",
            a1 - a0,
            (a1 - a0) as f64 / n_sims as f64,
            (b1 - b0) as f64 / 1e6,
            fit.reserves.draw_matrix().iter().sum::<f64>()
        );
    }
    for (name, odp) in [("odp-1y", true), ("mack-1y", false)] {
        if which != "1y" && which != "both" {
            break;
        }
        let m = prospicio_reserving::OneYearMethod::ChainLadder(
            prospicio_reserving::ChainLadder::default(),
        );
        let (a0, b0) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        let t = Instant::now();
        if odp {
            let _ = OdpBootstrap {
                n_sims,
                seed: 1,
                ..Default::default()
            }
            .one_year(&tri, "paid", &m)
            .unwrap();
        } else {
            let _ = MackBootstrap {
                n_sims,
                seed: 1,
                ..Default::default()
            }
            .one_year(&tri, "paid", &m)
            .unwrap();
        }
        let el = t.elapsed();
        let (a1, b1) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        println!(
            "{name} {n_sims} sims: {el:?}, {} allocs ({:.1}/sim), {:.1} MB",
            a1 - a0,
            (a1 - a0) as f64 / n_sims as f64,
            (b1 - b0) as f64 / 1e6
        );
    }
    if which != "odp" {
        let b = MackBootstrap {
            n_sims,
            seed: 1,
            ..Default::default()
        };
        let (a0, b0) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        let t = Instant::now();
        let fit = b.fit(&tri, "paid").unwrap();
        let el = t.elapsed();
        let (a1, b1) = (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        );
        println!(
            "mack {n_sims} sims: {el:?}, {} allocs ({:.1}/sim), {:.1} MB, mean {}",
            a1 - a0,
            (a1 - a0) as f64 / n_sims as f64,
            (b1 - b0) as f64 / 1e6,
            fit.reserves.draw_matrix().iter().sum::<f64>()
        );
    }
}
