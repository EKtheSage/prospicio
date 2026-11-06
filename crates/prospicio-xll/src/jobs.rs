//! The functions that simulate or sample, run off Excel's calculation
//! threads (`docs/design/xll.md`, decision 48).
//!
//! Excel calls an asynchronous function with a handle and gets its result
//! later, through `xlAsyncReturn`, so a bootstrap or an MCMC fit no longer
//! holds a calculation thread. Here a call starts the run for its inputs,
//! or joins the run of the same inputs already going, or answers from the
//! results kept from earlier runs. Every one of these functions is a pure
//! function of its inputs and seed (decisions 14 and 21), so a kept result
//! is exactly what a rerun gives: the kept results save time and change no
//! number.
//!
//! When Excel interrupts a recalculation (the user types while it
//! calculates), the handles it gave are dropped on its side, but the run
//! goes on and its result is kept, so the recalculation that follows
//! answers at once. Delivering to a dropped handle is harmless: Excel
//! refuses it (`xlretInvAsynchronousContext`).
//!
//! This part is portable and tested with plain `cargo test`; `xll.rs`
//! delivers to Excel.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};

use crate::call;
use crate::sheet::Grid;

/// The exported symbols of the functions Excel calls asynchronously:
/// every function with a `sims` argument and every MCMC fit.
pub const ASYNC: &[&str] = &[
    "pro_res_odp",
    "pro_res_odp_sims",
    "pro_res_odp_cdr",
    "pro_res_mack_boot",
    "pro_res_mack_boot_sims",
    "pro_res_mack_boot_cdr",
    "pro_res_bayes_odp",
    "pro_res_bayes_bf",
    "pro_cap_copula_sim",
];

/// How many results are kept, and how many cells they may hold together
/// (a 100,000-simulation table of eleven columns is 1.1 million cells).
const KEPT_RESULTS: usize = 32;
const KEPT_CELLS: usize = 4_000_000;

/// Where a result goes once it is known: an Excel handle in `xll.rs`, a
/// channel in the tests.
pub type Deliver = Box<dyn FnOnce(Arc<Grid>) + Send>;

enum State {
    /// Running, with everyone waiting for it.
    Running(Vec<Deliver>),
    Done(Arc<Grid>),
}

#[derive(Default)]
struct Jobs {
    states: HashMap<String, State>,
    /// Kept results, oldest first.
    kept: VecDeque<String>,
    kept_cells: usize,
}

static JOBS: LazyLock<Mutex<Jobs>> = LazyLock::new(Mutex::default);

/// The key of a call: the symbol and every argument. `{:?}` writes each
/// number exactly (shortest round trip), so two calls share a key only when
/// their inputs are equal.
fn key(symbol: &str, args: &[Grid]) -> String {
    format!("{symbol}\u{0}{args:?}")
}

/// Runs `symbol` on `args` off the caller's thread and hands the result to
/// `deliver`, exactly once and never on the caller's thread: from the kept
/// results, from a run of the same inputs already going, or from a new run.
pub fn submit(symbol: &str, args: Vec<Grid>, deliver: Deliver) {
    let key = key(symbol, &args);
    let mut jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    match jobs.states.get_mut(&key) {
        Some(State::Done(grid)) => {
            let grid = Arc::clone(grid);
            drop(jobs);
            std::thread::spawn(move || deliver(grid));
        }
        Some(State::Running(waiting)) => waiting.push(deliver),
        None => {
            jobs.states
                .insert(key.clone(), State::Running(vec![deliver]));
            drop(jobs);
            let symbol = symbol.to_owned();
            std::thread::spawn(move || {
                #[cfg(test)]
                tests::RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                finish(key, Arc::new(call(&symbol, args)));
            });
        }
    }
}

/// Keeps `grid` as the result of `key` and delivers it to everyone waiting.
fn finish(key: String, grid: Arc<Grid>) {
    let mut jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
    let waiting = match jobs
        .states
        .insert(key.clone(), State::Done(Arc::clone(&grid)))
    {
        Some(State::Running(waiting)) => waiting,
        _ => Vec::new(),
    };
    jobs.kept.push_back(key);
    jobs.kept_cells += grid.cells.len();
    while jobs.kept.len() > KEPT_RESULTS || jobs.kept_cells > KEPT_CELLS {
        let Some(old) = jobs.kept.pop_front() else {
            break;
        };
        if let Some(State::Done(g)) = jobs.states.remove(&old) {
            jobs.kept_cells -= g.cells.len();
        }
    }
    drop(jobs);
    for deliver in waiting {
        deliver(Arc::clone(&grid));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheet::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    pub(super) static RUNS: AtomicUsize = AtomicUsize::new(0);

    /// The tests share `JOBS` and `RUNS`, so they take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn raa() -> Grid {
        let rows: [&[f64]; 10] = [
            &[
                5012., 8269., 10907., 11805., 13539., 16181., 18009., 18608., 18662., 18834.,
            ],
            &[
                106., 4285., 5396., 10666., 13782., 15599., 15496., 16169., 16704.,
            ],
            &[3410., 8992., 13873., 16141., 18735., 22214., 22863., 23466.],
            &[5655., 11555., 15766., 21266., 23425., 26083., 27067.],
            &[1092., 9565., 15836., 22169., 25955., 26180.],
            &[1513., 6445., 11702., 12935., 15852.],
            &[557., 4020., 10946., 12314.],
            &[1351., 6947., 13112.],
            &[3133., 5395.],
            &[2063.],
        ];
        let mut cells = Vec::with_capacity(100);
        for r in rows {
            for c in 0..10 {
                cells.push(r.get(c).map_or(Value::Empty, |&x| Value::Number(x)));
            }
        }
        Grid {
            rows: 10,
            columns: 10,
            cells,
        }
    }

    fn odp_args(sims: f64, seed: f64) -> Vec<Grid> {
        let mut args = vec![
            raa(),
            Grid::scalar(Value::Number(sims)),
            Grid::scalar(Value::Number(seed)),
        ];
        let n = crate::FUNCTIONS
            .iter()
            .find(|f| f.symbol == "pro_res_odp")
            .unwrap()
            .args
            .len();
        args.resize_with(n, Grid::missing);
        args
    }

    fn submit_to(tx: &mpsc::Sender<Arc<Grid>>, args: Vec<Grid>) {
        let tx = tx.clone();
        submit("pro_res_odp", args, Box::new(move |g| tx.send(g).unwrap()));
    }

    fn recv(rx: &mpsc::Receiver<Arc<Grid>>) -> Arc<Grid> {
        rx.recv_timeout(Duration::from_secs(60)).expect("a result")
    }

    #[test]
    fn a_result_is_the_synchronous_call() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (tx, rx) = mpsc::channel();
        submit_to(&tx, odp_args(2000.0, 11.0));
        let got = recv(&rx);
        assert_eq!(*got, call("pro_res_odp", odp_args(2000.0, 11.0)));
    }

    #[test]
    fn calls_with_the_same_inputs_share_one_run() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let before = RUNS.load(Ordering::SeqCst);
        let (tx, rx) = mpsc::channel();
        // Three cells with the same inputs while the run goes on, one after.
        for _ in 0..3 {
            submit_to(&tx, odp_args(20_000.0, 12.0));
        }
        let first = recv(&rx);
        submit_to(&tx, odp_args(20_000.0, 12.0));
        for _ in 0..3 {
            assert_eq!(*recv(&rx), *first);
        }
        assert_eq!(RUNS.load(Ordering::SeqCst) - before, 1);
        // Another seed is another run.
        submit_to(&tx, odp_args(20_000.0, 13.0));
        assert_ne!(*recv(&rx), *first);
        assert_eq!(RUNS.load(Ordering::SeqCst) - before, 2);
    }

    #[test]
    fn an_error_comes_back_as_its_text() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (tx, rx) = mpsc::channel();
        let mut args = odp_args(2000.0, 1.0);
        args[0] = Grid::scalar(Value::Text("not a triangle".into()));
        submit_to(&tx, args);
        let got = recv(&rx);
        assert!(matches!(got.as_scalar(), Some(Value::Text(t)) if t.starts_with("#PROSPICIO!")));
    }

    #[test]
    fn kept_results_are_bounded() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let (tx, rx) = mpsc::channel();
        for seed in 0..(KEPT_RESULTS + 5) {
            submit_to(&tx, odp_args(100.0, 1000.0 + seed as f64));
            recv(&rx);
        }
        let jobs = JOBS.lock().unwrap();
        assert!(jobs.kept.len() <= KEPT_RESULTS);
        assert!(jobs.kept_cells <= KEPT_CELLS);
        let done = jobs
            .states
            .values()
            .filter(|s| matches!(s, State::Done(_)))
            .count();
        assert_eq!(done, jobs.kept.len());
    }

    #[test]
    fn every_async_symbol_is_a_function() {
        for s in ASYNC {
            assert!(crate::FUNCTIONS.iter().any(|f| f.symbol == *s), "{s}");
        }
    }
}
