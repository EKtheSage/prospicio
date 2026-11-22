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
use std::sync::{Arc, LazyLock, Mutex, mpsc};

use crate::call;
use crate::sheet::Grid;

/// Set when Excel unloads the add-in (`xlAutoClose`): a worker that ends
/// afterwards must not call into Excel (`xlAsyncReturn`). Runs still finish
/// and are kept; only the reply to Excel is skipped.
#[derive(Default)]
pub struct Shutdown(std::sync::atomic::AtomicBool);

impl Shutdown {
    pub const fn new() -> Self {
        Self(std::sync::atomic::AtomicBool::new(false))
    }

    /// The add-in is closing: later replies are skipped.
    pub fn close(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// The add-in is open (again): replies go through.
    pub fn open(&self) {
        self.0.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Runs `reply` unless the add-in is closing; says whether it ran.
    pub fn reply_unless_closed(&self, reply: impl FnOnce()) -> bool {
        if self.0.load(std::sync::atomic::Ordering::SeqCst) {
            return false;
        }
        reply();
        true
    }
}

/// The add-in's shutdown flag, set by `xlAutoClose`.
pub static SHUTDOWN: Shutdown = Shutdown::new();

/// The exported symbols of the functions Excel calls asynchronously:
/// every function with a `sims` or `replicates` argument and every MCMC fit.
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
    "pro_cap_nested_sim",
    "pro_sev_fit_ks",
];

/// How many results are kept, and how many cells they may hold together
/// (a 100,000-simulation table of eleven columns is 1.1 million cells).
const KEPT_RESULTS: usize = 32;
const KEPT_CELLS: usize = 4_000_000;

type Task = Box<dyn FnOnce() + Send>;

/// A fixed set of threads taking tasks in turn, so a sheet of many
/// asynchronous cells waits in a queue instead of starting a thread each.
struct Pool {
    queue: mpsc::Sender<Task>,
}

impl Pool {
    fn new(threads: usize) -> Self {
        let (queue, tasks) = mpsc::channel::<Task>();
        let tasks = Arc::new(Mutex::new(tasks));
        for _ in 0..threads {
            let tasks = Arc::clone(&tasks);
            #[cfg(test)]
            tests::THREADS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::thread::spawn(move || {
                loop {
                    let task = tasks.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok(task) = task else { break };
                    // A panicking task (a delivery callback) must not
                    // take its thread from the pool.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                }
            });
        }
        Self { queue }
    }

    fn run(&self, task: Task) {
        // The threads never stop, so the queue stays open.
        let _ = self.queue.send(task);
    }
}

/// The threads that run simulations (as many as there are processors, from
/// 2 to 8: a run is a calculation, and more threads would only compete),
/// and the two that answer cells from a result already known, each building
/// Excel's copy of the grid in turn.
struct Pools {
    runs: Pool,
    deliveries: Pool,
}

static POOLS: LazyLock<Pools> = LazyLock::new(|| Pools {
    runs: Pool::new(
        std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .clamp(2, 8),
    ),
    deliveries: Pool::new(2),
});

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

/// Locks the table. A panic while it was held (a delivery callback that
/// panics, say) poisons the mutex, but every update leaves the maps and
/// counters consistent between statements, so the data is safe to keep
/// using. Panicking here instead would take Excel down on every later call.
fn lock_jobs() -> std::sync::MutexGuard<'static, Jobs> {
    JOBS.lock().unwrap_or_else(|e| e.into_inner())
}

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
    let mut jobs = lock_jobs();
    match jobs.states.get_mut(&key) {
        Some(State::Done(grid)) => {
            let grid = Arc::clone(grid);
            // Most recently used goes last in `kept`.
            let at = jobs.kept.iter().position(|k| *k == key);
            if let Some(used) = at.and_then(|at| jobs.kept.remove(at)) {
                jobs.kept.push_back(used);
            }
            drop(jobs);
            POOLS.deliveries.run(Box::new(move || deliver(grid)));
        }
        Some(State::Running(waiting)) => waiting.push(deliver),
        None => {
            jobs.states
                .insert(key.clone(), State::Running(vec![deliver]));
            drop(jobs);
            let symbol = symbol.to_owned();
            POOLS.runs.run(Box::new(move || {
                #[cfg(test)]
                tests::RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                finish(key, Arc::new(call(&symbol, args)));
            }));
        }
    }
}

/// Keeps `grid` as the result of `key` and delivers it to everyone waiting.
/// A result over [`KEPT_CELLS`] is delivered and not kept, so it does not
/// push the others out; past [`KEPT_RESULTS`] or [`KEPT_CELLS`] together,
/// the least recently used results go.
fn finish(key: String, grid: Arc<Grid>) {
    let mut jobs = lock_jobs();
    let keep = grid.cells.len() <= KEPT_CELLS;
    let previous = if keep {
        jobs.states
            .insert(key.clone(), State::Done(Arc::clone(&grid)))
    } else {
        jobs.states.remove(&key)
    };
    let waiting = match previous {
        Some(State::Running(waiting)) => waiting,
        _ => Vec::new(),
    };
    if keep {
        jobs.kept.push_back(key);
        jobs.kept_cells += grid.cells.len();
    }
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

    #[test]
    fn no_reply_to_excel_after_close() {
        let flag = Shutdown::new();
        let mut replies = 0;
        assert!(flag.reply_unless_closed(|| replies += 1));
        flag.close();
        assert!(!flag.reply_unless_closed(|| replies += 1));
        flag.open();
        assert!(flag.reply_unless_closed(|| replies += 1));
        assert_eq!(replies, 2);
    }

    use crate::sheet::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use std::time::Duration;

    pub(super) static RUNS: AtomicUsize = AtomicUsize::new(0);
    /// The threads `submit` has started (runs and deliveries).
    pub(super) static THREADS: AtomicUsize = AtomicUsize::new(0);

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
        let jobs = lock_jobs();
        assert!(jobs.kept.len() <= KEPT_RESULTS);
        assert!(jobs.kept_cells <= KEPT_CELLS);
        let done = jobs
            .states
            .values()
            .filter(|s| matches!(s, State::Done(_)))
            .count();
        assert_eq!(done, jobs.kept.len());
    }

    /// Forgets every kept result (the tests take turns, so none is running).
    fn reset() {
        let mut jobs = lock_jobs();
        jobs.states.clear();
        jobs.kept.clear();
        jobs.kept_cells = 0;
    }

    fn copula_args(sims: f64, dim: f64) -> Vec<Grid> {
        let mut args = vec![
            Grid::scalar(Value::Text("gaussian".into())),
            Grid::scalar(Value::Number(0.5)),
            Grid::scalar(Value::Number(sims)),
            Grid::scalar(Value::Number(dim)),
        ];
        let n = crate::FUNCTIONS
            .iter()
            .find(|f| f.symbol == "pro_cap_copula_sim")
            .unwrap()
            .args
            .len();
        args.resize_with(n, Grid::missing);
        args
    }

    #[test]
    fn many_cells_on_one_result_start_few_threads() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let runs = RUNS.load(Ordering::SeqCst);
        let (tx, rx) = mpsc::channel();
        submit_to(&tx, odp_args(500.0, 31.0));
        recv(&rx);
        // The result is kept; 200 more cells ask for it at once.
        let threads = THREADS.load(Ordering::SeqCst);
        for _ in 0..200 {
            submit_to(&tx, odp_args(500.0, 31.0));
        }
        for _ in 0..200 {
            recv(&rx);
        }
        assert_eq!(RUNS.load(Ordering::SeqCst) - runs, 1);
        let started = THREADS.load(Ordering::SeqCst) - threads;
        assert!(started <= 16, "{started} threads for 200 calls");
    }

    #[test]
    fn a_result_too_big_to_keep_leaves_the_others() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let runs = RUNS.load(Ordering::SeqCst);
        let (tx, rx) = mpsc::channel();
        submit_to(&tx, odp_args(100.0, 41.0));
        recv(&rx);
        // 1,000,000 rows by 5 columns is over the 4,000,000 cells kept.
        let tx2 = tx.clone();
        submit(
            "pro_cap_copula_sim",
            copula_args(1_000_000.0, 5.0),
            Box::new(move |g| tx2.send(g).unwrap()),
        );
        let big = recv(&rx);
        assert_eq!(big.rows, 1_000_000);
        assert_eq!(RUNS.load(Ordering::SeqCst) - runs, 2);
        // The small result is still kept: no new run.
        submit_to(&tx, odp_args(100.0, 41.0));
        recv(&rx);
        assert_eq!(RUNS.load(Ordering::SeqCst) - runs, 2);
    }

    #[test]
    fn the_least_recently_used_result_is_evicted() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        let (tx, rx) = mpsc::channel();
        for seed in 0..KEPT_RESULTS {
            submit_to(&tx, odp_args(100.0, 2000.0 + seed as f64));
            recv(&rx);
        }
        // Seed 2000 is the oldest; use it again, then add one more result.
        submit_to(&tx, odp_args(100.0, 2000.0));
        recv(&rx);
        submit_to(&tx, odp_args(100.0, 3000.0));
        recv(&rx);
        let runs = RUNS.load(Ordering::SeqCst);
        // The used result survived: no run.
        submit_to(&tx, odp_args(100.0, 2000.0));
        recv(&rx);
        assert_eq!(
            RUNS.load(Ordering::SeqCst),
            runs,
            "the used result was evicted"
        );
        // The next oldest, never used again, was the one evicted.
        submit_to(&tx, odp_args(100.0, 2001.0));
        recv(&rx);
        assert_eq!(RUNS.load(Ordering::SeqCst), runs + 1);
    }

    #[test]
    fn a_poisoned_table_still_serves_calls() {
        let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        // A thread that panics while it holds the table poisons the mutex.
        let _ = std::thread::spawn(|| {
            let _jobs = JOBS.lock().unwrap_or_else(|e| e.into_inner());
            panic!("poison the jobs table");
        })
        .join();
        assert!(JOBS.is_poisoned());
        let (tx, rx) = mpsc::channel();
        submit_to(&tx, odp_args(2000.0, 21.0));
        let got = recv(&rx);
        assert_eq!(*got, call("pro_res_odp", odp_args(2000.0, 21.0)));
    }

    #[test]
    fn every_async_symbol_is_a_function() {
        for s in ASYNC {
            assert!(crate::FUNCTIONS.iter().any(|f| f.symbol == *s), "{s}");
        }
    }
}
