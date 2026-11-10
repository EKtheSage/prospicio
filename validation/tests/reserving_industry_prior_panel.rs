//! The industry prior per line on the calibration panel (issue #604,
//! `reserving-bayes.md` decision 29): run by hand, as the calibration
//! study is (nothing in CI runs it; it needs the panel, which is not in
//! the repo).
//!
//! The panel is the 200 Schedule P triangles of the calibration study
//! (`docs/design/reserving-calibration.md`), built from the repository
//! root with
//!
//! ```text
//! uv run --no-project --with "cas-schedule-p[screens]==2026.6.13" python -I validation/scripts/calibration_panel.py
//! ```
//!
//! into `target/calibration/panel.csv` (or the path in
//! `RISK_RS_CALIBRATION_PANEL`). Then
//!
//! ```text
//! cargo test --release -p prospicio-validation --test validation reserving_industry_prior_panel:: -- --ignored --nocapture
//! ```
//!
//! prints, for each of the four lines (50 companies each) and for paid and
//! case incurred, the prior from the triangles at the panel's valuation
//! (accident years 1988 to 1997 at 31 December 1997, the upper triangle)
//! and from the full squares (the outcomes at 120 months), and checks the
//! printed values against the ones recorded in
//! `knowledge/findings/industry-prior-calibration-panel.md`. The prior is
//! deterministic, so the check is to 1e-6 relative, on the panel's pinned
//! sha256 only.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use prospicio_reserving::{DevelopmentColumn, Grain, IndustryPriorSettings, Long, Month, Triangle};

/// The calibration study's panel (cas-schedule-p 2026.6.13).
const PANEL_SHA256: &str = "075b392525b26c505c251ea1d94f1641b4c6ab4216ee085655e5ec678fff59d2";

fn panel_path() -> PathBuf {
    match std::env::var_os("RISK_RS_CALIBRATION_PANEL") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/calibration/panel.csv"),
    }
}

/// One company's square: (origin, age in months, paid, case incurred,
/// premium).
type Rows = Vec<(i32, u32, f64, f64, f64)>;

fn read_panel(path: &Path) -> (BTreeMap<(String, String), Rows>, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "cannot read the calibration panel at {} ({e}); build it with \
             validation/scripts/calibration_panel.py (module docs)",
            path.display()
        )
    });
    let text = std::str::from_utf8(&bytes).unwrap();
    let mut companies: BTreeMap<(String, String), Rows> = BTreeMap::new();
    for row in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = row.split(',').collect();
        let num = |i: usize| f[i].parse::<f64>().unwrap();
        companies
            .entry((f[0].to_string(), f[1].to_string()))
            .or_default()
            .push((num(2) as i32, num(3) as u32, num(4), num(6), num(7)));
    }
    (companies, bytes)
}

/// The company's triangle of `column` (0 paid, 1 case incurred), the upper
/// triangle at 1997 or the full square, and each origin's premium.
fn company(rows: &Rows, incurred: bool, square: bool) -> (Triangle, Vec<f64>) {
    let mut rows: Vec<_> = rows
        .iter()
        .filter(|r| square || r.0 + (r.1 / 12) as i32 - 1 <= 1997)
        .collect();
    rows.sort_by_key(|r| (r.0, r.1));
    let origin: Vec<Month> = rows.iter().map(|r| Month::january(r.0)).collect();
    let ages: Vec<u32> = rows.iter().map(|r| r.1).collect();
    let values: Vec<f64> = rows
        .iter()
        .map(|r| if incurred { r.3 } else { r.2 })
        .collect();
    let mut premium = BTreeMap::new();
    for r in &rows {
        premium.insert(r.0, r.4);
    }
    let triangle = Triangle::from_long(&Long {
        keys: &[],
        origin: &origin,
        development: DevelopmentColumn::Age(&ages),
        values: &[("loss", &values)],
        origin_grain: Grain::Year,
        development_grain: Grain::Year,
        cumulative: true,
    })
    .unwrap();
    (triangle, premium.into_values().collect())
}

/// The recorded values: line, basis, view, then apriori, between_cv,
/// within_cv, process_cv, cv, companies used, origins used.
type Recorded = (
    &'static str,
    &'static str,
    &'static str,
    [f64; 5],
    usize,
    usize,
);

/// Recorded on 2026-11-02 (Windows, release build) from the panel of the pinned
/// sha256 under the default settings.
const RECORDED: &[Recorded] = &[
    (
        "ca",
        "paid",
        "triangle",
        [
            0.6661937390809503,
            0.2592228649407747,
            0.2917534126425043,
            0.2959912838351046,
            0.2917534126425043,
        ],
        50,
        450,
    ),
    (
        "ca",
        "paid",
        "square",
        [
            0.6627427461416565,
            0.2403492840127948,
            0.3267651443704012,
            0.2720507123895394,
            0.3267651443704012,
        ],
        50,
        500,
    ),
    (
        "ca",
        "incurred",
        "triangle",
        [
            0.6611501567886697,
            0.24398821178453103,
            0.3412384973554554,
            0.5179464836155979,
            0.3412384973554554,
        ],
        50,
        500,
    ),
    (
        "ca",
        "incurred",
        "square",
        [
            0.6647556734402076,
            0.24002687178942342,
            0.3264427831636487,
            0.5099183095115646,
            0.3264427831636487,
        ],
        50,
        500,
    ),
    (
        "pa",
        "paid",
        "triangle",
        [
            0.7797958420738237,
            0.13877613142681824,
            0.15403959417985727,
            0.14364188572456915,
            0.15403959417985727,
        ],
        50,
        450,
    ),
    (
        "pa",
        "paid",
        "square",
        [
            0.7605991649141792,
            0.12495104431216553,
            0.16465746210920768,
            0.14254681310984366,
            0.16465746210920768,
        ],
        50,
        500,
    ),
    (
        "pa",
        "incurred",
        "triangle",
        [
            0.7643506551424284,
            0.12938976572332697,
            0.1591232707564428,
            0.3248793194669435,
            0.1591232707564428,
        ],
        50,
        500,
    ),
    (
        "pa",
        "incurred",
        "square",
        [
            0.7621760013499776,
            0.1247400362157415,
            0.16573562663119198,
            0.3169024978119069,
            0.16573562663119198,
        ],
        50,
        500,
    ),
    (
        "wc",
        "paid",
        "triangle",
        [
            0.6384565975563108,
            0.18904389979214273,
            0.29269443480191426,
            0.16968294831840988,
            0.29269443480191426,
        ],
        50,
        450,
    ),
    (
        "wc",
        "paid",
        "square",
        [
            0.6226467516523405,
            0.1744414724898924,
            0.30075006328638304,
            0.3220252016571749,
            0.30075006328638304,
        ],
        50,
        500,
    ),
    (
        "wc",
        "incurred",
        "triangle",
        [
            0.65115395177505,
            0.18900503710548772,
            0.29610463949251253,
            0.6901827274478789,
            0.29610463949251253,
        ],
        50,
        500,
    ),
    (
        "wc",
        "incurred",
        "square",
        [
            0.650260338446586,
            0.1771032624860797,
            0.29967707569607627,
            0.5650848370495694,
            0.29967707569607627,
        ],
        50,
        500,
    ),
    (
        "ol",
        "paid",
        "triangle",
        [
            0.4958485291043509,
            0.38408214770594484,
            0.47049709480954477,
            0.3435593143945586,
            0.47049709480954477,
        ],
        50,
        400,
    ),
    (
        "ol",
        "paid",
        "square",
        [
            0.492655351675203,
            0.3474594378290748,
            0.4938536595274906,
            0.5851856496695884,
            0.4938536595274906,
        ],
        50,
        500,
    ),
    (
        "ol",
        "incurred",
        "triangle",
        [
            0.5014022213358155,
            0.39411387916200535,
            0.46174885429419743,
            0.8898139825528908,
            0.46174885429419743,
        ],
        50,
        500,
    ),
    (
        "ol",
        "incurred",
        "square",
        [
            0.5012005263135029,
            0.35618413395930726,
            0.4889697449777761,
            1.1506861756831506,
            0.4889697449777761,
        ],
        50,
        500,
    ),
];

#[test]
#[ignore = "needs the calibration panel (module docs); run by hand"]
fn the_industry_prior_per_line_on_the_calibration_panel() {
    let (companies, bytes) = read_panel(&panel_path());
    let pinned = sha256_hex(&bytes) == PANEL_SHA256;
    if !pinned {
        println!("the panel's sha256 is not the pinned one: values printed, not checked");
    }
    let settings = IndustryPriorSettings::default();
    println!(
        "line basis view | apriori (se) | between within process cv | used | pct developed 12 24 36 60 | sd of beta at 24, 60, 120"
    );
    let mut seen = Vec::new();
    for line in ["ca", "pa", "wc", "ol"] {
        for (basis, incurred) in [("paid", false), ("incurred", true)] {
            for (view, square) in [("triangle", false), ("square", true)] {
                let (triangles, premiums): (Vec<Triangle>, Vec<Vec<f64>>) = companies
                    .iter()
                    .filter(|((l, _), _)| l == line)
                    .map(|(_, rows)| company(rows, incurred, square))
                    .unzip();
                let prior = settings
                    .from_triangles(&triangles, "loss", &premiums)
                    .unwrap_or_else(|e| panic!("{line} {basis} {view}: {e}"));
                let r = &prior.report;
                let pd = &prior.percent_developed;
                let sd = &prior.pattern.sd;
                println!(
                    "{line} {basis} {view} | {:.4} ({:.4}) | {:.4} {:.4} {:.4} {:.4} | {} of {}, {} origins | {:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} | excluded {} | warnings {:?}",
                    prior.apriori,
                    prior.apriori_se,
                    prior.between_cv,
                    prior.within_cv,
                    prior.process_cv,
                    prior.cv,
                    r.n_companies_used,
                    r.n_companies,
                    r.n_origins_used,
                    pd[0],
                    pd[1],
                    pd[2],
                    pd[4],
                    sd[0],
                    sd[3],
                    sd[8],
                    r.excluded.len(),
                    r.warnings,
                );
                seen.push((
                    line,
                    basis,
                    view,
                    [
                        prior.apriori,
                        prior.between_cv,
                        prior.within_cv,
                        prior.process_cv,
                        prior.cv,
                    ],
                    r.n_companies_used,
                    r.n_origins_used,
                ));
            }
        }
    }
    for row in &seen {
        println!(
            "    ({:?}, {:?}, {:?}, {:?}, {}, {}),",
            row.0, row.1, row.2, row.3, row.4, row.5
        );
    }
    if pinned {
        assert_eq!(seen.len(), RECORDED.len(), "every row recorded");
        for (got, want) in seen.iter().zip(RECORDED) {
            assert_eq!((got.0, got.1, got.2), (want.0, want.1, want.2));
            for (g, w) in got.3.iter().zip(want.3) {
                assert!(
                    (g - w).abs() <= 1e-6 * w.abs().max(1e-3),
                    "{} {} {}: {g} recorded {w}",
                    got.0,
                    got.1,
                    got.2
                );
            }
            assert_eq!((got.4, got.5), (want.4, want.5));
        }
    }
}

/// SHA-256 of `bytes` as lowercase hex (FIPS 180-4), so the test needs no
/// dependency; the calibration study's own check uses the same digest.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut data = bytes.to_vec();
    let bits = (bytes.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bits.to_be_bytes());
    for chunk in data.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}
