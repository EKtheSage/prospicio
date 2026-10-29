//! The triangle readers (`prospicio-reserving`'s `io` feature,
//! `docs/design/triangle.md` decision 6) on the repository's data files: RAA
//! read from `data/raa.csv` and from the three sheets of `data/raa.xlsx`
//! (written by `scripts/raa_xlsx.py`) equals the RAA every other test
//! builds, and errors name the cell.

use prospicio_reserving::{Error, FileLayout, Triangle};
use prospicio_validation::{triangle, triangle_columns};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");

/// The same cells, origins, ages and valuation; measure names may differ.
fn assert_same_cells(a: &Triangle, b: &Triangle) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.origins(), b.origins());
    assert_eq!(a.development(), b.development());
    assert_eq!(a.valuation(), b.valuation());
    let [ni, nc, no, nd] = a.shape();
    for i in 0..ni {
        for c in 0..nc {
            for o in 0..no {
                for d in 0..nd {
                    assert_eq!(a.get(i, c, o, d), b.get(i, c, o, d), "({i}, {c}, {o}, {d})");
                }
            }
        }
    }
}

#[test]
fn raa_from_csv_equals_the_built_in_raa() {
    let layout = FileLayout::long("origin", "development", ["value"]).comment(Some(b'#'));
    let tri = Triangle::read_csv(format!("{DATA}/raa.csv"), &layout).unwrap();
    assert_eq!(tri, triangle_columns("raa", &["value"]));
    assert_same_cells(&tri, &triangle("raa"));
}

#[test]
fn raa_from_excel_equals_the_built_in_raa() {
    let raa = triangle("raa");
    let path = format!("{DATA}/raa.xlsx");
    // The first sheet, wide: years and ages as numbers.
    let wide = Triangle::read_excel(&path, None, &FileLayout::wide("origin", "paid")).unwrap();
    assert_eq!(wide.columns(), ["paid"]);
    assert_same_cells(&wide, &raa);
    // Long, from B3, origins as date cells.
    let layout = FileLayout::long("origin", "development", ["paid"]);
    let long = Triangle::read_excel(&path, Some("long"), &layout).unwrap();
    assert_eq!(long, wide);
}

#[test]
fn excel_errors_name_the_cell_and_sheet() {
    let path = format!("{DATA}/raa.xlsx");
    let layout = FileLayout::long("origin", "development", ["paid"]);
    let message = |sheet| match Triangle::read_excel(&path, Some(sheet), &layout) {
        Err(Error::File(why)) => why,
        other => panic!("expected a file error, got {other:?}"),
    };
    let bad = message("bad");
    assert!(
        bad.ends_with("cell C3 (column \"paid\") of sheet \"bad\": \"n/a\" is not a number"),
        "{bad}"
    );
    let none = message("none");
    assert!(
        none.ends_with("has no sheet \"none\"; its sheets are \"wide\", \"long\", \"bad\""),
        "{none}"
    );
    // A long layout on the wide sheet: no development column.
    let wide = message("wide");
    assert!(wide.contains("no column named \"development\""), "{wide}");
}
