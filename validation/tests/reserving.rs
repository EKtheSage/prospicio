//! Reserving parity against R ChainLadder on RAA.
//!
//! Runs against the provisional sandbox until `act-reserving` lands.

use act_validation::{check, reference, triangle};
use risk_rs::chain_ladder::{ChainLadder, Mack};
use risk_rs::triangle::Triangle;

#[test]
fn raa_matches_r_chainladder() {
    let tri = Triangle::from_cumulative(triangle("raa")).unwrap();
    let cl = ChainLadder::fit(&tri).unwrap();
    let mack = Mack::fit(&tri).unwrap();
    check(&reference("raa_chainladder_r.csv"), |c| {
        match c.get("quantity") {
            "ata_factor" => cl.factors.get(c.number("arg")? as usize).copied(),
            "total_reserve" => Some(cl.total_reserve()),
            "total_standard_error" => Some(mack.total_standard_error),
            _ => None,
        }
    });
}

#[test]
fn raa_dataset_shape() {
    let rows = triangle("raa");
    assert_eq!(rows.len(), 10);
    assert_eq!(rows[0].len(), 10);
    assert_eq!(rows[9], vec![2063.0]);
}
