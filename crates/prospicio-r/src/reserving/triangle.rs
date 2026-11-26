//! The loss triangle, `Triangle`.

use super::*;

/// A loss triangle: index × column × origin × development.
#[extendr]
pub(crate) struct Triangle {
    pub(crate) inner: TriangleInner,
}

impl From<TriangleInner> for Triangle {
    fn from(inner: TriangleInner) -> Self {
        Self { inner }
    }
}

#[extendr]
impl Triangle {
    /// Builds a triangle from a long table. `keys` holds the key columns
    /// named `key_names`, column by column; `values` the value columns,
    /// column by column. Development is `ages`, or the valuation months
    /// when `development_is_valuation`.
    #[allow(clippy::too_many_arguments)]
    fn from_long(
        key_names: Vec<String>,
        keys: Vec<String>,
        origin_year: &[i32],
        origin_month: &[i32],
        ages: &[f64],
        valuation_year: &[i32],
        valuation_month: &[i32],
        development_is_valuation: bool,
        names: Vec<String>,
        values: &[f64],
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
    ) -> Result<Self> {
        let origin = months(origin_year, origin_month, "origin")?;
        let n = origin.len();
        if keys.len() != n * key_names.len() {
            return Err(Error::Other(format!(
                "keys has {} entries, expected {n} rows times {} keys",
                keys.len(),
                key_names.len()
            )));
        }
        let key_values: Vec<Vec<&str>> = (0..key_names.len())
            .map(|k| {
                keys[k * n..(k + 1) * n]
                    .iter()
                    .map(String::as_str)
                    .collect()
            })
            .collect();
        let key_columns: Vec<(&str, &[&str])> = key_names
            .iter()
            .zip(&key_values)
            .map(|(name, v)| (name.as_str(), v.as_slice()))
            .collect();
        if values.len() != n * names.len() {
            return Err(Error::Other(format!(
                "values has {} entries, expected {n} rows times {} columns",
                values.len(),
                names.len()
            )));
        }
        let columns: Vec<(&str, &[f64])> = names
            .iter()
            .enumerate()
            .map(|(c, name)| (name.as_str(), &values[c * n..(c + 1) * n]))
            .collect();
        let valuations;
        let lags: Vec<Lag>;
        let development = if development_is_valuation {
            valuations = months(valuation_year, valuation_month, "development")?;
            DevelopmentColumn::Valuation(&valuations)
        } else {
            lags = ages.iter().map(|&a| age(a)).collect::<Result<_>>()?;
            DevelopmentColumn::Age(&lags)
        };
        let inner = TriangleInner::from_long(&Long {
            keys: &key_columns,
            origin: &origin,
            development,
            values: &columns,
            origin_grain: grain(origin_grain, "origin_grain")?,
            development_grain: grain(development_grain, "development_grain")?,
            cumulative,
        })
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    /// Reads a triangle from a CSV file (`format` `"csv"`) or a sheet of a
    /// workbook (`"excel"`; `sheet` `NULL` for the first). Long when
    /// `development` is given, wide otherwise (`columns` then names the one
    /// measure, `"values"` when empty, and `ages` may name the age columns).
    /// `comment` is a CSV comment character, `""` for none.
    #[allow(clippy::too_many_arguments)]
    fn read_file(
        path: &str,
        format: &str,
        sheet: Nullable<String>,
        origin: &str,
        development: Nullable<String>,
        columns: Vec<String>,
        keys: Vec<String>,
        ages: Nullable<Vec<f64>>,
        origin_grain: &str,
        development_grain: &str,
        cumulative: bool,
        development_is_valuation: bool,
        comment: &str,
    ) -> Result<Self> {
        let layout = match development {
            Nullable::NotNull(development) => {
                if matches!(ages, Nullable::NotNull(_)) {
                    return Err(Error::Other(
                        "ages name the columns of a wide layout; give development = NULL".into(),
                    ));
                }
                FileLayout::long(origin, development, columns)
            }
            Nullable::Null => {
                if development_is_valuation {
                    return Err(Error::Other(
                        "development_is_valuation needs a long layout: give the development column"
                            .into(),
                    ));
                }
                let column = match columns.as_slice() {
                    [] => "values".to_string(),
                    [one] => one.clone(),
                    _ => {
                        return Err(Error::Other(
                            "a wide layout has one measure: give columns as one name".into(),
                        ));
                    }
                };
                let layout = FileLayout::wide(origin, column);
                match ages {
                    Nullable::NotNull(ages) => {
                        layout.ages(ages.into_iter().map(age).collect::<Result<Vec<Lag>>>()?)
                    }
                    Nullable::Null => layout,
                }
            }
        };
        let comment = match comment.as_bytes() {
            [] => None,
            [c] if c.is_ascii() => Some(*c),
            _ => {
                return Err(Error::Other(format!(
                    "comment must be one ASCII character, got {comment:?}"
                )));
            }
        };
        let layout = layout
            .keys(keys)
            .grains(
                grain(origin_grain, "origin_grain")?,
                grain(development_grain, "development_grain")?,
            )
            .cumulative(cumulative)
            .development_is_valuation(development_is_valuation)
            .comment(comment);
        let inner = match format {
            "csv" => TriangleInner::read_csv(path, &layout),
            "excel" => {
                let sheet = match &sheet {
                    Nullable::NotNull(s) => Some(s.as_str()),
                    Nullable::Null => None,
                };
                TriangleInner::read_excel(path, sheet, &layout)
            }
            other => {
                return Err(Error::Other(format!(
                    "format must be \"csv\" or \"excel\", got {other:?}"
                )));
            }
        }
        .map_err(to_r)?;
        Ok(Self { inner })
    }

    fn shape(&self) -> Vec<i32> {
        self.inner.shape().iter().map(|&n| n as i32).collect()
    }

    /// Names of the key columns.
    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// Index labels as a named list of key columns, one row per label.
    fn index(&self) -> List {
        let index = self.inner.index();
        key_columns(
            self.inner
                .key_names()
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    let values = index.iter().map(|l| l.parts()[k].clone()).collect();
                    (name.clone(), values)
                })
                .collect(),
        )
    }

    /// Index labels with their parts joined by " / " ("Total" without keys).
    fn index_names(&self) -> Vec<String> {
        self.inner.index().iter().map(Label::to_string).collect()
    }

    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    fn origins(&self) -> Vec<String> {
        self.inner
            .origins()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    fn development(&self) -> Vec<i32> {
        self.inner.development().iter().map(|&a| a as i32).collect()
    }

    fn origin_grain(&self) -> String {
        self.inner.origin_grain().to_string()
    }

    fn development_grain(&self) -> String {
        self.inner.development_grain().to_string()
    }

    /// Valuation month as "YYYY-MM".
    fn valuation(&self) -> String {
        self.inner.valuation().to_string()
    }

    fn is_cumulative(&self) -> bool {
        self.inner.is_cumulative()
    }

    /// Values, row-major over (index, column, origin, development).
    fn values(&self) -> Vec<f64> {
        let [ni, nc, no, nd] = self.inner.shape();
        let mut out = Vec::with_capacity(ni * nc * no * nd);
        for i in 0..ni {
            for c in 0..nc {
                for o in 0..no {
                    for d in 0..nd {
                        out.push(self.inner.get(i, c, o, d).unwrap_or(f64::NAN));
                    }
                }
            }
        }
        out
    }

    /// The long table: the key columns (a named list), origin start (year,
    /// month), age and the value columns (NaN where a measure is not
    /// observed on a row).
    fn to_long(&self) -> List {
        let long = self.inner.to_long();
        let values = List::from_values(long.values.iter().map(|(_, v)| v.clone()));
        let names: Vec<String> = long.values.iter().map(|(n, _)| n.clone()).collect();
        list!(
            keys = key_columns(long.keys),
            origin_year = long.origin.iter().map(|m| m.year()).collect::<Vec<i32>>(),
            origin_month = long
                .origin
                .iter()
                .map(|m| m.month() as i32)
                .collect::<Vec<i32>>(),
            development = long
                .development
                .iter()
                .map(|&a| a as i32)
                .collect::<Vec<i32>>(),
            names = names,
            values = values
        )
    }

    fn to_incremental(&self) -> Self {
        self.inner.to_incremental().into()
    }

    fn to_cumulative(&self) -> Self {
        self.inner.to_cumulative().into()
    }

    /// Latest values, row-major over (index, column, origin); NaN where an
    /// origin has no observation.
    fn latest_diagonal(&self) -> Vec<f64> {
        let diagonal = self.inner.latest_diagonal();
        let [ni, nc, no] = diagonal.shape();
        let mut out = Vec::with_capacity(ni * nc * no);
        for i in 0..ni {
            for c in 0..nc {
                for o in 0..no {
                    out.push(diagonal.get(i, c, o).map_or(f64::NAN, |(_, v)| v));
                }
            }
        }
        out
    }

    fn link_ratios(&self) -> Self {
        self.inner.link_ratios().into()
    }

    /// Keeps the segments whose value of `keys[k]` is in `values[[k]]` (a
    /// character vector) for every `k`, then the named columns (`NULL`
    /// keeps every column).
    fn select(
        &self,
        keys: Vec<String>,
        values: List,
        columns: Nullable<Vec<String>>,
    ) -> Result<Self> {
        let values: Vec<Vec<String>> = values
            .values()
            .map(|v| {
                Vec::<String>::try_from(v)
                    .map_err(|_| Error::Other("key values must be character".into()))
            })
            .collect::<Result<_>>()?;
        if values.len() != keys.len() {
            return Err(Error::Other("one set of values is needed per key".into()));
        }
        let values: Vec<Vec<&str>> = values
            .iter()
            .map(|v| v.iter().map(String::as_str).collect())
            .collect();
        let conditions: Vec<(&str, &[&str])> = keys
            .iter()
            .zip(&values)
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let mut inner = self.inner.select(&conditions).map_err(to_r)?;
        if let Nullable::NotNull(columns) = columns {
            let columns: Vec<&str> = columns.iter().map(String::as_str).collect();
            inner = inner.select_columns(&columns).map_err(to_r)?;
        }
        Ok(inner.into())
    }

    /// Sums the segments that share the values of `keys`.
    fn group_by(&self, keys: Vec<String>) -> Result<Self> {
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        Ok(self.inner.group_by(&keys).map_err(to_r)?.into())
    }

    fn grain(&self, origin_grain: &str, development_grain: &str) -> Result<Self> {
        let inner = self
            .inner
            .grain(
                grain(origin_grain, "origin_grain")?,
                grain(development_grain, "development_grain")?,
            )
            .map_err(to_r)?;
        Ok(inner.into())
    }

    /// One-segment triangles stacked as segments under the key `key`, one
    /// name each (the core's `Triangle::stack`, #838).
    fn stack(names: Vec<String>, triangles: List, key: &str, columns: Vec<String>) -> Result<Self> {
        if names.len() != triangles.len() {
            return Err(Error::Other("one name per triangle".into()));
        }
        let items = triangles
            .values()
            .zip(&names)
            .map(|(t, name)| {
                <&Triangle>::try_from(&t)
                    .map(|t| t.inner.clone())
                    .map_err(|_| Error::Other(format!("{name:?} is not a triangle")))
            })
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<(&str, &TriangleInner)> =
            names.iter().map(String::as_str).zip(items.iter()).collect();
        let cols: Vec<&str> = columns.iter().map(String::as_str).collect();
        Ok(TriangleInner::stack(&refs, key, &cols)
            .map_err(to_r)?
            .into())
    }

    /// The ages up to `age` months (a whole number, 0 or more).
    fn max_age(&self, age: i32) -> Result<Self> {
        let age = u32::try_from(age).map_err(|_| Error::Other("`age` must be 0 or more".into()))?;
        Ok(self.inner.max_age(age).map_err(to_r)?.into())
    }

    /// One segment (chosen by `keys` = `values`, one value each) and
    /// measure (`NULL`: the only one) as `origins`, `development` and
    /// `values` row-major over origin × development, NaN where unobserved.
    fn view(
        &self,
        keys: Vec<String>,
        values: Vec<String>,
        column: Nullable<String>,
    ) -> Result<List> {
        let column = match &column {
            Nullable::NotNull(c) => Some(c.as_str()),
            Nullable::Null => None,
        };
        let view = self
            .inner
            .view(&choice(&keys, &values)?, column)
            .map_err(to_r)?;
        let (no, nd) = (view.origins.len(), view.development.len());
        let cells: Vec<f64> = (0..no)
            .flat_map(|o| (0..nd).map(move |d| (o, d)))
            .map(|(o, d)| view.get(o, d).unwrap_or(f64::NAN))
            .collect();
        Ok(list!(
            origins = view
                .origins
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            development = view
                .development
                .iter()
                .map(|&a| a as i32)
                .collect::<Vec<_>>(),
            values = cells
        ))
    }

    /// One row per segment and measure: the key columns (a named list),
    /// `column`, `n_origins`, `first_origin` and `last_origin` (period
    /// labels), `valuation` ("YYYY-MM"), with "" where the segment has no
    /// observed value, `latest` and `cumulative`.
    fn summary(&self) -> List {
        let summary = self.inner.summary();
        let rows = &summary.rows;
        let keys = summary
            .key_names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let values = rows.iter().map(|r| r.label.parts()[k].clone()).collect();
                (name.clone(), values)
            })
            .collect();
        let text = |x: Option<String>| x.unwrap_or_default();
        list!(
            keys = key_columns(keys),
            column = rows.iter().map(|r| r.column.clone()).collect::<Vec<_>>(),
            n_origins = rows.iter().map(|r| r.n_origins as i32).collect::<Vec<_>>(),
            first_origin = rows
                .iter()
                .map(|r| text(r.first_origin.map(|p| p.to_string())))
                .collect::<Vec<_>>(),
            last_origin = rows
                .iter()
                .map(|r| text(r.last_origin.map(|p| p.to_string())))
                .collect::<Vec<_>>(),
            valuation = rows
                .iter()
                .map(|r| text(r.valuation.map(|m| m.to_string())))
                .collect::<Vec<_>>(),
            latest = rows.iter().map(|r| r.latest).collect::<Vec<_>>(),
            cumulative = summary.cumulative
        )
    }

    /// The printout: the grid of one segment and measure, otherwise the
    /// summary table, with at most `max_rows` rows and `max_cols` ages (0
    /// for no limit).
    fn to_text(&self, max_rows: f64, max_cols: f64) -> Result<String> {
        let max_rows = whole(max_rows, "max_rows")? as usize;
        let max_cols = whole(max_cols, "max_cols")? as usize;
        Ok(self.inner.to_text(max_rows, max_cols))
    }

    /// `tail` is a `ReservingTail`; `ldf` a selected pattern as
    /// `ldf_pattern()` in R encodes it, or `NULL`.
    fn chain_ladder(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ChainLadderFit> {
        let inner = pattern(
            average,
            sigma_interpolation,
            &excl,
            tail,
            &ldf,
            self.inner.origin_grain(),
        )?
        .fit_segments(&self.inner, column)
        .map_err(to_r)?;
        Ok(ChainLadderFit { inner })
    }

    /// A development curve (`development_curve()`) fitted to the estimated
    /// factors of `column`, averaged as `average` with the exclusions
    /// `excl`.
    #[allow(clippy::too_many_arguments)]
    fn development_curve(
        &self,
        column: &str,
        average: &str,
        excl: Robj,
        shape: &str,
        fit_from: Nullable<f64>,
        fit_to: Nullable<f64>,
        keep_observed_until: Nullable<f64>,
    ) -> Result<DevelopmentCurveFit> {
        let curve = development_curve_settings(shape, fit_from, fit_to, keep_observed_until)?;
        let dev = curve_development(&self.inner, column, average, &excl)?;
        let inner = curve.fit(&dev).map_err(to_r)?;
        Ok(DevelopmentCurveFit { inner })
    }

    /// `development_curve_compare()`: each of `shapes` fitted to the same
    /// factors, a list of columns `shape`, `n_points`, `n_parameters`,
    /// `sum_of_squares`, `r_squared` and `adjusted_r_squared`.
    #[allow(clippy::too_many_arguments)]
    fn development_curve_compare(
        &self,
        column: &str,
        average: &str,
        excl: Robj,
        shapes: Vec<String>,
        fit_from: Nullable<f64>,
        fit_to: Nullable<f64>,
        keep_observed_until: Nullable<f64>,
    ) -> Result<List> {
        let curve =
            development_curve_settings("exponential", fit_from, fit_to, keep_observed_until)?;
        let shapes = shapes
            .iter()
            .map(|s| curve_shape(s, "shapes"))
            .collect::<Result<Vec<_>>>()?;
        let dev = curve_development(&self.inner, column, average, &excl)?;
        let rows = curve.compare(&dev, &shapes).map_err(to_r)?;
        let num = |f: fn(&DevelopmentCurveInner) -> f64| rows.iter().map(f).collect::<Vec<f64>>();
        Ok(list!(
            shape = rows
                .iter()
                .map(|r| curve_name(r.curve.shape))
                .collect::<Vec<_>>(),
            n_points = num(|r| r.n_points as f64),
            n_parameters = num(|r| r.n_parameters as f64),
            sum_of_squares = num(|r| r.sum_of_squares),
            r_squared = num(|r| r.r_squared),
            adjusted_r_squared = num(|r| r.adjusted_r_squared)
        ))
    }

    /// Mack's calendar-year effect test on every observed link ratio of
    /// `column`, at confidence level `ci`.
    fn calendar_year_effect(&self, column: &str, ci: f64) -> Result<CalendarYearTest> {
        Ok(CalendarYearTest {
            inner: calendar_year_effect(&self.inner, column, ci).map_err(to_r)?,
        })
    }

    /// Mack's development-factor correlation test on every observed link
    /// ratio of `column`, at confidence level `ci`.
    fn development_correlation(&self, column: &str, ci: f64) -> Result<DevelopmentCorrelationTest> {
        Ok(DevelopmentCorrelationTest {
            inner: development_correlation(&self.inner, column, ci).map_err(to_r)?,
        })
    }

    /// Friedland's development exhibit of `column`: the selection is the
    /// chain ladder of `average` ... `ldf` (as in `chain_ladder`); the
    /// averages are Friedland's six when `friedland`, else one per
    /// `average_names` with its kind in `average_kinds` and its exclusions
    /// in `average_excl` (a list of exclusion lists).
    #[allow(clippy::too_many_arguments)]
    fn development_exhibit(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
        friedland: bool,
        average_names: Vec<String>,
        average_kinds: Vec<String>,
        average_excl: List,
    ) -> Result<DevelopmentExhibit> {
        let selection = pattern(
            average,
            sigma_interpolation,
            &excl,
            tail,
            &ldf,
            self.inner.origin_grain(),
        )?;
        let averages = if friedland {
            friedland_averages()
        } else {
            if average_kinds.len() != average_names.len()
                || average_excl.len() != average_names.len()
            {
                return Err(Error::Other(
                    "each average needs a name, a kind and its exclusions".into(),
                ));
            }
            average_names
                .into_iter()
                .zip(&average_kinds)
                .zip(average_excl.values())
                .map(|((name, kind), excl)| {
                    Ok((name, development(kind, "log-linear", &excl, None)?))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let inner =
            DevelopmentExhibitInner::with_averages(&self.inner, column, &selection, &averages)
                .map_err(to_r)?;
        Ok(DevelopmentExhibit { inner })
    }

    /// The ultimate exhibit of the latest `reported` and `paid` columns,
    /// either `NULL`.
    fn ultimate_exhibit(
        &self,
        reported: Nullable<String>,
        paid: Nullable<String>,
    ) -> Result<UltimateExhibit> {
        let opt = |v: &Nullable<String>| match v {
            Nullable::NotNull(s) => Some(s.clone()),
            Nullable::Null => None,
        };
        let (reported, paid) = (opt(&reported), opt(&paid));
        let inner =
            UltimateExhibitInner::from_triangle(&self.inner, reported.as_deref(), paid.as_deref())
                .map_err(to_r)?;
        Ok(UltimateExhibit { inner })
    }

    /// `tail` is a `ReservingTail`; `tail_sigma` and `tail_std_err` are
    /// `NULL` to extrapolate them; `ldf` as in `chain_ladder`.
    #[allow(clippy::too_many_arguments)]
    fn mack(
        &self,
        column: &str,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
        ldf: Robj,
    ) -> Result<MackFit> {
        let inner = Mack {
            development: development(
                average,
                sigma_interpolation,
                &excl,
                selected_pattern(&ldf, self.inner.origin_grain())?,
            )?,
            tail: tail_arg(&tail)?,
            tail_sigma: optional(tail_sigma),
            tail_std_err: optional(tail_std_err),
        }
        .fit_segments(&self.inner, column)
        .map_err(to_r)?;
        Ok(MackFit { inner })
    }

    /// The Munich chain ladder on the `paid` and `incurred` columns; each
    /// sigma interpolation is `"log-linear"` or `"mack"`, each tail a
    /// `ReservingTail`.
    fn munich_chain_ladder(
        &self,
        paid: &str,
        incurred: &str,
        sigma_paid: &str,
        sigma_incurred: &str,
        tail_paid: Robj,
        tail_incurred: Robj,
    ) -> Result<MunichFit> {
        let dev = |sigma: &str| -> Result<Development> {
            let sigma_interpolation = SigmaInterpolation::from_name(sigma).ok_or_else(|| {
                crate::unknown_name("sigma_interpolation", sigma, SigmaInterpolation::NAMES)
            })?;
            Ok(Development {
                sigma_interpolation,
                ..Default::default()
            })
        };
        let inner = MunichChainLadder {
            paid: dev(sigma_paid)?,
            incurred: dev(sigma_incurred)?,
            tail_paid: tail_arg(&tail_paid)?,
            tail_incurred: tail_arg(&tail_incurred)?,
        }
        .fit_segments(&self.inner, paid, incurred)
        .map_err(to_r)?;
        Ok(MunichFit { inner })
    }

    /// The expected loss ratio method on loss `column` with exposure from
    /// the `exposure` column; the chain ladder gives the development
    /// pattern reported.
    #[allow(clippy::too_many_arguments)]
    fn expected_loss(
        &self,
        column: &str,
        exposure: &str,
        apriori: Robj,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ExpectedLossFit> {
        let inner = ExpectedLoss {
            apriori: apriori_arg(&apriori)?,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.inner.origin_grain(),
            )?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn bornhuetter_ferguson(
        &self,
        column: &str,
        exposure: &str,
        apriori: Robj,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ExpectedLossFit> {
        let inner = BornhuetterFerguson {
            apriori: apriori_arg(&apriori)?,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.inner.origin_grain(),
            )?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn benktander(
        &self,
        column: &str,
        exposure: &str,
        apriori: Robj,
        n_iters: f64,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
    ) -> Result<ExpectedLossFit> {
        let n_iters = usize::try_from(whole(n_iters, "n_iters")?)
            .map_err(|_| Error::Other(format!("n_iters {n_iters} is too large")))?;
        let inner = Benktander {
            apriori: apriori_arg(&apriori)?,
            n_iters,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.inner.origin_grain(),
            )?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn cape_cod(
        &self,
        column: &str,
        exposure: &str,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
    ) -> Result<CapeCodFit> {
        let inner = CapeCod {
            trend,
            decay,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.inner.origin_grain(),
            )?,
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(CapeCodFit { inner })
    }

    /// The ODP bootstrap of every segment, the segments depending on each
    /// other as `dependence` ("independent", "synchronized" or
    /// "rank_correlation" with the row-major `spearman` matrix) says.
    /// `tail` is a `ReservingTail`, developed past the oldest age in every
    /// simulation; `tail_std_err` is `NULL` to keep a constant tail fixed.
    /// `excl` is the link ratio exclusions as `development_args()` in R
    /// encodes them. With `cells` the fit keeps each simulation's future
    /// incremental values (`OdpBootstrap::fit_segments_with_cells`).
    #[allow(clippy::too_many_arguments)]
    fn odp_bootstrap(
        &self,
        column: &str,
        n_sims: f64,
        seed: f64,
        process: &str,
        tail: Robj,
        tail_std_err: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
        excl: Robj,
        cells: bool,
    ) -> Result<OdpBootstrapFit> {
        let dependence = segment_dependence(dependence, spearman)?;
        let (n_periods, drop_high, drop_low) = exclusions(&excl)?;
        let boot = OdpBootstrap {
            tail: tail_arg(&tail)?,
            tail_std_err: optional(tail_std_err),
            pattern: selected_pattern(&ldf, self.inner.origin_grain())?,
            n_periods,
            drop_high,
            drop_low,
            ..bootstrap(n_sims, seed, process, dependence)?
        };
        let inner = if cells {
            boot.fit_segments_with_cells(&self.inner, column)
        } else {
            boot.fit_segments(&self.inner, column)
        }
        .map_err(to_r)?;
        Ok(OdpBootstrapFit { inner })
    }

    /// The Bayesian ODP chain ladder of a single-segment triangle column:
    /// normal priors with sds `intercept_sd`, `origin_sd` and
    /// `development_sd` on the log scale, `process` error ("gamma" or
    /// "none"), NUTS with `chains`, `tune` and `draws` per chain,
    /// `target_accept` and `max_depth`, and `n_sims` predictive
    /// simulations, all replayed from `seed`.
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp(
        &self,
        column: &str,
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
    ) -> Result<BayesOdpFit> {
        let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
        let model = BayesOdp {
            intercept_sd,
            origin_sd,
            development_sd,
            process: boot.process,
            sampler: BayesSampler {
                chains: whole(chains, "chains")? as usize,
                tune: whole(tune, "tune")? as usize,
                draws: whole(draws, "draws")? as usize,
                seed: 0,
                target_accept,
                max_depth: whole(max_depth, "max_depth")?,
            },
            n_sims: boot.n_sims,
            seed: boot.seed,
            ..BayesOdp::default()
        };
        let inner = model.fit(&self.inner, column).map_err(to_r)?;
        Ok(BayesOdpFit { inner })
    }

    /// The Bayesian ODP chain ladder of every segment of a triangle column,
    /// with the settings of `bayes_odp`, each segment's own posterior, and
    /// the segments depending on each other as `dependence`
    /// ("independent", "rank_correlation" with the row-major `spearman`
    /// matrix, or "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp_segments(
        &self,
        column: &str,
        intercept_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<BayesOdpFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let boot = bootstrap(n_sims, seed, process, SegmentDependence::Independent)?;
        let model = BayesOdp {
            intercept_sd,
            origin_sd,
            development_sd,
            process: boot.process,
            sampler: BayesSampler {
                chains: whole(chains, "chains")? as usize,
                tune: whole(tune, "tune")? as usize,
                draws: whole(draws, "draws")? as usize,
                seed: 0,
                target_accept,
                max_depth: whole(max_depth, "max_depth")?,
            },
            n_sims: boot.n_sims,
            seed: boot.seed,
            dependence,
        };
        let inner = model.fit_segments(&self.inner, column).map_err(to_r)?;
        Ok(BayesOdpFits { inner })
    }

    /// The one-year view of `bayes_odp` on a single-segment triangle
    /// column by PSIS reweighting: `priors` are `intercept_sd`, `origin_sd`
    /// and `development_sd`, `run` is `chains`, `tune`, `draws`, `n_sims`,
    /// `seed`, `target_accept` and `max_depth`; a simulation whose Pareto
    /// k is above `k_threshold` is handled as `refit`
    /// ("moment_match_then_refit", "refit_flagged", "reweight_only" or
    /// "refit_all") says, refitting at most `max_refits` (`NULL` for no
    /// limit), with `one_year_sims` simulations (`NULL` for the
    /// model's one-year default).
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp_one_year(
        &self,
        column: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
    ) -> Result<BayesOneYearFit> {
        let model = bayes_odp_model(&priors, process, &run, SegmentDependence::Independent)?;
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, one_year_sims)?;
        let inner = model
            .one_year(&self.inner, column, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::Odp(inner),
        })
    }

    /// The one-year view of `bayes_odp_segments` on every segment of a
    /// triangle column, with the settings of `bayes_odp_one_year` and the
    /// segments depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_odp_one_year_segments(
        &self,
        column: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<BayesOneYearFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let model = bayes_odp_model(&priors, process, &run, dependence)?;
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, one_year_sims)?;
        let inner = model
            .one_year_segments(&self.inner, column, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFits {
            inner: BayesOneYearFitsOf::Odp(inner),
        })
    }

    /// Meyers' CRC, CSR or CAY model (`model`, "crc", "csr" or "cay") of a
    /// single-segment triangle column with the premium of each origin from
    /// the `premium` column: normal priors on `logelr` (mean `logelr_mean`,
    /// sd `logelr_sd`), the origin and development effects and `gamma`, a
    /// beta prior with both shapes `rho_shape` on `(1 - rho) / 2`, NUTS with
    /// `chains`, `tune` and `draws` per chain, `target_accept` and
    /// `max_depth`, and `n_sims` predictive simulations, all replayed from
    /// `seed`.
    #[allow(clippy::too_many_arguments)]
    fn meyers(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
    ) -> Result<MeyersFit> {
        let meyers = meyers_settings(
            model,
            [
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
            ],
            [chains, tune, draws, n_sims, seed, target_accept, max_depth],
            SegmentDependence::Independent,
        )?;
        let inner = meyers.fit(&self.inner, column, premium).map_err(to_r)?;
        Ok(MeyersFit { inner })
    }

    /// Meyers' model of every segment of a triangle column, with the
    /// settings of `meyers`, each segment's own posterior, and the segments
    /// depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn meyers_segments(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        logelr_mean: f64,
        logelr_sd: f64,
        origin_sd: f64,
        development_sd: f64,
        gamma_sd: f64,
        rho_shape: f64,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<MeyersFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let meyers = meyers_settings(
            model,
            [
                logelr_mean,
                logelr_sd,
                origin_sd,
                development_sd,
                gamma_sd,
                rho_shape,
            ],
            [chains, tune, draws, n_sims, seed, target_accept, max_depth],
            dependence,
        )?;
        let inner = meyers
            .fit_segments(&self.inner, column, premium)
            .map_err(to_r)?;
        Ok(MeyersFits { inner })
    }

    /// The one-year view of `meyers` on a single-segment triangle column
    /// by PSIS reweighting: `priors` are `logelr_mean`, `logelr_sd`,
    /// `origin_sd`, `development_sd`, `gamma_sd` and `rho_shape`, `run` is
    /// `chains`, `tune`, `draws`, `n_sims`, `seed`, `target_accept` and
    /// `max_depth`; a simulation whose Pareto k is above `k_threshold` is
    /// handled as `refit` ("moment_match_then_refit", "refit_flagged",
    /// "reweight_only" or "refit_all") says.
    #[allow(clippy::too_many_arguments)]
    fn meyers_one_year(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        priors: Vec<f64>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
    ) -> Result<BayesOneYearFit> {
        let meyers = meyers_settings(
            model,
            fixed::<6>(&priors, "priors")?,
            fixed::<7>(&run, "run")?,
            SegmentDependence::Independent,
        )?;
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, one_year_sims)?;
        let inner = meyers
            .one_year(&self.inner, column, premium, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::Meyers(Box::new(inner)),
        })
    }

    /// The one-year view of `meyers_segments` on every segment of a
    /// triangle column, with the settings of `meyers_one_year` and the
    /// segments depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn meyers_one_year_segments(
        &self,
        column: &str,
        premium: &str,
        model: &str,
        priors: Vec<f64>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<BayesOneYearFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let meyers = meyers_settings(
            model,
            fixed::<6>(&priors, "priors")?,
            fixed::<7>(&run, "run")?,
            dependence,
        )?;
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, one_year_sims)?;
        let inner = meyers
            .one_year_segments(&self.inner, column, premium, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFits {
            inner: BayesOneYearFitsOf::Meyers(inner),
        })
    }

    /// The diagonal backtest of `column` over `models`, one `NamedModel`
    /// per spec: each spec is a `list(name, kind, model, process, priors,
    /// run)` where `kind` is "odp_glm", "bayes_odp" or "meyers" (`model`
    /// then names the Meyers model, `priors` the priors of
    /// `bayes_odp_model` or `meyers_settings`, `run` their run settings;
    /// the backtest's own `n_sims` and `seed` replace the spec's).
    /// `premium` is one value per origin (`NULL` for none). Returns the
    /// flat pieces of the `Backtest`; R builds the tables.
    #[allow(clippy::too_many_arguments)]
    fn diagonal_backtest(
        &self,
        column: &str,
        premium: Nullable<Vec<f64>>,
        specs: List,
        n_diagonals: f64,
        n_sims: f64,
        seed: f64,
        interval: f64,
    ) -> Result<List> {
        let premium: Option<Vec<f64>> = premium.into();
        let cells = TriangleFrame::new(&self.inner, column, premium.as_deref()).map_err(to_r)?;
        let mut models = Vec::new();
        for spec in specs.values() {
            let spec = List::try_from(spec)
                .map_err(|_| Error::Other("each model spec must be a list".into()))?;
            let field = |name: &str| {
                spec.dollar(name)
                    .map_err(|_| Error::Other(format!("model spec has no {name}")))
            };
            let text = |name: &str| -> Result<String> {
                String::try_from(field(name)?)
                    .map_err(|_| Error::Other(format!("model spec {name} must be a string")))
            };
            let numbers = |name: &str| -> Result<Vec<f64>> {
                Vec::<f64>::try_from(field(name)?)
                    .map_err(|_| Error::Other(format!("model spec {name} must be numeric")))
            };
            let name = text("name")?;
            let independent = SegmentDependence::Independent;
            let model = match text("kind")?.as_str() {
                "odp_glm" => NamedModel::new(
                    name,
                    GlmCandidate {
                        name: "odp".into(),
                        terms: Terms::new()
                            .intercept()
                            .factor("origin")
                            .factor("development"),
                        glm: Glm::over_dispersed_poisson(),
                    },
                ),
                "bayes_odp" => NamedModel::new(
                    name,
                    bayes_odp_model(
                        &numbers("priors")?,
                        &text("process")?,
                        &numbers("run")?,
                        independent,
                    )?,
                ),
                "meyers" => NamedModel::new(
                    name,
                    meyers_settings(
                        &text("model")?,
                        fixed::<6>(&numbers("priors")?, "priors")?,
                        fixed::<7>(&numbers("run")?, "run")?,
                        independent,
                    )?,
                ),
                "compartmental" => {
                    let ptr = field("ptr")?;
                    let handle = <&CompartmentalBacktestModel>::try_from(&ptr)
                        .map_err(|_| Error::Other("model spec ptr must be a candidate".into()))?;
                    let c = &handle.inner;
                    if c.paid() != column || c.triangle() != &self.inner {
                        return Err(Error::Other(format!(
                            "{} was built on another triangle or paid column than the \
                             backtest's: build it with compartmental_candidate() on this \
                             triangle and column \"{column}\"",
                            c.name()
                        )));
                    }
                    // A clone shares the candidate's fits with the handle.
                    NamedModel::new(name, c.clone())
                }
                other => {
                    return Err(Error::Other(format!("unknown model kind \"{other}\"")));
                }
            };
            models.push(model);
        }
        let refs: Vec<&dyn TriangleModel> =
            models.iter().map(|m| m as &dyn TriangleModel).collect();
        let bt = diagonal_backtest(
            &cells,
            &refs,
            whole(n_diagonals, "n_diagonals")? as usize,
            whole(n_sims, "n_sims")? as usize,
            whole(seed, "seed")?,
            interval,
        )
        .map_err(to_r)?;
        backtest_list(&cells, &bt)
    }

    /// Gesmann and Morris' compartmental model of a single-segment
    /// triangle: paid from the `paid` column, outstanding from
    /// the `outstanding` column ("column") or as the `outstanding` column
    /// of incurred less paid ("incurred"), each origin's premium from the
    /// `premium` column. `priors` and `families` are encoded as
    /// `compartmental_priors_values()` in R encodes them; `settings` are the
    /// likelihood, scale, paid response, fitted responses, rate order and
    /// start; `run` is `chains`, `tune`, `draws`, `n_sims`, `seed`,
    /// `target_accept` and `max_depth`.
    #[allow(clippy::too_many_arguments)]
    fn compartmental(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
    ) -> Result<CompartmentalFit> {
        let model = compartmental_settings(
            &priors,
            &families,
            &settings,
            &run,
            SegmentDependence::Independent,
        )?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        let inner = model.fit(&self.inner, paid, os, premium).map_err(to_r)?;
        Ok(CompartmentalFit { inner })
    }

    /// A `Compartmental` as a candidate of the diagonal backtest on this
    /// triangle: `compartmental`'s arguments, the outstanding backtest of
    /// the handle sharing the fits of the backtests it is put in.
    #[allow(clippy::too_many_arguments)]
    fn compartmental_candidate(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
    ) -> Result<CompartmentalBacktestModel> {
        let model = compartmental_settings(
            &priors,
            &families,
            &settings,
            &run,
            SegmentDependence::Independent,
        )?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        Ok(CompartmentalBacktestModel {
            inner: CompartmentalCandidate::new(model, &self.inner, paid, os, premium),
        })
    }

    /// `compartmental` for every segment of the triangle, each with its own
    /// posterior and the segments depending on each other as `dependence`
    /// ("independent", "rank_correlation" with the row-major `spearman`
    /// matrix, or "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn compartmental_segments(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<CompartmentalFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let model = compartmental_settings(&priors, &families, &settings, &run, dependence)?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        let inner = model
            .fit_segments(&self.inner, paid, os, premium)
            .map_err(to_r)?;
        Ok(CompartmentalFits { inner })
    }

    /// The one-year view of `compartmental` by PSIS reweighting, with the
    /// arguments of `compartmental`; a simulation whose Pareto k is above
    /// `k_threshold` is handled as `refit` says (`NULL` for the model's
    /// default: moment matching, or reweighting only with development
    /// effects), refitting at most `max_refits` (`NULL` for no limit), with
    /// `one_year_sims` simulations (`NULL` for the model's `n_sims` capped
    /// at 2,000).
    #[allow(clippy::too_many_arguments)]
    fn compartmental_one_year(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: Nullable<String>,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
        light_refit: Nullable<Vec<f64>>,
    ) -> Result<CompartmentalOneYearFit> {
        let model = compartmental_settings(
            &priors,
            &families,
            &settings,
            &run,
            SegmentDependence::Independent,
        )?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        let refit = match &refit {
            Nullable::NotNull(r) => Some(r.as_str()),
            Nullable::Null => None,
        };
        let mut view = one_year_settings(k_threshold, refit, max_refits, one_year_sims)?;
        (view.light_refit, view.no_light_refit) = light_refit_arg(light_refit)?;
        let inner = model
            .one_year(&self.inner, paid, os, premium, &view)
            .map_err(to_r)?;
        Ok(CompartmentalOneYearFit { inner })
    }

    /// The one-year view of `compartmental_segments` on every segment of a
    /// triangle, with the arguments of `compartmental_one_year` and the
    /// segments depending on each other as `dependence` ("independent",
    /// "rank_correlation" with the row-major `spearman` matrix, or
    /// "synchronized", an error) says.
    #[allow(clippy::too_many_arguments)]
    fn compartmental_one_year_segments(
        &self,
        paid: &str,
        outstanding: &str,
        outstanding_from: &str,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
        k_threshold: f64,
        refit: Nullable<String>,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
        light_refit: Nullable<Vec<f64>>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
    ) -> Result<CompartmentalOneYearFits> {
        let dependence = segment_dependence(dependence, spearman)?;
        let model = compartmental_settings(&priors, &families, &settings, &run, dependence)?;
        let os = compartmental_outstanding(outstanding, outstanding_from)?;
        let refit = match &refit {
            Nullable::NotNull(r) => Some(r.as_str()),
            Nullable::Null => None,
        };
        let mut view = one_year_settings(k_threshold, refit, max_refits, one_year_sims)?;
        (view.light_refit, view.no_light_refit) = light_refit_arg(light_refit)?;
        let inner = model
            .one_year_segments(&self.inner, paid, os, premium, &view)
            .map_err(to_r)?;
        Ok(CompartmentalOneYearFits { inner })
    }

    /// Paths of the mean outstanding and cumulative paid drawn from the
    /// priors alone: `list(outstanding, paid)`, two predictive
    /// distributions with dimensions `origin` and `age`. The arguments are
    /// those of `compartmental`.
    fn compartmental_prior_predictive(
        &self,
        premium: &str,
        priors: Vec<f64>,
        families: Vec<String>,
        settings: Vec<String>,
        run: Vec<f64>,
    ) -> Result<List> {
        let model = compartmental_settings(
            &priors,
            &families,
            &settings,
            &run,
            SegmentDependence::Independent,
        )?;
        let paths = model
            .prior_predictive(&self.inner, premium, model.n_sims)
            .map_err(to_r)?;
        Ok(list!(
            outstanding = PredictiveDistribution {
                inner: paths.outstanding
            },
            paid = PredictiveDistribution { inner: paths.paid }
        ))
    }

    /// The Bayesian Bornhuetter-Ferguson of a single-segment triangle
    /// column with each origin's exposure from the `exposure` column: a
    /// gamma prior with mean `apriori` times the exposure and coefficient
    /// of variation `cv` (infinite for a flat prior) on each origin's row
    /// parameter (its expected ultimate divided by the sum of the
    /// unnormalised pattern), a normal prior with sd `development_sd` on each
    /// log development effect, `process` error ("gamma" or "none"), NUTS
    /// with `chains`, `tune` and `draws` per chain, `target_accept` and
    /// `max_depth`, and `n_sims` predictive simulations, all replayed from
    /// `seed`.
    #[allow(clippy::too_many_arguments)]
    fn bayes_bf(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
    ) -> Result<BayesBfFit> {
        let model = bayes_bf_model(
            &[apriori, cv, development_sd],
            process,
            &[chains, tune, draws, n_sims, seed, target_accept, max_depth],
        )?;
        let inner = model.fit(&self.inner, column, exposure).map_err(to_r)?;
        Ok(BayesBfFit { inner })
    }

    /// The one-year view of `bayes_bf` on a single-segment triangle column
    /// by PSIS reweighting: `priors` are `apriori`, `cv` and
    /// `development_sd`, `run` is `chains`, `tune`, `draws`, `n_sims`,
    /// `seed`, `target_accept` and `max_depth`; a simulation whose Pareto
    /// k is above `k_threshold` is handled as `refit`
    /// ("moment_match_then_refit", "refit_flagged", "reweight_only" or
    /// "refit_all") says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_bf_one_year(
        &self,
        column: &str,
        exposure: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
    ) -> Result<BayesOneYearFit> {
        let model = bayes_bf_model(&priors, process, &run)?;
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, one_year_sims)?;
        let inner = model
            .one_year(&self.inner, column, exposure, &settings)
            .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::Bf(inner),
        })
    }

    /// The Bayesian Bornhuetter-Ferguson with a normalised development
    /// pattern, of a single-segment triangle column with each origin's
    /// exposure from the `exposure` column: a gamma prior with mean
    /// `apriori` times the exposure and coefficient of variation `cv`
    /// (infinite for a flat prior) on each origin's expected ultimate, a
    /// normal prior with sd `development_sd` on each log ratio of a share
    /// of the ultimate to the first age's, `process` error ("gamma" or "none"), NUTS
    /// with `chains`, `tune` and `draws` per chain, `target_accept` and
    /// `max_depth`, and `n_sims` predictive simulations, all replayed from
    /// `seed`.
    #[allow(clippy::too_many_arguments)]
    fn bayes_bf_normalised(
        &self,
        column: &str,
        exposure: &str,
        apriori: f64,
        cv: f64,
        development_sd: f64,
        process: &str,
        chains: f64,
        tune: f64,
        draws: f64,
        n_sims: f64,
        seed: f64,
        target_accept: f64,
        max_depth: f64,
        common_cv: f64,
        pattern_ages: Nullable<Vec<f64>>,
        pattern_mean: Vec<f64>,
        pattern_sd: Vec<f64>,
    ) -> Result<BayesBfNormalisedFit> {
        let model = bayes_bf_normalised_model(
            &[apriori, cv, development_sd, common_cv],
            process,
            &[chains, tune, draws, n_sims, seed, target_accept, max_depth],
        )?;
        let inner = match pattern_arg(pattern_ages, pattern_mean, pattern_sd)? {
            Some(p) => model.fit_with_pattern_prior(&self.inner, column, exposure, &p),
            None => model.fit(&self.inner, column, exposure),
        }
        .map_err(to_r)?;
        Ok(BayesBfNormalisedFit { inner })
    }

    /// The one-year view of `bayes_bf_normalised` on a single-segment triangle column
    /// by PSIS reweighting: `priors` are `apriori`, `cv` and
    /// `development_sd`, `run` is `chains`, `tune`, `draws`, `n_sims`,
    /// `seed`, `target_accept` and `max_depth`; a simulation whose Pareto
    /// k is above `k_threshold` is handled as `refit`
    /// ("moment_match_then_refit", "refit_flagged", "reweight_only" or
    /// "refit_all") says.
    #[allow(clippy::too_many_arguments)]
    fn bayes_bf_normalised_one_year(
        &self,
        column: &str,
        exposure: &str,
        priors: Vec<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: &str,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
        pattern_ages: Nullable<Vec<f64>>,
        pattern_mean: Vec<f64>,
        pattern_sd: Vec<f64>,
    ) -> Result<BayesOneYearFit> {
        let model = bayes_bf_normalised_model(&priors, process, &run)?;
        let settings = one_year_settings(k_threshold, Some(refit), max_refits, one_year_sims)?;
        let inner = match pattern_arg(pattern_ages, pattern_mean, pattern_sd)? {
            Some(p) => {
                model.one_year_with_pattern_prior(&self.inner, column, exposure, &settings, &p)
            }
            None => model.one_year(&self.inner, column, exposure, &settings),
        }
        .map_err(to_r)?;
        Ok(BayesOneYearFit {
            inner: BayesOneYearOf::BfNormalised(inner),
        })
    }

    /// The hierarchical Bayesian Bornhuetter-Ferguson of a triangle column
    /// with a segment per program and each origin's exposure from the
    /// `exposure` column: `priors` are `apriori`, `cv`, `development_sd`,
    /// `tau_scale` and `omega_scale`, `level_cv` the coefficient of
    /// variation of the industry loss ratio's prior (`NULL` for a flat prior
    /// on its log), `process` error ("gamma" or "none"), `run` is `chains`,
    /// `tune`, `draws`, `n_sims`, `seed`, `target_accept` and `max_depth`,
    /// and the pattern prior is optional.
    #[allow(clippy::too_many_arguments)]
    fn hierarchical_bf(
        &self,
        column: &str,
        exposure: &str,
        priors: Vec<f64>,
        level_cv: Nullable<f64>,
        process: &str,
        run: Vec<f64>,
        pattern_ages: Nullable<Vec<f64>>,
        pattern_mean: Vec<f64>,
        pattern_sd: Vec<f64>,
    ) -> Result<HierarchicalBfFit> {
        let model = hierarchical_bf_model(&priors, level_cv, process, &run)?;
        let inner = match pattern_arg(pattern_ages, pattern_mean, pattern_sd)? {
            Some(p) => model.fit_with_pattern_prior(&self.inner, column, exposure, &p),
            None => model.fit(&self.inner, column, exposure),
        }
        .map_err(to_r)?;
        Ok(HierarchicalBfFit { inner })
    }

    /// The one-year view of `bayes_hierarchical_bf` over every program at
    /// once, by PSIS reweighting: the arguments of `hierarchical_bf`, then
    /// `k_threshold`, `refit` ("moment_match_then_refit", "refit_flagged",
    /// "reweight_only" or "refit_all", or `NULL` for the model's default),
    /// `max_refits`, `one_year_sims`, the
    /// pattern prior and `light_refit` (as `light_refit_arg` reads it).
    #[allow(clippy::too_many_arguments)]
    fn hierarchical_bf_one_year(
        &self,
        column: &str,
        exposure: &str,
        priors: Vec<f64>,
        level_cv: Nullable<f64>,
        process: &str,
        run: Vec<f64>,
        k_threshold: f64,
        refit: Nullable<String>,
        max_refits: Nullable<f64>,
        one_year_sims: Nullable<f64>,
        pattern_ages: Nullable<Vec<f64>>,
        pattern_mean: Vec<f64>,
        pattern_sd: Vec<f64>,
        light_refit: Nullable<Vec<f64>>,
    ) -> Result<HierarchicalBfOneYearFit> {
        let model = hierarchical_bf_model(&priors, level_cv, process, &run)?;
        let refit = match &refit {
            Nullable::NotNull(r) => Some(r.as_str()),
            Nullable::Null => None,
        };
        let mut settings = one_year_settings(k_threshold, refit, max_refits, one_year_sims)?;
        (settings.light_refit, settings.no_light_refit) = light_refit_arg(light_refit)?;
        let inner = match pattern_arg(pattern_ages, pattern_mean, pattern_sd)? {
            Some(p) => {
                model.one_year_with_pattern_prior(&self.inner, column, exposure, &settings, &p)
            }
            None => model.one_year(&self.inner, column, exposure, &settings),
        }
        .map_err(to_r)?;
        Ok(HierarchicalBfOneYearFit { inner })
    }

    /// The lifetime view under Mack's bootstrap: each origin's reserve
    /// simulated to the last age with `process` ("gamma", "lognormal",
    /// "residuals", "normal" or "none"), Mack's model averaged as `average`
    /// with `sigma_interpolation`, the residuals centred before resampling
    /// if `centre_residuals`, and the segments depending on each other as
    /// `dependence` and `spearman` say (as `odp_bootstrap`); then to
    /// ultimate with `tail` (a `ReservingTail`), whose `tail_sigma` and
    /// `tail_std_err` are `NULL` to extrapolate them. With `cells` the fit
    /// keeps each simulation's future incremental values
    /// (`MackBootstrap::fit_segments_with_cells`).
    #[allow(clippy::too_many_arguments)]
    fn mack_bootstrap(
        &self,
        column: &str,
        n_sims: f64,
        seed: f64,
        process: &str,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        centre_residuals: bool,
        tail: Robj,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
        cells: bool,
    ) -> Result<MackBootstrapFit> {
        let boot = MackBootstrap {
            tail: tail_arg(&tail)?,
            tail_sigma: optional(tail_sigma),
            tail_std_err: optional(tail_std_err),
            ..mack_bootstrap(
                n_sims,
                seed,
                process,
                average,
                sigma_interpolation,
                &excl,
                centre_residuals,
                segment_dependence(dependence, spearman)?,
                selected_pattern(&ldf, self.inner.origin_grain())?,
            )?
        };
        let inner = if cells {
            boot.fit_segments_with_cells(&self.inner, column)
        } else {
            boot.fit_segments(&self.inner, column)
        }
        .map_err(to_r)?;
        Ok(MackBootstrapFit { inner })
    }

    /// The one-year view of `method` ("chain_ladder", "expected_loss",
    /// "bornhuetter_ferguson", "benktander" or "cape_cod") by re-reserving
    /// on the ODP bootstrap. `exposure` is `NULL` for the chain ladder and
    /// names the exposure column of the other methods; each method reads
    /// only its own settings (`apriori`, `n_iters`, `trend`, `decay`). The
    /// segments depend on each other as `dependence` and `spearman` say (as
    /// `odp_bootstrap`). With `bootstrap_tail`, the bootstrap takes
    /// `method`'s `tail` too, with `tail_std_err` (`NULL` to keep a constant
    /// tail fixed), and the origins whose tail step falls in the coming year
    /// take it.
    #[allow(clippy::too_many_arguments)]
    fn odp_one_year(
        &self,
        column: &str,
        method: &str,
        exposure: Nullable<String>,
        apriori: Robj,
        n_iters: f64,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        bootstrap_tail: bool,
        tail_std_err: Nullable<f64>,
        n_sims: f64,
        seed: f64,
        process: &str,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
        bootstrap_ldf: bool,
    ) -> Result<OneYearFit> {
        let boot_tail = if bootstrap_tail {
            tail_arg(&tail)?
        } else {
            Tail::default()
        };
        let method = one_year_method(
            method,
            exposure,
            apriori,
            n_iters,
            trend,
            decay,
            pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.inner.origin_grain(),
            )?,
        )?;
        let dependence = segment_dependence(dependence, spearman)?;
        let inner = OdpBootstrap {
            tail: boot_tail,
            tail_std_err: optional(tail_std_err),
            one_year_pattern: one_year_pattern(bootstrap_ldf),
            ..bootstrap(n_sims, seed, process, dependence)?
        }
        .one_year_segments(&self.inner, column, &method)
        .map_err(to_r)?;
        Ok(OneYearFit {
            inner: OneYear::Odp(inner),
        })
    }

    /// The one-year view of `method`, as `odp_one_year`, under Mack's
    /// process: Mack's bootstrap with `process` ("gamma", "lognormal",
    /// "residuals", "normal" or "none"), Mack's model averaged as
    /// `mack_average` with `mack_sigma_interpolation`, the residuals centred
    /// before resampling if `centre_residuals`, and the segments depending
    /// on each other as `dependence` and `spearman` say. With
    /// `bootstrap_tail`, Mack's bootstrap takes `method`'s `tail` too, its
    /// `tail_sigma` and `tail_std_err` `NULL` to extrapolate them.
    #[allow(clippy::too_many_arguments)]
    fn mack_one_year(
        &self,
        column: &str,
        method: &str,
        exposure: Nullable<String>,
        apriori: Robj,
        n_iters: f64,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        bootstrap_tail: bool,
        tail_sigma: Nullable<f64>,
        tail_std_err: Nullable<f64>,
        n_sims: f64,
        seed: f64,
        process: &str,
        mack_average: &str,
        mack_sigma_interpolation: &str,
        mack_excl: Robj,
        centre_residuals: bool,
        dependence: &str,
        spearman: Nullable<Vec<f64>>,
        ldf: Robj,
        bootstrap_ldf: bool,
    ) -> Result<OneYearFit> {
        let boot_tail = if bootstrap_tail {
            tail_arg(&tail)?
        } else {
            Tail::default()
        };
        let method = one_year_method(
            method,
            exposure,
            apriori,
            n_iters,
            trend,
            decay,
            pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.inner.origin_grain(),
            )?,
        )?;
        let inner = MackBootstrap {
            tail: boot_tail,
            tail_sigma: optional(tail_sigma),
            tail_std_err: optional(tail_std_err),
            one_year_pattern: one_year_pattern(bootstrap_ldf),
            ..mack_bootstrap(
                n_sims,
                seed,
                process,
                mack_average,
                mack_sigma_interpolation,
                &mack_excl,
                centre_residuals,
                segment_dependence(dependence, spearman)?,
                None,
            )?
        }
        .one_year_segments(&self.inner, column, &method)
        .map_err(to_r)?;
        Ok(OneYearFit {
            inner: OneYear::Mack(inner),
        })
    }

    /// Clark's LDF method; `max_age` is `Inf` to develop to infinity.
    fn clark_ldf(&self, column: &str, curve: &str, max_age: f64) -> Result<ClarkFit> {
        let inner = ClarkLdf {
            curve: growth_curve(curve)?,
            max_age: clark_max_age(max_age),
        }
        .fit_segments(&self.inner, column)
        .map_err(to_r)?;
        Ok(ClarkFit { inner })
    }

    /// Clark's Cape Cod method, with each origin's exposure the latest
    /// value of column `exposure`.
    fn clark_cape_cod(
        &self,
        column: &str,
        exposure: &str,
        curve: &str,
        max_age: f64,
    ) -> Result<ClarkFit> {
        let inner = ClarkCapeCod {
            curve: growth_curve(curve)?,
            max_age: clark_max_age(max_age),
        }
        .fit_segments(&self.inner, column, exposure)
        .map_err(to_r)?;
        Ok(ClarkFit { inner })
    }

    /// Merz and Wuthrich's paid-incurred chain on the `paid` and `incurred`
    /// columns (`reserving-v02.md`, decision 32). `sigma` and `tau` are
    /// `NULL` to estimate them; an empty prior pair leaves that side
    /// non-informative.
    #[allow(clippy::too_many_arguments)]
    fn paid_incurred_chain(
        &self,
        paid: &str,
        incurred: &str,
        sigma: Nullable<Vec<f64>>,
        tau: Nullable<Vec<f64>>,
        extrapolation: &str,
        convention: &str,
        phi_mean: Vec<f64>,
        phi_sd: Vec<f64>,
        psi_mean: Vec<f64>,
        psi_sd: Vec<f64>,
    ) -> Result<PaidIncurredFit> {
        let given = |x: Nullable<Vec<f64>>| match x {
            Nullable::NotNull(v) => Some(v),
            Nullable::Null => None,
        };
        let extrapolation = VarianceExtrapolation::from_name(extrapolation).ok_or_else(|| {
            crate::unknown_name("extrapolation", extrapolation, VarianceExtrapolation::NAMES)
        })?;
        let convention = PicConvention::from_name(convention)
            .ok_or_else(|| crate::unknown_name("convention", convention, PicConvention::NAMES))?;
        let inner = PaidIncurredChain {
            priors: PicPriors {
                phi_mean,
                phi_sd,
                psi_mean,
                psi_sd,
            },
            sigma: given(sigma),
            tau: given(tau),
            extrapolation,
            convention,
        }
        .fit(&self.inner, paid, incurred)
        .map_err(to_r)?;
        Ok(PaidIncurredFit { inner })
    }
}

extendr_module! {
    mod triangle;
    impl Triangle;
}
