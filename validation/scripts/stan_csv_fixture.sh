#!/usr/bin/env bash
# Regenerate validation/data/stan_eight_schools/chain-{1..4}.csv, real
# CmdStan output for the Stan CSV reader (#395), then the reference values
# with validation/scripts/r_stan_csv.R.
#
#   CMDSTAN=/path/to/cmdstan-2.36.0 validation/scripts/stan_csv_fixture.sh
#
# Four chains of 250 draws after 150 warmup draws, saved (save_warmup=1) so
# the reader's warmup handling runs on real output; seed 20261023.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(cd "$here/../data" && pwd)/stan_eight_schools
work=$(mktemp -d)
cp "$here/stan_eight_schools.stan" "$work/eight_schools.stan"
make -C "$CMDSTAN" "$work/eight_schools" > /dev/null
cat > "$work/data.json" <<'JSON'
{"J": 8, "y": [28, 8, -3, 7, -1, 1, 18, 12], "sigma": [15, 10, 16, 11, 9, 11, 10, 18]}
JSON
mkdir -p "$out"
for c in 1 2 3 4; do
  "$work/eight_schools" sample num_warmup=150 num_samples=250 save_warmup=1 \
    data file="$work/data.json" random seed=20261023 id=$c \
    output file="$out/chain-$c.csv" > /dev/null
done
# Keep the files free of the machine's paths.
sed -i -e "s|$work|<tmp>|g" -e "s|$out|validation/data/stan_eight_schools|g" "$out"/chain-*.csv
Rscript "$here/r_stan_csv.R"
