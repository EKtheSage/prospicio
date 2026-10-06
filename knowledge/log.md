# Bundle history

## 2026-10-05

* **Initialization**: Created the bundle with [datasets](/datasets/index.md), [references](/references/index.md), [findings](/findings/index.md) and [environment](/environment/index.md), from knowledge collected while building the Probability, Aggregate and Models lanes (PRs #104 to #116).
* **Expected-loss reference**: Added [chainladder-python expected-loss estimators](/references/chainladder-python-expected-loss.md), from the Reserving lane's expected-loss methods (branch claude/v02-expected-loss).
* **Clark references**: Added [R ChainLadder ClarkLDF and ClarkCapeCod](/references/r-chainladder-clark.md) and [chainladder-python ClarkLDF](/references/chainladder-python-clark.md), from the Reserving lane's Clark methods (branch claude/v02-clark).
* **Clark ELR cap**: Recorded in [R ChainLadder ClarkLDF and ClarkCapeCod](/references/r-chainladder-clark.md) that R bounds the Cape Cod ELR at 10 without a warning, where act_reserving is unbounded (branch claude/v02-clark).
* **Expected-loss reference update**: [chainladder-python expected-loss estimators](/references/chainladder-python-expected-loss.md) now records the Benktander closed form, its divergence below cdf 1/2, and how incremental exposure and off-diagonal origins differ from act_reserving (branch claude/v02-expected-loss).
