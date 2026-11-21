### Excel

- `PRO.CAP.CASHFLOW` reads `u` and a per-simulation `commission` range together. A row blank in one range and not in the other is now an error naming the cell, where each range dropped its blanks on its own and the commissions were paired with the wrong loss ratios. A row blank in both is still dropped, and one commission for all simulations or none is read as before.
