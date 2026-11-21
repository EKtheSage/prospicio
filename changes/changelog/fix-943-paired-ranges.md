### Excel

- Functions that take ranges meant to line up row by row (`PRO.PRC.PROFILE`,
  the tabulated exposure curve, the piecewise Pareto, payout patterns, rate
  histories, sliding-scale anchors, outcome tables with probabilities, the
  benchmark curve) read them together. A row blank in every range is still
  dropped; a row blank in one range and not in another is now an error naming
  the range and the cell, where each range's blanks were dropped on their own
  and the values paired one row off. Ranges of different sizes are an error
  (#1000).
