### Distributions and probability

- `iman_conover`, `iman_conover_with` and the group reordering by correlation sort
  their columns on several threads once there are 4096 simulations or more and
  two or more components; the results are bit-for-bit the same (#971).
