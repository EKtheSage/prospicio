### Pricing

- An ILF or ELF table read a rounding error past its last limit (for
  example `0.2 + 0.4` on a table ending at `0.6`) returns the last value
  instead of NaN, so a casualty profile's exposure factors stay finite (#939, #994).
