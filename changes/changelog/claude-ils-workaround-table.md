### Examples

- `ils_deal_analysis` and its R twin use the fixes of the closed workaround issues: `monthly_patterns`
  (#601), which also corrects an earned pattern that fell after month 24 and held the flat
  structure's median IRR at 18% where it is 40% (2,000 simulations), `loss_ratio_draws` (#602)
  and `deal_ave` on an illustrative actual paid (#603); its workaround table gives each row's
  status, with #600 kept on purpose and #605 done by `submission-schema`. The paid pattern's
  basis is #722.
