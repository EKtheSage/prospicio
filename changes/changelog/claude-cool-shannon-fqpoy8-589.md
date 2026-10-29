### Pricing

- A deal's and a book's price adequacy in one table: `DealBook::adequacy(rule, assets)` gives each deal and the book the premium charged and kept (less every deduction), the expected loss and margin, the allocated (Euler) and marginal assets and capital and the return on each, the technical and marginal premiums, the target premium that returns the book's target on the allocated capital, and the adequacy, kept over target (`time-value.md` decision 36). Python `DealBook.adequacy`, R `book_adequacy()` (#589).
