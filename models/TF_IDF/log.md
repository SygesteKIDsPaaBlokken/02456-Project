# Keeping TF and IDF dicts
Gives a search time of 300 docs per sec, meaning 8_841_823/300 /60/60 = 8.186873148148148 hours for a search...

**With cdist:** ~200 it/s
**With custom:** ~200 it/s
**With 1 - cosine:** < 200 it/s

# TF.IDF matrix
Requires a lot more memory, an thus chunking. Estimated space required 8_841_823/10_000 * 3 GB = 880 GB...
If there was a way to remove the sparsity of the matrix...

However a query