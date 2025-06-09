import lancedb

tbl = lancedb.connect("data/output")["Brochure"]

print(tbl.count_rows())

print(tbl.search().to_pandas().head(20))