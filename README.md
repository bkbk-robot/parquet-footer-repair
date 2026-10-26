# parquet-footer-repair

Recover a Parquet file whose **footer was never written** — the classic result of a
training/recording job that got killed (SIGTERM, OOM, `Ctrl-C`, power loss) before the
writer could close the file.

The data pages are still on disk and intact. Only the trailing `FileMetaData` (thrift
compact) and the 8-byte footer length/magic are missing, so every reader refuses to open
the file.

`repair.py` rebuilds that footer by scanning the page headers, then re-encodes a valid
`FileMetaData`. The schema, key/value metadata, `created_by` and `column_orders` are
copied byte-for-byte from a reference file with the same schema.

```
$ python repair.py --broken data/chunk-000/file-000.parquet --ref good.parquet --out fixed.parquet
pages scanned      : 20 row group(s)
row groups written : 20 (1946, 1946, 1946, 1946, 1946, 1946, ...)
rows recovered     : 38914
footer size        : 11732 bytes
written            : fixed.parquet (1470494 bytes)
verify             : readable, 38914 rows, 7 columns
```

## Why this exists

A dataset recording run of a SO-101 arm was killed by a 30-minute command timeout at
39 episodes. LeRobot's Parquet writer only writes the footer on close, so the file
became unreadable while all 35,022 frames (39 × 898) were still on disk. Re-recording
meant hours of teleoperation; the data was fine, only the index was missing.

This tool recovered all 39 episodes, with the per-column statistics matching the values
computed from the recovered data bit-for-bit, and `LeRobotDataset` loading the result
without complaint.

## Install

Python ≥ 3.9 and `pyarrow` — nothing else:

```bash
pip install pyarrow
```

## Usage

```bash
python repair.py \
  --broken  data/chunk-000/file-000.parquet \   # file missing its footer
  --ref     good.parquet \                      # intact file, same schema
  --out     fixed.parquet \
  --rows-per-group 1946                         # optional, see below
```

| Flag | Meaning |
| --- | --- |
| `--broken` | the damaged file (must still start with `PAR1`) |
| `--ref` | an intact file with the **same schema and column order**; row count doesn't matter |
| `--out` | where to write the repaired copy (input is never touched) |
| `--rows-per-group` | rows per row group as the producer wrote them (e.g. `898` frames). Defaults to the reference's row-group size |
| `--no-verify` | skip the read-back check |

Exit code `0` on success, `2` if the repaired file cannot be read back.

Always copy the damaged file first and repair into a **new** path — this tool only reads
`--broken`.

## What it handles, and what it doesn't

Works when:

- pages are intact and only the footer is missing (the common crash case);
- the last row group is partially written — incomplete groups are dropped, complete ones
  are kept and reported;
- dictionary-encoded columns, nested/list columns (`observation.state`, `action`),
  SNAPPY / ZSTD / GZIP / uncompressed data;
- row groups whose sizes differ slightly from the reference (per-group row counts are
  recomputed from the recovered pages, not assumed).

Does not work / limits:

- **truncated page data inside a needed row group** — pages cannot be rebuilt, only the
  index can;
- files whose columns have **no dictionary pages and no consistent row-group size**
  (no reliable chunk boundary; the scan will say so by dropping or mis-splitting groups —
  check the reported row count against what you expect);
- encrypted Parquet, or column/offset indexes that were never flushed (they are optional
  metadata; readers regenerate or ignore them);
- one row-group layout assumption: each row group is written column by column, as
  pyarrow, pandas and LeRobot do.

## Verify it yourself

```bash
python tests/test_roundtrip.py
```

Five self-contained round-trip tests (no fixtures, no network): each writes a Parquet
file, cuts the footer, repairs it with the file itself as reference, then asserts the
recovered table is **equal** to the original — mixed types, uncompressed, ZSTD,
inferred row-group size, and a truncated trailing row group.

Measured on real data (2026-10, pyarrow 25.0.1):

| File | Rows | Row groups | Result |
| --- | --- | --- | --- |
| LeRobot `data/file-000.parquet` (7 cols incl. nested list) | 38,914 | 20 | 100 % rows, `Table.equals()` → True |
| LeRobot `block_task_v1` | 8,980 | 5 | 100 % rows, `Table.equals()` → True |
| LeRobot `meta/episodes/file-000.parquet` | 20 | 2 | 100 % rows, `Table.equals()` → True |
| Crash case that motivated the tool | 35,022 (39 eps × 898) | 39 | all episodes recovered, `LeRobotDataset` loads |

## How it works

1. Read the reference footer and keep the raw bytes of the schema list, `key_value_metadata`,
   `created_by` and `column_orders`.
2. Walk the damaged file page by page, parsing thrift-compact `PageHeader`s. A dictionary
   page starts a new column chunk; value counts come from the data page headers.
3. Group chunks into row groups (one row group = one chunk per column), recompute each
   group's row count from the repetition factors of the schema.
4. Re-encode `FileMetaData` → `RowGroup` → `ColumnChunk` → `ColumnMetaData` in thrift
   compact, append the 4-byte footer length and the trailing `PAR1`.
5. Read the result back with `pyarrow` and print row/column counts and null counts.

## 中文速览

Parquet 写入进程被 kill 时，数据页还在，但尾部 `FileMetaData` 没写 → 整个文件读不了。
本工具扫描页头重建 footer，schema 等元数据从同构完好文件按字节复用。
用法：`python repair.py --broken 坏文件 --ref 完好参考文件 --out 新文件 [--rows-per-group N]`。
实测 LeRobot 39 集 35,022 帧全量恢复；自带 5 个往返测试（截掉 footer 再修，比对全等）。

## License

MIT
