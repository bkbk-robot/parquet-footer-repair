#!/usr/bin/env python3
"""Self-contained round-trip test: write a Parquet, delete its footer, repair it, compare.

Run: python tests/test_roundtrip.py
No network, no fixture files: the reference file is generated on the fly.
"""

import io
import os
import struct
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repair  # noqa: E402


def strip_footer(path, out_path):
    """Cut the FileMetaData + trailing length + magic, keeping all page data."""
    with open(path, "rb") as fh:
        raw = fh.read()
    meta_len = struct.unpack("<I", raw[-8:-4])[0]
    cut = len(raw) - 8 - meta_len
    with open(out_path, "wb") as fh:
        fh.write(raw[:cut])
    return cut, meta_len


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pfr-")

    def _write(self, table, row_group_size, compression="snappy"):
        path = os.path.join(self.tmp, "ref.parquet")
        pq.write_table(table, path, row_group_size=row_group_size, compression=compression)
        return path

    def _roundtrip(self, table, row_group_size, compression="snappy"):
        ref = self._write(table, row_group_size, compression)
        broken = os.path.join(self.tmp, "broken.parquet")
        fixed = os.path.join(self.tmp, "fixed.parquet")
        cut, meta_len = strip_footer(ref, broken)

        # the truncated file must be unreadable before repair
        with self.assertRaises(Exception):
            pq.ParquetFile(broken).metadata

        repair.repair(broken, ref, fixed, rows_per_group=row_group_size)
        recovered = pq.read_table(fixed)
        expected = pq.read_table(ref)
        self.assertEqual(recovered.schema, expected.schema)
        self.assertEqual(recovered.num_rows, expected.num_rows)
        self.assertTrue(recovered.equals(expected), "recovered table differs from original")
        return cut, meta_len, recovered

    def test_mixed_columns(self):
        table = pa.table({
            "index": pa.array(range(12), pa.int64()),
            "timestamp": pa.array([i * 0.033 for i in range(12)], pa.float64()),
            "label": pa.array([f"step-{i % 3}" for i in range(12)]),
            "flag": pa.array([i % 2 == 0 for i in range(12)]),
        })
        cut, meta_len, recovered = self._roundtrip(table, row_group_size=4)
        print(f"  mixed_columns: footer {meta_len} B at offset {cut}, "
              f"{recovered.num_rows} rows recovered")
        self.assertEqual(recovered.num_rows, 12)

    def test_uncompressed(self):
        table = pa.table({"a": pa.array(range(6), pa.int32()),
                          "b": pa.array([i * 1.5 for i in range(6)], pa.float32())})
        self._roundtrip(table, row_group_size=3, compression="none")

    def test_zstd(self):
        table = pa.table({"a": pa.array(range(8), pa.int64()),
                          "b": pa.array(["x"] * 8)})
        self._roundtrip(table, row_group_size=4, compression="zstd")

    def test_inferred_row_group_size(self):
        """--rows-per-group omitted: fall back to the reference's row group size."""
        table = pa.table({"a": pa.array(range(9), pa.int64())})
        ref = self._write(table, row_group_size=3)
        broken = os.path.join(self.tmp, "broken2.parquet")
        fixed = os.path.join(self.tmp, "fixed2.parquet")
        strip_footer(ref, broken)
        repair.repair(broken, ref, fixed)
        self.assertTrue(pq.read_table(fixed).equals(pq.read_table(ref)))

    def test_partial_trailing_row_group_is_dropped(self):
        """A crashed writer leaves a half-written last row group: keep the complete ones."""
        table = pa.table({"a": pa.array(range(16), pa.int64()),
                          "b": pa.array([float(i) for i in range(16)])})
        ref = self._write(table, row_group_size=4)
        with open(ref, "rb") as fh:
            raw = fh.read()
        meta_len = struct.unpack("<I", raw[-8:-4])[0]
        body = raw[: len(raw) - 8 - meta_len]
        broken = os.path.join(self.tmp, "broken3.parquet")
        with open(broken, "wb") as fh:
            fh.write(body[:-80])  # last page cut mid-body
        fixed = os.path.join(self.tmp, "fixed3.parquet")
        repair.repair(broken, ref, fixed, rows_per_group=4)
        recovered = pq.read_table(fixed)
        self.assertEqual(recovered.num_columns, 2)
        self.assertEqual(recovered.num_rows, 12)  # 3 complete row groups of 4
        self.assertTrue(recovered.to_pydict()["a"] == list(range(12)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
