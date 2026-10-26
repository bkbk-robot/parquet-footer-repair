#!/usr/bin/env python3
"""Repair a Parquet file whose footer was never written.

Symptom: `pyarrow.parquet.ParquetFile(path)` raises
`ArrowInvalid: Parquet file size is ... smaller than footer` or
`Could not open Parquet input source: ... Expected magic bytes`. The file
starts with the 4-byte magic `PAR1` and contains intact page data, but the
trailing FileMetaData (thrift compact) plus the 8-byte footer is missing,
because the writer process was killed (SIGTERM/SIGKILL, OOM, power loss)
before it could close the file.

Strategy: the data pages are fine. Scan the page headers to recover each
column chunk's offsets, value counts and byte sizes, then re-encode a valid
FileMetaData. Schema, key/value metadata, created_by and column_orders are
borrowed byte-for-byte from a reference file with the same schema.

Reference file ("--ref") must be an intact Parquet file with the same
schema and the same physical column order. It does not need the same number
of rows or row groups: the per-row-group value counts are supplied by
--rows-per-group (or inferred from the reference's column/row ratio).

Usage:
    python repair.py --broken broken.parquet --ref good.parquet --out fixed.parquet \
        [--rows-per-group 898] [--no-verify]

Exit code 0 on success, 2 when the repaired file cannot be read back.
"""

from __future__ import annotations

import argparse
import struct
import sys

import pyarrow.parquet as pq

# ---------------------------------------------------------------- thrift compact
T_STOP, T_TRUE, T_FALSE, T_BYTE = 0, 1, 2, 3
T_I16, T_I32, T_I64, T_DOUBLE = 4, 5, 6, 7
T_BIN, T_LIST, T_SET, T_MAP, T_STRUCT = 8, 9, 10, 11, 12

PHYS_TYPE = {
    "BOOLEAN": 0,
    "INT32": 1,
    "INT64": 2,
    "INT96": 3,
    "FLOAT": 4,
    "DOUBLE": 5,
    "BYTE_ARRAY": 6,
    "FIXED_LEN_BYTE_ARRAY": 7,
}

CODEC = {
    "UNCOMPRESSED": 0,
    "SNAPPY": 1,
    "GZIP": 2,
    "LZO": 3,
    "BROTLI": 4,
    "LZ4": 5,
    "ZSTD": 6,
    "LZ4_RAW": 7,
}

ENCODING = {
    "PLAIN": 0,
    "PLAIN_DICTIONARY": 2,
    "RLE": 3,
    "BIT_PACKED": 4,
    "DELTA_BINARY_PACKED": 5,
    "DELTA_LENGTH_BYTE_ARRAY": 6,
    "DELTA_BYTE_ARRAY": 7,
    "RLE_DICTIONARY": 8,
    "BYTE_STREAM_SPLIT": 9,
}

PAGE_TYPE = {0: "DATA_PAGE", 1: "INDEX_PAGE", 2: "DICTIONARY_PAGE", 3: "DATA_PAGE_V2"}


def read_uvarint(buf, pos):
    """Read a thrift compact unsigned varint; return (value, new_pos)."""
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def read_zigzag(buf, pos):
    value, pos = read_uvarint(buf, pos)
    return (value >> 1) ^ -(value & 1), pos


def skip_value(buf, pos, ftype):
    """Skip one thrift compact value of the given type field type."""
    if ftype in (T_TRUE, T_FALSE):
        return pos
    if ftype == T_BYTE:
        return pos + 1
    if ftype in (T_I16, T_I32, T_I64):
        _, pos = read_zigzag(buf, pos)
        return pos
    if ftype == T_DOUBLE:
        return pos + 8
    if ftype == T_BIN:
        size, pos = read_uvarint(buf, pos)
        return pos + size
    if ftype in (T_LIST, T_SET):
        header = buf[pos]
        pos += 1
        elem_type = header & 0x0F
        size = header >> 4
        if size == 15:
            size, pos = read_uvarint(buf, pos)
        for _ in range(size):
            pos = skip_value(buf, pos, elem_type)
        return pos
    if ftype == T_MAP:
        header = buf[pos]
        pos += 1
        size = header >> 4
        if size == 15:
            size, pos = read_uvarint(buf, pos)
        if size == 0:
            return pos
        kv = buf[pos]
        pos += 1
        ktype, vtype = kv >> 4, kv & 0x0F
        for _ in range(size):
            pos = skip_value(buf, pos, ktype)
            pos = skip_value(buf, pos, vtype)
        return pos
    if ftype == T_STRUCT:
        while True:
            header = buf[pos]
            if header == 0:
                return pos + 1
            ftype2 = header & 0x0F
            delta = header >> 4
            pos += 1
            if delta == 0:
                _, pos = read_zigzag(buf, pos)
            pos = skip_value(buf, pos, ftype2)
    raise ValueError(f"unsupported thrift type {ftype} at offset {pos}")


def parse_fields(buf, pos, end=None):
    """Parse a thrift compact struct into {field_id: (type, value_start, value_end)}."""
    fields = {}
    field_id = 0
    while end is None or pos < end:
        header = buf[pos]
        if header == 0:
            return fields, pos + 1
        ftype = header & 0x0F
        delta = header >> 4
        pos += 1
        if delta == 0:
            field_id, pos = read_zigzag(buf, pos)
        else:
            field_id += delta
        start = pos
        pos = skip_value(buf, pos, ftype)
        fields[field_id] = (ftype, start, pos)
    return fields, pos


def parse_page_header(buf, pos):
    fields, next_pos = parse_fields(buf, pos)
    page_type = uncompressed = compressed = num_values = None
    if 1 in fields:
        page_type, _ = read_zigzag(buf, fields[1][1])
    if 2 in fields:
        uncompressed, _ = read_zigzag(buf, fields[2][1])
    if 3 in fields:
        compressed, _ = read_zigzag(buf, fields[3][1])
    # Number of rows lives in the page-type-specific sub-struct:
    # data_page_header (field 5) for DATA_PAGE, data_page_header_v2 (field 8) for DATA_PAGE_V2.
    for field_id in (5, 8):
        if field_id in fields:
            sub, _ = parse_fields(buf, fields[field_id][1])
            if 1 in sub:
                num_values, _ = read_zigzag(buf, sub[1][1])
            break
    return {
        "type": page_type,
        "uncompressed": uncompressed,
        "compressed": compressed,
        "num_values": num_values,
    }, next_pos - pos


# ---------------------------------------------------------------- thrift compact writer
def write_uvarint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def write_zigzag(value):
    return write_uvarint((value << 1) ^ (value >> 63) if value < 0 else (value << 1))


def field_header(field_id, last_id, ftype):
    delta = field_id - last_id
    if 0 < delta <= 15:
        return bytes([(delta << 4) | ftype])
    return bytes([ftype]) + write_zigzag(field_id)


def list_header(size, elem_type):
    if size < 15:
        return bytes([(size << 4) | elem_type])
    return bytes([0xF0 | elem_type]) + write_uvarint(size)


class StructWriter:
    """Minimal thrift compact struct encoder with delta field ids."""

    def __init__(self):
        self.buf = bytearray()
        self.last_id = 0

    def i32(self, field_id, value):
        self.buf += field_header(field_id, self.last_id, T_I32)
        self.last_id = field_id
        self.buf += write_zigzag(value)

    def i64(self, field_id, value):
        self.buf += field_header(field_id, self.last_id, T_I64)
        self.last_id = field_id
        self.buf += write_zigzag(value)

    def binary(self, field_id, value):
        if isinstance(value, str):
            value = value.encode()
        self.buf += field_header(field_id, self.last_id, T_BIN)
        self.last_id = field_id
        self.buf += write_uvarint(len(value)) + value

    def list_of_binary(self, field_id, values):
        """Write list<string>: one list header followed by bare length-prefixed values."""
        self.buf += field_header(field_id, self.last_id, T_LIST)
        self.last_id = field_id
        self.buf += list_header(len(values), T_BIN)
        for value in values:
            payload = value.encode() if isinstance(value, str) else value
            self.buf += write_uvarint(len(payload)) + payload

    def raw(self, field_id, ftype, payload):
        """Embed already-encoded bytes, e.g. a list copied from the reference footer."""
        self.buf += field_header(field_id, self.last_id, ftype)
        self.last_id = field_id
        self.buf += payload

    def close(self):
        return bytes(self.buf) + b"\x00"


# ---------------------------------------------------------------- reference metadata
def ref_footer(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != b"PAR1" or raw[-4:] != b"PAR1":
        raise SystemExit(f"{path}: not a Parquet file (magic bytes missing)")
    meta_len = struct.unpack("<I", raw[-8:-4])[0]
    footer = raw[-8 - meta_len:-8]
    fields, _ = parse_fields(footer, 0)
    return fields, footer


def plan_from_ref(ref_path, rows_per_group=None):
    """Derive schema bytes and per-column layout information from the reference file."""
    meta = pq.ParquetFile(ref_path).metadata
    rg0 = meta.row_group(0)
    n_cols = rg0.num_columns
    rows0 = rg0.num_rows
    cols = [rg0.column(j) for j in range(n_cols)]

    paths = [c.path_in_schema for c in cols]
    phys = [c.physical_type for c in cols]
    factors = []
    for c in cols:
        ratio = c.num_values / rows0 if rows0 else 1
        if ratio != int(ratio):
            raise SystemExit(
                f"column {c.path_in_schema}: num_values/rows = {ratio} is not an integer; "
                "cannot infer repetition factors"
            )
        factors.append(int(ratio))

    codecs = {c.compression for c in cols}
    if len(codecs) != 1:
        raise SystemExit(f"reference mixes compression codecs: {sorted(codecs)}")
    codec = codecs.pop()

    encodings = []
    for c in cols:
        missing = [e for e in c.encodings if e not in ENCODING]
        if missing:
            raise SystemExit(f"column {c.path_in_schema}: unsupported encodings {missing}")
        encodings.append([ENCODING[e] for e in c.encodings])

    rows = rows_per_group if rows_per_group else rows0
    expect = [f * rows for f in factors]
    return {
        "paths": paths,
        "phys": phys,
        "factors": factors,
        "codec": codec,
        "encodings": encodings,
        "rows": rows,
        "expect": expect,
    }


# ---------------------------------------------------------------- page scan
def scan_chunks(raw, expect):
    """Walk page headers of `raw` and group them into column chunks and row groups."""
    n_cols = len(expect)
    chunks, groups = [], []
    pos = 4  # after leading PAR1
    current = None      # chunk under construction
    col_index = 0
    partial = False

    def close_chunk(end):
        nonlocal current, chunks, col_index
        current["end"] = end
        chunks.append(current)
        current = None
        col_index += 1
        if col_index == n_cols:
            groups.append(chunks)
            chunks = []
            col_index = 0

    while pos < len(raw):
        try:
            header, header_len = parse_page_header(raw, pos)
        except (ValueError, IndexError):
            partial = True  # truncated page header: stop at the last complete row group
            break
        page_type = header["type"]
        if page_type is None:
            partial = True
            break
        page_end = pos + header_len + (header["compressed"] or 0)
        if page_end > len(raw):
            partial = True  # page body cut short by the crash
            break
        if page_type == 2:  # dictionary page starts a new column chunk
            if current is not None:
                close_chunk(pos)
            current = {"start": pos, "dict_off": pos, "data_off": None,
                       "num_values": 0, "uncompressed": 0}
        elif page_type in (0, 3):
            if current is None:
                current = {"start": pos, "dict_off": None, "data_off": None,
                           "num_values": 0, "uncompressed": 0}
            if current["data_off"] is None:
                current["data_off"] = pos
            current["num_values"] += header["num_values"] or 0
        else:  # index page: nothing to record
            pos = page_end
            continue

        current["uncompressed"] += header_len + (header["uncompressed"] or 0)
        pos = page_end

        # Two closing rules, whichever fires first:
        #  - a dictionary page always starts the next column chunk;
        #  - the value count reaching the expected count (columns without
        #    dictionary encoding, e.g. booleans, carry no start marker).
        # Row-group sizes may vary slightly between groups; the count rule uses the
        # reference-derived estimate, and per-group row counts are recomputed after
        # the scan, so one-group drift is tolerated.
        if page_type in (0, 3) and current["num_values"] >= expect[col_index]:
            close_chunk(pos)

    if current is not None:
        if partial:
            current = None  # truncated tail: drop it
        else:
            close_chunk(pos)
    if chunks:
        partial = True  # trailing row group with fewer columns than the schema

    return groups, partial


def build_footer(raw, groups, plan, ref_fields, reference_footer):
    n_cols = len(plan["paths"])
    complete = [g for g in groups if len(g) == n_cols]
    dropped = len(groups) - len(complete)
    rows_per_group = [
        min(g[j]["num_values"] // plan["factors"][j] for j in range(n_cols)) for g in complete
    ]
    num_rows = sum(rows_per_group)

    meta = StructWriter()
    meta.i32(1, 2)                                     # version
    ftype, start, end = ref_fields[2]                  # schema: byte-for-byte copy
    meta.raw(2, ftype, reference_footer[start:end])

    rg_bufs = []
    for group, rows in zip(complete, rows_per_group):
        rg = StructWriter()
        rg.buf += field_header(1, rg.last_id, T_LIST) + list_header(n_cols, T_STRUCT)
        rg.last_id = 1
        for j, chunk in enumerate(group):
            chunk_meta = StructWriter()
            chunk_meta.i32(1, PHYS_TYPE[plan["phys"][j]])
            chunk_meta.buf += field_header(2, chunk_meta.last_id, T_LIST)
            chunk_meta.buf += list_header(len(plan["encodings"][j]), T_I32)
            chunk_meta.last_id = 2
            for enc in plan["encodings"][j]:
                chunk_meta.buf += write_zigzag(enc)
            chunk_meta.list_of_binary(3, plan["paths"][j])
            chunk_meta.i32(4, CODEC[plan["codec"]])
            chunk_meta.i64(5, chunk["num_values"])
            chunk_meta.i64(6, chunk["uncompressed"])
            chunk_meta.i64(7, chunk["end"] - chunk["start"])
            chunk_meta.i64(9, chunk["data_off"])
            if chunk["dict_off"] is not None:
                chunk_meta.i64(11, chunk["dict_off"])

            col = StructWriter()
            col.i64(2, chunk["dict_off"] if chunk["dict_off"] is not None else chunk["data_off"])
            col.raw(3, T_STRUCT, chunk_meta.close())
            rg.buf += col.close()

        rg.i64(2, sum(chunk["uncompressed"] for chunk in group))
        rg.i64(3, rows)
        rg_bufs.append(rg.close())

    meta.i64(3, num_rows)
    meta.buf += field_header(4, meta.last_id, T_LIST) + list_header(len(rg_bufs), T_STRUCT)
    meta.last_id = 4
    for rg_bytes in rg_bufs:
        meta.buf += rg_bytes
    # key_value_metadata (5), created_by (6), column_orders (7): reuse from reference
    for field_id in (5, 6, 7):
        if field_id in ref_fields:
            ftype, start, end = ref_fields[field_id]
            meta.raw(field_id, ftype, reference_footer[start:end])

    return meta.close(), num_rows, rows_per_group, dropped


def repair(broken_path, ref_path, out_path, rows_per_group=None):
    ref_fields, reference_footer = ref_footer(ref_path)
    plan = plan_from_ref(ref_path, rows_per_group)

    with open(broken_path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != b"PAR1":
        raise SystemExit(f"{broken_path}: leading magic bytes are not PAR1")
    if raw[-4:] == b"PAR1":
        print("warning: file already ends with PAR1 (footer may be present); repairing anyway")

    groups, partial = scan_chunks(raw, plan["expect"])
    footer, num_rows, rows_per_group_counts, dropped = build_footer(
        raw, groups, plan, ref_fields, reference_footer
    )
    # Drop the bytes belonging to row groups that were discarded.
    keep_end = 4
    for group in groups:
        if len(group) == len(plan["paths"]):
            keep_end = group[-1]["end"]
    payload = raw[:keep_end]

    out = payload + footer + struct.pack("<I", len(footer)) + b"PAR1"
    with open(out_path, "wb") as fh:
        fh.write(out)

    print(f"pages scanned      : {len(groups)} row group(s){' (last one partial)' if partial else ''}")
    print(f"row groups written : {len(rows_per_group_counts)}"
          f" ({', '.join(str(r) for r in rows_per_group_counts[:6])}"
          f"{', ...' if len(rows_per_group_counts) > 6 else ''} rows each)")
    if dropped:
        print(f"dropped partial row group(s): {dropped}")
    print(f"rows recovered     : {num_rows}")
    print(f"footer size        : {len(footer)} bytes")
    print(f"written            : {out_path} ({len(out)} bytes)")
    return out_path, num_rows


def verify(path, expect_rows=None):
    table = pq.read_table(path)
    print(f"verify             : readable, {table.num_rows} rows, {table.num_columns} columns")
    if expect_rows is not None and table.num_rows != expect_rows:
        print(f"verify             : WARNING expected {expect_rows} rows")
        return False
    for name, column in zip(table.column_names, table.columns):
        print(f"  {name:24s} {str(column.type):18s} nulls={column.null_count}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--broken", required=True, help="Parquet file missing its footer")
    parser.add_argument("--ref", required=True, help="intact Parquet file with the same schema")
    parser.add_argument("--out", required=True, help="path for the repaired file")
    parser.add_argument("--rows-per-group", type=int, default=None,
                        help="cells per column per row group as written by the producer "
                             "(e.g. 898 frames). Defaults to the reference's row group size.")
    parser.add_argument("--no-verify", action="store_true", help="skip read-back verification")
    args = parser.parse_args(argv)

    _, rows = repair(args.broken, args.ref, args.out, args.rows_per_group)
    if args.no_verify:
        return 0
    ok = True
    try:
        ok = verify(args.out, rows)
    except Exception as exc:  # pragma: no cover - surfaced to the CLI user
        print(f"verify failed: {exc}")
        return 2
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
