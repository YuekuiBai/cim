"""
Weight Mapper - Maps linear weights to 2D CIM array
CIM array: row=1024bit, col=4096bit
Weight [Cin, Cout] -> mapped along row(Cin) and col(Cout*bitwidth)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

CIM_ROW_BITS = 1024
CIM_COL_BITS = 4096


@dataclass
class WeightMapping:
    weight_name: str
    cin: int
    cout: int
    dtype: str  # i8/i16/i32
    bit_width: int
    # Mapped region in CIM array
    row_start: int = 0
    row_end: int = 0
    col_start: int = 0
    col_end: int = 0
    # Split info
    num_row_splits: int = 1
    num_col_splits: int = 1
    copies: int = 1

    def to_dict(self):
        return {
            "weight": self.weight_name,
            "shape": [self.cin, self.cout],
            "dtype": self.dtype,
            "bit_width": self.bit_width,
            "mapping": [self.row_start, self.col_start, self.row_end, self.col_end],
            "row_splits": self.num_row_splits,
            "col_splits": self.num_col_splits,
        }


class WeightMapper:
    def __init__(self, row_bits=CIM_ROW_BITS, col_bits=CIM_COL_BITS):
        self.row_bits = row_bits
        self.col_bits = col_bits
        self.mappings: List[WeightMapping] = []

    def map_weights(self, graph) -> List[WeightMapping]:
        """Map all linear layer weights to CIM array"""
        self.mappings = []
        for node in graph.nodes:
            if node.node_type.value == "linear":
                weight_name = node.weight_name
                cin = node.cin
                cout = node.cout

                # Determine dtype from weight array
                if weight_name in graph.weights:
                    w = graph.weights[weight_name]
                    dtype, bit_width = self._infer_dtype(w)
                else:
                    dtype = "i8"
                    bit_width = 8

                mapping = self._map_single_weight(
                    weight_name, cin, cout, dtype, bit_width
                )
                self.mappings.append(mapping)
        return self.mappings

    def _infer_dtype(self, weight_array):
        w = weight_array
        if np.issubdtype(w.dtype, np.int8):
            return "i8", 8
        elif np.issubdtype(w.dtype, np.int16):
            return "i16", 16
        else:
            return "i8", 8  # Default to int8

    def _map_single_weight(self, name, cin, cout, dtype, bit_width):
        col_needed = cout * bit_width
        row_needed = cin

        num_row_splits = max(1, (row_needed + self.row_bits - 1) // self.row_bits)
        num_col_splits = max(1, (col_needed + self.col_bits - 1) // self.col_bits)

        row_end = min(row_needed, self.row_bits)
        col_end = min(col_needed, self.col_bits)

        mapping = WeightMapping(
            weight_name=name,
            cin=cin,
            cout=cout,
            dtype=dtype,
            bit_width=bit_width,
            row_start=0,
            row_end=row_end,
            col_start=0,
            col_end=col_end,
            num_row_splits=num_row_splits,
            num_col_splits=num_col_splits,
        )
        return mapping

    def report(self) -> str:
        lines = ["=== CIM Array Weight Mapping ==="]
        lines.append(f"  Array: {self.row_bits} rows x {self.col_bits} cols")
        for m in self.mappings:
            lines.append(
                f"  {m.weight_name}: [{m.cin}x{m.cout}] {m.dtype} -> "
                f"row[{m.row_start}:{m.row_end}] col[{m.col_start}:{m.col_end}] "
                f"({m.num_row_splits}R x {m.num_col_splits}C splits)"
            )
        return "\n".join(lines)
