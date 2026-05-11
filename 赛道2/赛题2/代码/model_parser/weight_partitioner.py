"""
Weight Partitioner - Splits weights to fit Sub-Cube dimensions
Supports horizontal and vertical splitting, and replication
"""

from typing import List, Optional
from compiler_mapping.weight_cube import WeightSection, Operator


class WeightPartitioner:
    """Partitions operator weights into Weight-Cube sections"""

    def __init__(self, subcube_hw: int, depth: int):
        self.subcube_hw = subcube_hw
        self.depth = depth

    def partition(self, operators: List[Operator]) -> List[WeightSection]:
        """Partition all operator weights into sections that fit in Sub-Cube"""
        sections = []

        for op in operators:
            rows, cols = op.weight_shape[0], op.weight_shape[1]

            # Calculate depth needed: depth = ceil(params / (hw * hw))
            params = rows * cols
            max_2d_capacity = self.subcube_hw * self.subcube_hw

            # If weight is larger than one Sub-Cube layer, split horizontally/vertically
            row_splits = max(1, (rows + self.subcube_hw - 1) // self.subcube_hw)
            col_splits = max(1, (cols + self.subcube_hw - 1) // self.subcube_hw)

            total_sections = row_splits * col_splits
            section_idx = 0

            for rs in range(row_splits):
                r_start = rs * self.subcube_hw
                r_end = min(r_start + self.subcube_hw, rows)

                for cs in range(col_splits):
                    c_start = cs * self.subcube_hw
                    c_end = min(c_start + self.subcube_hw, cols)

                    split_rows = r_end - r_start
                    split_cols = c_end - c_start

                    # Depth needed for this section (capped at config depth)
                    depth_needed = max(1, (split_rows * split_cols + max_2d_capacity - 1) // max_2d_capacity)
                    depth_needed = min(depth_needed, self.depth)

                    section = WeightSection(
                        name=f"{op.name}_sec_{section_idx}",
                        shape_2d=[split_rows, split_cols],
                        depth=depth_needed,
                        parent_operator=op.name,
                        section_idx=section_idx,
                        total_sections=total_sections,
                    )
                    sections.append(section)
                    section_idx += 1

        return sections

    def partition_with_replication(
        self, operators: List[Operator], replication_map: Optional[dict] = None
    ) -> List[WeightSection]:
        """Partition with optional weight replication for parallelism"""
        sections = self.partition(operators)

        if replication_map is None:
            return sections

        replicated = []
        for section in sections:
            copies = replication_map.get(section.parent_operator, 1)
            for i in range(copies):
                r = WeightSection(
                    name=f"{section.name}_rep{i}",
                    shape_2d=section.shape_2d,
                    depth=section.depth,
                    parent_operator=section.parent_operator,
                    section_idx=section.section_idx,
                    total_sections=section.total_sections,
                    is_replicated=(copies > 1),
                    replication_idx=i,
                )
                replicated.append(r)

        return replicated
