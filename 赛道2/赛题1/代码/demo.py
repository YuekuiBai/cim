"""
CIM Compiler Interactive Demo Script
Provides interactive visualization and demonstration of the compilation pipeline
Supports: real-time IR visualization, SRAM layout animation, instruction tracing
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx_parser.onnx_loader import ONNXParser
from optimizer.pipeline import OptimizationPipeline
from resource_manager.sram_allocator import SRAMAllocator
from resource_manager.weight_mapper import WeightMapper
from instruction_gen.lowering import IRLowering

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch


class CompilationDemo:
    """Interactive demonstration of CIM compilation pipeline"""
    
    def __init__(self, model_path, output_dir="demo_output"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams.update({
            'font.sans-serif': ['Droid Sans Fallback', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
        })
    
    def run_demo(self):
        """Run complete demonstration"""
        print("=" * 60)
        print("CIM Compiler Interactive Demo")
        print("=" * 60)
        
        self._step1_parse()
        self._step2_optimize()
        self._step3_allocate()
        self._step4_map_weights()
        self._step5_lower()
        self._generate_summary()
        
        print("\nDemo completed! Check output directory for visualizations.")
    
    def _step1_parse(self):
        print("\n[Demo Step 1] Parsing ONNX model...")
        parser = ONNXParser(self.model_path)
        self.ir = parser.parse()
        
        self._visualize_ir_graph(self.ir, "original")
        self._print_ir_summary(self.ir, "Original IR")
    
    def _step2_optimize(self):
        print("\n[Demo Step 2] Running optimization pipeline...")
        pipeline = OptimizationPipeline()
        self.ir_optimized, opt_stats = pipeline.optimize(self.ir)
        
        self._visualize_ir_graph(self.ir_optimized, "optimized")
        self._print_ir_summary(self.ir_optimized, "Optimized IR")
        self._visualize_optimization_stats(opt_stats)
    
    def _step3_allocate(self):
        print("\n[Demo Step 3] SRAM allocation with Interval Graph Coloring...")
        self.allocator = SRAMAllocator(use_interval_coloring=True)
        self.layout = self.allocator.allocate(self.ir_optimized)
        
        self._visualize_sram_layout(self.layout)
        self._visualize_tensor_intervals(self.ir_optimized)
    
    def _step4_map_weights(self):
        print("\n[Demo Step 4] Mapping weights to CIM array...")
        self.w_mapper = WeightMapper()
        self.mappings = self.w_mapper.map_weights(self.ir_optimized)
        
        self._visualize_weight_mapping(self.mappings)
    
    def _step5_lower(self):
        print("\n[Demo Step 5] Lowering to ISA...")
        lowering = IRLowering(self.allocator, self.mappings)
        self.instructions = lowering.lower(self.ir_optimized)
        
        self._visualize_instruction_flow(self.instructions)
    
    def _visualize_ir_graph(self, ir, suffix):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        node_colors = {
            'linear': '#2196F3',
            'elementwise': '#4CAF50',
            'input': '#FF9800',
            'output': '#F44336',
            'constant': '#9C27B0',
            'placeholder': '#00BCD4',
        }
        
        y_step = 1.5
        x_positions = {}
        y_positions = {}
        
        for i, node in enumerate(ir.nodes):
            x = 0
            y = (len(ir.nodes) - i) * y_step
            x_positions[node.node_id] = x
            y_positions[node.node_id] = y
            
            node_type = node.node_type.value
            color = node_colors.get(node_type, '#9E9E9E')
            
            label = f"{node.node_id}\n({node_type})"
            ax.text(x, y, label, ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8, edgecolor='black'))
        
        for node in ir.nodes:
            for inp in node.inputs:
                if inp in x_positions:
                    src_y = y_positions.get(inp, 0)
                    dst_y = y_positions[node.node_id]
                    ax.annotate('', xy=(0, dst_y + 0.3), xytext=(0, src_y - 0.3),
                               arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, len(ir.nodes) * y_step + 1)
        ax.set_title(f'IR Computation Graph ({suffix})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        legend_elements = [mpatches.Patch(facecolor=c, label=t) for t, c in node_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.savefig(os.path.join(self.output_dir, f'ir_graph_{suffix}.png'), dpi=150)
        plt.close()
        print(f"  Saved: ir_graph_{suffix}.png")
    
    def _print_ir_summary(self, ir, title):
        print(f"\n  {title} Summary:")
        print(f"    Nodes: {len(ir.nodes)}")
        print(f"    Tensors: {len(ir.tensors)}")
        print(f"    Inputs: {ir.input_names}")
        print(f"    Outputs: {ir.output_names}")
        
        node_types = {}
        for node in ir.nodes:
            t = node.node_type.value
            node_types[t] = node_types.get(t, 0) + 1
        print(f"    Node types: {node_types}")
    
    def _visualize_optimization_stats(self, stats):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        passes = list(stats['passes'].keys())
        removed = [stats['passes'][p]['removed'] for p in passes]
        
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        bars = ax.bar(passes, removed, color=colors[:len(passes)], edgecolor='white', linewidth=0.5)
        
        for bar, val in zip(bars, removed):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Nodes Removed', fontsize=11)
        ax.set_title('Optimization Pass Effectiveness', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'optimization_stats.png'), dpi=150)
        plt.close()
        print("  Saved: optimization_stats.png")
    
    def _visualize_sram_layout(self, layout):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        region_colors = {
            'input': '#2196F3',
            'output': '#4CAF50',
            'acc': '#FF9800',
            'tmp': '#F44336',
            'bias': '#9C27B0',
            'tensor': '#00BCD4',
        }
        
        y = 0
        bar_height = 0.8
        for region in layout.regions:
            color = '#9E9E9E'
            for key, c in region_colors.items():
                if key in region.name.lower():
                    color = c
                    break
            
            rect = mpatches.FancyBboxPatch(
                (region.start / 524288 * 100, y),
                (region.size / 524288 * 100), bar_height,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=0.8, edgecolor='white', linewidth=0.5
            )
            ax.add_patch(rect)
            
            if region.size > 100:
                ax.text(region.start / 524288 * 100 + region.size / 524288 * 50, y + bar_height/2,
                       f"{region.tensor_name}\n({region.size}B)",
                       ha='center', va='center', fontsize=7, fontweight='bold')
            y += bar_height + 0.2
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, y)
        ax.set_xlabel('SRAM Address (% of 512KB)', fontsize=11)
        ax.set_title('SRAM Memory Layout Visualization', fontsize=13, fontweight='bold')
        ax.set_yticks([])
        
        legend_elements = [mpatches.Patch(facecolor=c, label=t) for t, c in region_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.savefig(os.path.join(self.output_dir, 'sram_layout.png'), dpi=150)
        plt.close()
        print("  Saved: sram_layout.png")
    
    def _visualize_tensor_intervals(self, ir):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        tensor_first_use = {}
        tensor_last_use = {}
        
        for i, node in enumerate(ir.nodes):
            for inp in node.inputs:
                if inp not in tensor_first_use:
                    tensor_first_use[inp] = i
                tensor_last_use[inp] = i
            for out in node.outputs:
                tensor_first_use[out] = i
                tensor_last_use[out] = i
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(tensor_first_use)))
        
        for idx, (tensor_name, start) in enumerate(sorted(tensor_first_use.items())):
            end = tensor_last_use.get(tensor_name, start + 1)
            color = colors[idx % len(colors)]
            
            ax.barh(tensor_name, end - start, left=start, height=0.8,
                   color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Program Point (Node Index)', fontsize=11)
        ax.set_title('Tensor Live Intervals', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'tensor_intervals.png'), dpi=150)
        plt.close()
        print("  Saved: tensor_intervals.png")
    
    def _visualize_weight_mapping(self, mappings):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cim_array = np.zeros((1024, 4096), dtype=np.float32)
        
        for idx, m in enumerate(mappings):
            row_start, col_start, row_end, col_end = m.row_start, m.col_start, m.row_end, m.col_end
            color_value = (idx + 1) / (len(mappings) + 1)
            cim_array[row_start:row_end, col_start:col_end] = color_value
        
        im = ax.imshow(cim_array, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Column (4096 bits)', fontsize=11)
        ax.set_ylabel('Row (1024 bits)', fontsize=11)
        ax.set_title('CIM Array Weight Mapping', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Weight Layer')
        
        plt.savefig(os.path.join(self.output_dir, 'weight_mapping.png'), dpi=150)
        plt.close()
        print("  Saved: weight_mapping.png")
    
    def _visualize_instruction_flow(self, instructions):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        opcode_counts = {}
        for instr in instructions:
            if instr.opcode != "//":
                opcode_counts[instr.opcode] = opcode_counts.get(instr.opcode, 0) + 1
        
        sorted_opcodes = sorted(opcode_counts.items(), key=lambda x: -x[1])
        opcodes, counts = zip(*sorted_opcodes[:10])
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(opcodes)))
        bars = ax.barh(opcodes, counts, color=colors, edgecolor='white', linewidth=0.5)
        
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(val), ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Instruction Count', fontsize=11)
        ax.set_title('Instruction Distribution', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'instruction_flow.png'), dpi=150)
        plt.close()
        print("  Saved: instruction_flow.png")
    
    def _generate_summary(self):
        summary = {
            "model": self.model_path,
            "ir_nodes_original": len(self.ir.nodes),
            "ir_nodes_optimized": len(self.ir_optimized.nodes),
            "sram_usage_bytes": self.layout.total_used,
            "sram_reuse_count": self.layout.reuse_count,
            "total_instructions": len([i for i in self.instructions if i.opcode != "//"]),
            "weight_mappings": len(self.mappings),
        }
        
        summary_path = os.path.join(self.output_dir, 'demo_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  Demo Summary:")
        for k, v in summary.items():
            print(f"    {k}: {v}")
        print(f"\n  Saved: demo_summary.json")


def main():
    parser = argparse.ArgumentParser(description="CIM Compiler Demo")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default="demo_output", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    demo = CompilationDemo(args.model, args.output)
    demo.run_demo()


if __name__ == "__main__":
    import argparse
    main()
