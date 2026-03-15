"""Inspect .mpmodel files — print graph, shapes, memory requirements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from mempipe_convert.mpmodel import MpModel, OpType, Shape, load_model


@dataclass
class ModelInfo:
    """Human-readable summary of a .mpmodel file."""

    name: str
    input_shapes: List[Shape]
    output_shapes: List[Shape]
    num_ops: int
    op_types: List[str]
    num_tensors: int
    tensor_names: List[str]
    tensor_shapes: Dict[str, Shape]
    weights_bytes: int
    total_params: int
    quant_method: str
    platform_hints: str
    estimated_flops: int
    estimated_activation_bytes: int

    def __str__(self) -> str:
        lines = [
            f"Model: {self.name}",
            f"  Inputs:  {[str(s.dims) for s in self.input_shapes]}",
            f"  Outputs: {[str(s.dims) for s in self.output_shapes]}",
            f"  Ops:     {self.num_ops} ({', '.join(self.op_types)})",
            f"  Tensors: {self.num_tensors}",
            f"  Params:  {self.total_params:,} ({self.weights_bytes:,} bytes)",
            f"  Est. FLOPs:       {self.estimated_flops:,}",
            f"  Est. Activations: {self.estimated_activation_bytes:,} bytes",
        ]
        if self.quant_method:
            lines.append(f"  Quantization: {self.quant_method}")
        if self.platform_hints:
            lines.append(f"  Platform: {self.platform_hints}")

        if self.tensor_shapes:
            lines.append("  Tensor shapes:")
            for name, s in self.tensor_shapes.items():
                lines.append(f"    {name}: {s.dims}")

        return "\n".join(lines)


def inspect_model(path_or_bytes) -> ModelInfo:
    """Inspect a .mpmodel file and return a ModelInfo summary.

    Args:
        path_or_bytes: A file path (str/Path) or raw bytes.

    Returns:
        ModelInfo with graph, shape, and memory information.
    """
    if isinstance(path_or_bytes, (str, Path)):
        data = Path(path_or_bytes).read_bytes()
    else:
        data = path_or_bytes

    model = load_model(data)
    return _analyze(model)


def _analyze(model: MpModel) -> ModelInfo:
    """Analyze a loaded model and compute derived stats."""
    op_types = []
    seen = set()
    for node in model.graph:
        name = node.op_type.name
        if name not in seen:
            op_types.append(name)
            seen.add(name)

    # Count total parameters (float32 → 4 bytes each)
    total_params = len(model.weights_blob) // 4

    # Estimate FLOPs and activation memory
    flops = 0
    act_bytes = 0
    for node in model.graph:
        f, a = _estimate_op(node, model.tensor_shapes)
        flops += f
        act_bytes += a

    return ModelInfo(
        name=model.metadata.name,
        input_shapes=model.metadata.input_shapes,
        output_shapes=model.metadata.output_shapes,
        num_ops=len(model.graph),
        op_types=op_types,
        num_tensors=len(model.tensor_names),
        tensor_names=list(model.tensor_names),
        tensor_shapes=dict(model.tensor_shapes),
        weights_bytes=len(model.weights_blob),
        total_params=total_params,
        quant_method=model.metadata.quant_method,
        platform_hints=model.metadata.platform_hints,
        estimated_flops=flops,
        estimated_activation_bytes=act_bytes,
    )


def _estimate_op(
    node: OpNode,
    tensor_shapes: Dict[str, Shape],
) -> tuple:
    """Rough FLOPs + activation byte estimate for one op node.

    Returns (flops, activation_bytes).
    """
    # For a proper estimate we'd need the full shape map. This is best-effort.
    op = node.op_type

    if op in (OpType.MatMul, OpType.Dense):
        # If we knew M,K,N: 2*M*K*N
        # Rough: use output shape if available
        return _rough_matmul_flops(node, tensor_shapes)
    elif op in (OpType.ReLU, OpType.Sigmoid, OpType.Softmax):
        # ~ N ops
        return 0, 0  # activation already counted by matmul
    elif op == OpType.Conv2D:
        return _rough_conv_flops(node, tensor_shapes)
    return 0, 0


def _rough_matmul_flops(node: OpNode, shapes: Dict[str, Shape]) -> tuple:
    return 0, 0  # TODO: wire up proper shape resolution


def _rough_conv_flops(node: OpNode, shapes: Dict[str, Shape]) -> tuple:
    return 0, 0  # TODO: wire up proper shape resolution
