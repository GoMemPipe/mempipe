"""End-to-end connectivity: ONNX transformer-style graphs must reach Go inference."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mempipe_convert.transformer import from_onnx_transformer


def _repo_root() -> Path:
    # tests/ -> mempipe-convert -> tools -> repo root (mempipe)
    return Path(__file__).resolve().parents[3]


def _make_unsqueeze_then_matmul_onnx(rng: np.random.Generator) -> onnx.ModelProto:
    """Graph: X [4] -> Unsqueeze(axis 0) -> [1,4] -> MatMul W [4,1] -> Y [1,1].

    Unsqueeze was previously skipped by the converter with no op, leaving zeros and
    disconnecting the network from the graph input.
    """
    x_vi = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y_vi = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])

    w_arr = rng.standard_normal((4, 1)).astype(np.float32)
    w_init = numpy_helper.from_array(w_arr, name="W")
    axes_arr = np.array([0], dtype=np.int64)
    axes_init = numpy_helper.from_array(axes_arr, name="axes_unsq")

    nodes = [
        helper.make_node("Unsqueeze", ["X", "axes_unsq"], ["X2"]),
        helper.make_node("MatMul", ["X2", "W"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "unsqueeze_matmul",
        inputs=[x_vi],
        outputs=[y_vi],
        initializer=[w_init, axes_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="mempipe-connectivity-test",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize("seed", [42])
def test_unsqueeze_matmul_two_inputs_differ_in_go(tmp_path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    onnx_path = tmp_path / "tiny_transformer.onnx"
    mpmodel_path = tmp_path / "tiny_transformer.mpmodel"

    model = _make_unsqueeze_then_matmul_onnx(rng)
    onnx.save(model, onnx_path)

    from_onnx_transformer(str(onnx_path), str(mpmodel_path), seq_len=8)

    repo = _repo_root()
    env = {**os.environ, "MPMODEL_CONNECTIVITY_PATH": str(mpmodel_path)}
    cmd = [
        "go",
        "test",
        "-run",
        "TestMpmodelTransformerConnectivityFromEnv",
        "-count=1",
        "./inference",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            "Go connectivity check failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def test_converted_graph_contains_reshape_not_only_skip(tmp_path) -> None:
    """Sanity: the fixture model must include a Reshape (from Unsqueeze lowering)."""
    from mempipe_convert.mpmodel import OpType, load_model

    rng = np.random.default_rng(0)
    onnx_path = tmp_path / "m.onnx"
    out_path = tmp_path / "m.mpmodel"
    onnx.save(_make_unsqueeze_then_matmul_onnx(rng), onnx_path)
    from_onnx_transformer(str(onnx_path), str(out_path), seq_len=8)

    m = load_model(out_path.read_bytes())
    types = [n.op_type for n in m.graph]
    assert OpType.Reshape in types
    assert OpType.MatMul in types
