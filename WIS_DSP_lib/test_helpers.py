from __future__ import annotations

import contextlib
import importlib.util
import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "assignments"



def _discover_module_names(directory: Path) -> tuple[str, ...]:
    return tuple(
        sorted(path.stem for path in directory.glob("*.py") if path.name != "__init__.py")
    )

@dataclass(frozen=True)
class RuntimePair:
    standalone_module: ModuleType
    standalone_stdout: str
    olmo_module: ModuleType
    olmo_stdout: str


def resolve_script_path(script_name: str | Path) -> Path:
    script_path = Path(script_name)
    if not script_path.is_absolute():
        script_path = SCRIPT_DIR / script_path
    if not script_path.exists():
        raise FileNotFoundError(f"Course script not found: {script_path}")
    return script_path


def build_runtime_name(case_name: str, variant: str, script_name: str | Path) -> str:
    relative_path = resolve_script_path(script_name).relative_to(SCRIPT_DIR)
    fragment = "_".join(part.replace("-", "_") for part in relative_path.with_suffix("").parts)
    return f"course_refactor_{case_name}_{variant}_{fragment}"


def load_module(
    module_name: str,
    script_name: str | Path,
) -> tuple[ModuleType, str]:
    script_path = resolve_script_path(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")

    existing_sys_path = sys.path.copy()
    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = existing_sys_path

    return module, stdout.getvalue().strip()


def load_runtime_pair(
    case_name: str,
    *,
    standalone_script: str | Path,
    olmo_script: str | Path,
) -> RuntimePair:
    standalone_module, standalone_stdout = load_module(
        build_runtime_name(case_name, "standalone", standalone_script),
        standalone_script,
    )
    olmo_module, olmo_stdout = load_module(
        build_runtime_name(case_name, "olmo", olmo_script),
        olmo_script,
    )
    return RuntimePair(
        standalone_module=standalone_module,
        standalone_stdout=standalone_stdout,
        olmo_module=olmo_module,
        olmo_stdout=olmo_stdout,
    )


def run_batch_case(
    case_name: str,
    *,
    standalone_script: str | Path,
    olmo_script: str | Path,
    standalone_batch_extractor: Any = None,
    assert_stdout: bool = False,
) -> RuntimePair:
    runtime = load_runtime_pair(
        case_name,
        standalone_script=standalone_script,
        olmo_script=olmo_script,
    )
    batch_extractor = standalone_batch_extractor or get_standalone_masked_batch_output
    assert_values_equal(
        get_olmo_batch_output(runtime.olmo_module),
        batch_extractor(runtime.standalone_module),
        "batch",
    )
    if assert_stdout:
        assert_stdout_equal(runtime.olmo_stdout, runtime.standalone_stdout)
    return runtime


def capture_main_output(module: ModuleType) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        module.main()
    return stdout.getvalue().strip()


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_sample_field(sample: Any, field_name: str) -> Any:
    if isinstance(sample, dict):
        return sample[field_name]
    return getattr(sample, field_name)


def _max_abs_diff(expected: torch.Tensor, actual: torch.Tensor) -> float:
    if expected.numel() == 0:
        return 0.0
    if expected.is_floating_point() or actual.is_floating_point():
        diff = expected.to(torch.float32) - actual.to(torch.float32)
    else:
        diff = expected.to(torch.int64) - actual.to(torch.int64)
    return float(diff.abs().max().item())


def assert_tensor_values_equal(expected: torch.Tensor, actual: torch.Tensor, label: str) -> None:
    assert isinstance(actual, torch.Tensor), f"Type mismatch for {label}"
    expected_cpu = expected.detach().cpu()
    actual_cpu = actual.detach().cpu()
    assert expected_cpu.shape == actual_cpu.shape, f"Shape mismatch for {label} (expected {tuple(expected_cpu.shape)}, got {tuple(actual_cpu.shape)})"
    assert expected_cpu.dtype == actual_cpu.dtype, f"Dtype mismatch for {label} (expected {expected_cpu.dtype}, got {actual_cpu.dtype})"
    assert torch.equal(expected_cpu, actual_cpu), (
        f"Tensor values differ for {label}: "
        f"shape={tuple(expected_cpu.shape)}, dtype={expected_cpu.dtype}, "
        f"max_abs_diff={_max_abs_diff(expected_cpu, actual_cpu)}"
        # print number of mismatches elements
        f", num_mismatches={(expected_cpu != actual_cpu).sum().item()}"

    )


def assert_tensor_compatible(expected: torch.Tensor, actual: torch.Tensor, label: str) -> None:
    assert isinstance(actual, torch.Tensor), f"Type mismatch for {label}"
    expected_cpu = expected.detach().cpu()
    actual_cpu = actual.detach().cpu()
    assert expected_cpu.shape == actual_cpu.shape, f"Shape mismatch for {label}"
    assert expected_cpu.dtype == actual_cpu.dtype, f"Dtype mismatch for {label}"
    if expected_cpu.is_floating_point():
        assert torch.isfinite(expected_cpu).all(), f"Non-finite values in expected {label}"
    if actual_cpu.is_floating_point():
        assert torch.isfinite(actual_cpu).all(), f"Non-finite values in actual {label}"


def assert_values_equal(expected: object, actual: object, label: str) -> None:
    if isinstance(expected, torch.Tensor):
        assert_tensor_values_equal(expected, actual, label)
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"Type mismatch for {label}"
        assert set(expected) == set(actual), f"Key mismatch for {label}"
        for key in sorted(expected):
            assert_values_equal(expected[key], actual[key], f"{label}.{key}")
        return

    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)), f"Type mismatch for {label}"
        assert len(expected) == len(actual), f"Length mismatch for {label}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            assert_values_equal(expected_item, actual_item, f"{label}[{index}]")
        return

    assert expected == actual, f"Value mismatch for {label}: {expected!r} != {actual!r}"


def assert_stdout_equal(expected: str, actual: str) -> None:
    assert expected == actual, (
        "Printed output differs between the OLMo-based and refactored run scripts: "
        f"OLMo={expected!r}, refactored={actual!r}"
    )


def assert_masked_view_equal(expected: Any, actual: Any, view_name: str) -> None:
    assert_tensor_values_equal(
        get_sample_field(expected, "timestamps"),
        get_sample_field(actual, "timestamps"),
        f"{view_name}.timestamps",
    )
    assert_tensor_values_equal(
        get_sample_field(expected, "sentinel2_l2a"),
        get_sample_field(actual, "sentinel2_l2a"),
        f"{view_name}.sentinel2_l2a",
    )
    assert_tensor_values_equal(
        get_sample_field(expected, "sentinel2_l2a_mask"),
        get_sample_field(actual, "sentinel2_l2a_mask"),
        f"{view_name}.sentinel2_l2a_mask",
    )


def get_olmo_batch_output(module: ModuleType) -> dict[str, torch.Tensor]:
    batch = module.batch[1]
    return {
        "timestamps": batch.timestamps,
        module.modality: getattr(batch, module.modality),
        module.mask_name: getattr(batch, module.mask_name),
    }


def get_standalone_masked_batch_output(module: ModuleType) -> dict[str, torch.Tensor]:
    return module.masked_batch


def get_standalone_tuple_batch_output(module: ModuleType) -> dict[str, torch.Tensor]:
    return module.batch[0]


def assert_parameter_gradients_equal(
    olmo_parameters: Iterable[torch.nn.Parameter],
    standalone_parameters: Iterable[torch.nn.Parameter],
    label_prefix: str,
) -> None:
    for index, (olmo_param, standalone_param) in enumerate(zip(olmo_parameters, standalone_parameters)):
        if standalone_param.requires_grad and olmo_param.requires_grad:
            assert_values_equal(
                olmo_param.grad,
                standalone_param.grad,
                f"{label_prefix}[{index}] {tuple(olmo_param.shape)}",
            )