from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "submodules/act/pact_blur.py"
FRONTEND_PATH = ROOT / "submodules/act/eval_pact_frontend_screen_row.py"
COLLISION_PATH = ROOT / "submodules/act/eval_pact_collision_row.py"

EXPECTED_FUNCTION = '''def blur_images(image_data, sigma):
    """FACTR visual curriculum: Gaussian-blur a (B, num_cam, C, H, W) 0-1 image batch.

    Training batches only — validation and eval always see sharp frames. Blurring
    the 0-1 tensor is equivalent to blurring after the ImageNet normalization inside
    the policy (the blur commutes with a per-channel affine).
    """
    if sigma < 0.1:
        return image_data
    from torchvision.transforms.functional import gaussian_blur
    b, k, c, h, w = image_data.shape
    kernel = 2 * math.ceil(3 * sigma) + 1
    flat = image_data.reshape(b * k, c, h, w)
    return gaussian_blur(flat, kernel_size=kernel, sigma=sigma).reshape(b, k, c, h, w)
'''


def load_module():
    spec = importlib.util.spec_from_file_location("pact_blur", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def test_blur_primitive_is_exact_frozen_upstream_function() -> None:
    assert function_source(MODULE_PATH, "blur_images") == EXPECTED_FUNCTION


def test_sigma_zero_is_same_object_and_bit_identical() -> None:
    module = load_module()
    image = torch.rand(2, 1, 3, 240, 320)
    blurred = module.blur_images(image, 0.0)
    assert blurred is image
    assert torch.equal(blurred, image)


def test_positive_sigma_changes_rgb_without_shape_or_range_change() -> None:
    module = load_module()
    image = torch.zeros(1, 1, 3, 240, 320)
    image[:, :, :, ::2, ::2] = 1.0
    blurred = module.blur_images(image, 0.5)
    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype
    assert not torch.equal(blurred, image)
    assert float(blurred.min()) >= 0.0
    assert float(blurred.max()) <= 1.0


def test_required_frontend_and_defence_in_depth_sites_both_call_blur() -> None:
    frontend = FRONTEND_PATH.read_text()
    collision = COLLISION_PATH.read_text()
    call = "pact_blur.blur_images("
    assert frontend.count(call) == 1
    assert collision.count(call) == 1
    assert frontend.index(call) < frontend.index("raw = self._raw_proximity(observation)")
    assert collision.index(call) < collision.index("raw = self._raw_proximity(observation)")


def test_collision_driver_records_and_threads_sigma() -> None:
    source = COLLISION_PATH.read_text()
    assert 'parser.add_argument("--blur-sigma", type=float, default=0.0)' in source
    assert '"blur_sigma": float(args.blur_sigma)' in source
    assert "blur_sigma=float(args.blur_sigma)" in source


def test_runtime_records_visual_proximity_and_action_diagnostics() -> None:
    collision = COLLISION_PATH.read_text()
    frontend = FRONTEND_PATH.read_text()
    assert "self._record_blur_diagnostic(sharp_image_tensor, image_tensor)" in frontend
    assert '"first_visual_input_changed": sharp_sha != input_sha' in collision
    assert '"first_raw_proximity_sha256"' in collision
    assert '"model_output_trace_sha256"' in collision
    assert collision.index("self._record_blur_diagnostic") < collision.index(
        "raw = self._raw_proximity(observation)"
    )
