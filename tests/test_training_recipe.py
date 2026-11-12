"""Focused tests for the §1 training recipe."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biosentinel.training_recipe import (
    build_resnet18,
    compute_class_weights,
    group_train_test_split,
    make_eval_transforms,
    make_train_transforms,
    per_class_recall_from_confusion,
    set_stage1_head_only,
    set_stage2_late_unfreeze,
    trainable_param_names,
)


def test_stage1_trains_head_only():
    model = build_resnet18(num_classes=4, pretrained=False)
    set_stage1_head_only(model)
    names = trainable_param_names(model)
    assert names, "stage 1 must leave something trainable"
    assert all(n.startswith("fc.") for n in names)


def test_stage2_unfreezes_layer4_and_head():
    model = build_resnet18(num_classes=4, pretrained=False)
    set_stage2_late_unfreeze(model)
    names = trainable_param_names(model)
    assert names
    assert all(n.startswith("layer4.") or n.startswith("fc.") for n in names)
    assert any(n.startswith("layer4.") for n in names)
    assert any(n.startswith("fc.") for n in names)


def test_stages_cover_all_edge_arches():
    from biosentinel.models import EDGE_ARCHES, build_classifier, head_prefixes

    for arch in EDGE_ARCHES:
        model = build_classifier(arch, num_classes=4, pretrained=False)
        set_stage1_head_only(model)
        stage1 = trainable_param_names(model)
        assert stage1, arch
        assert all(n.startswith(head_prefixes(model)) for n in stage1), arch
        set_stage2_late_unfreeze(model)
        stage2 = trainable_param_names(model)
        assert any(n.startswith(head_prefixes(model)) for n in stage2), arch
        assert any(
            not n.startswith(head_prefixes(model)) for n in stage2
        ), arch


def test_class_weights_inverse_frequency():
    w = compute_class_weights([100, 50, 25, 25])
    assert w.tolist() == [2.0, 4.0, 8.0, 8.0]
    assert w[1] > w[0]  # rarer class gets larger weight


def test_class_weights_capped_and_zero_safe():
    w = compute_class_weights([1000, 10])
    assert w.tolist() == pytest.approx([1.01, 10.1])
    w = compute_class_weights([100, 0])
    assert w.tolist() == [1.0, 1.0]


def test_transforms_emit_normalized_224_tensor():
    from PIL import Image

    img = Image.new("RGB", (300, 200), color=(128, 64, 32))
    for t in (make_train_transforms(), make_eval_transforms()):
        out = t(img)
        assert out.shape == (3, 224, 224)


def test_group_split_has_no_group_overlap():
    groups = ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"]
    train_idx, test_idx = group_train_test_split(groups, test_size=0.25, seed=0)
    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    assert train_groups.isdisjoint(test_groups)
    assert len(train_idx) + len(test_idx) == len(groups)


def test_per_class_recall_flags_weak_class():
    cm = [[8, 2], [5, 5]]
    assert per_class_recall_from_confusion(cm) == [0.8, 0.5]


def test_eval_transforms_deterministic_on_same_image():
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(10, 200, 90))
    t = make_eval_transforms()
    a, b = t(img), t(img)
    assert torch.allclose(a, b)
