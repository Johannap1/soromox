"""Tests for the pytree finiteness diagnostics."""

import jax
import jax.numpy as jnp
import pytest

from soromox.utils import (
    assert_all_finite,
    format_nonfinite_report,
    is_all_finite,
    nonfinite_leaves,
)

jax.config.update("jax_enable_x64", True)


def test_clean_tree_reports_nothing():
    tree = {"opt_ctr_params": {"Kp": jnp.ones(3), "Kd": jnp.zeros(3)}}

    assert nonfinite_leaves(tree) == []
    assert is_all_finite(tree)


def test_nan_leaf_is_reported_with_its_path():
    tree = {"opt_ctr_params": {"Kp": jnp.array([1.0, jnp.nan, 3.0])}}

    assert nonfinite_leaves(tree) == [("opt_ctr_params.Kp", "nan")]
    assert not is_all_finite(tree)


def test_inf_leaf_is_distinguished_from_nan():
    tree = {"a": jnp.array([jnp.inf]), "b": jnp.array([jnp.nan])}

    assert dict(nonfinite_leaves(tree)) == {"a": "inf", "b": "nan"}


def test_leaf_holding_both_is_labelled_nan_plus_inf():
    assert nonfinite_leaves({"m": jnp.array([jnp.nan, jnp.inf])}) == [("m", "nan+inf")]


def test_integer_leaves_are_skipped():
    assert nonfinite_leaves({"n": jnp.array([1, 2, 3])}) == []


def test_root_scalar_uses_a_placeholder_path():
    assert nonfinite_leaves(jnp.array(jnp.nan)) == [("<root>", "nan")]


def test_format_report_is_readable():
    report = [("opt_ctr_params.Kp", "nan"), ("q_ts", "inf")]

    message = format_nonfinite_report("iteration 3", report)

    assert "iteration 3" in message
    assert "opt_ctr_params.Kp (nan)" in message
    assert "q_ts (inf)" in message
    assert "all leaves finite" in format_nonfinite_report("iteration 3", [])


def test_assert_all_finite_passes_on_a_clean_tree():
    assert_all_finite({"Kp": jnp.ones(3)}, "gradient")


def test_assert_all_finite_names_the_offending_leaves():
    tree = {"grad": {"Kp": jnp.array([jnp.nan])}}

    with pytest.raises(FloatingPointError, match=r"grad\.Kp \(nan\)"):
        assert_all_finite(tree, "gradient")
