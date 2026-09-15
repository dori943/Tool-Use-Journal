"""Tests for executed EE-swap counting."""

from tuj.gt.ee_swap_metrics import count_executed_ee_metrics


def test_counts_exchange_not_initial_attach() -> None:
    metrics = count_executed_ee_metrics(
        [
            {"action_type": "EE_ATTACH", "status": "SUCCESS"},
            {"action_type": "EE_EXCHANGE_ENTRY", "status": "SUCCESS"},
            {"action_type": "EE_EXCHANGE", "status": "SUCCESS"},
            {"action_type": "PICK", "status": "SUCCESS"},
            {"action_type": "EE_EXCHANGE", "status": "FAILED"},
        ]
    )
    assert metrics["executed_ee_switches"] == 1
    assert metrics["executed_n_ee_attaches"] == 2
    assert metrics["executed_n_ee_detaches"] == 1
    assert metrics["executed_ee_exchange_entry"] == 1


def test_empty_steps() -> None:
    metrics = count_executed_ee_metrics([])
    assert metrics["executed_ee_switches"] == 0
