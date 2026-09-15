"""Simulator-only Table III D-panel preparation and scoring helpers.

This package is deliberately separate from the real-world D panel.  It creates
an auditable MuJoCo fixture and evaluator-only labels; it never reads or writes
the Real-5 GT and it never generates model predictions.
"""

