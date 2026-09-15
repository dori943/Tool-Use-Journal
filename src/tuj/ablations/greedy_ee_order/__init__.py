"""Greedy EE-order experiment (not the without-planner ablation).

Among dependency-ready subgoals, pick the (subgoal, EE) pair with the smallest
immediate EE-switch cost. Production C3-2 / ENTRIES are unchanged; graspable
2F recipes are exposed only when the runner opts into GREEDY_EXTRA_ENTRIES.
"""

from .plan import METHOD, VERSION, greedy_ee_order_plan

__all__ = ["METHOD", "VERSION", "greedy_ee_order_plan"]
