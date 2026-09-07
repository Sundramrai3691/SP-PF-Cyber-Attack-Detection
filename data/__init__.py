"""Adapters for external datasets kept separate from the synthetic SP-PF demo."""

from .faculty_adapter import load_faculty_scenario, list_faculty_scenarios

__all__ = ["load_faculty_scenario", "list_faculty_scenarios"]
