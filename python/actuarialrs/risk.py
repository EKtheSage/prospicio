"""Risk measures and dependence: distortion risk measures, capital allocation,
copulas, Iman-Conover reordering and extreme value tails."""

from .actuarialrs_native import (
    Allocation,
    ArchimedeanCopula,
    Distortion,
    GaussianCopula,
    Gpd,
    PotTail,
    StudentTCopula,
    allocate,
    capital,
    iman_conover,
    simulate,
)

__all__ = [
    "Distortion",
    "allocate",
    "capital",
    "Allocation",
    "GaussianCopula",
    "StudentTCopula",
    "ArchimedeanCopula",
    "simulate",
    "iman_conover",
    "Gpd",
    "PotTail",
]
