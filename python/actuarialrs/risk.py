"""Risk measures and dependence: distortion risk measures, allocation,
copulas, Iman-Conover reordering and extreme value tails."""

from .actuarialrs_native import (
    ArchimedeanCopula,
    Distortion,
    GaussianCopula,
    Gpd,
    PotTail,
    StudentTCopula,
    allocate,
    iman_conover,
    simulate,
)

__all__ = [
    "Distortion",
    "allocate",
    "GaussianCopula",
    "StudentTCopula",
    "ArchimedeanCopula",
    "simulate",
    "iman_conover",
    "Gpd",
    "PotTail",
]
