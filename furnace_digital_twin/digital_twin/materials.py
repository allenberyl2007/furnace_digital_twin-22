"""
Module matériaux
=================

Modélise les propriétés physiques d'alliages d'aluminium en fonction de la
température. Toutes les propriétés sont exprimées en unités SI.

Le module est conçu pour être extensible : ajouter un alliage revient à
instancier un `AluminumAlloy` avec ses coefficients et à l'enregistrer dans
`ALLOY_DATABASE`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict


KELVIN_OFFSET = 273.15


def c_to_k(temp_c: float) -> float:
    return temp_c + KELVIN_OFFSET


@dataclass
class AluminumAlloy:
    """Représente un alliage d'aluminium avec des propriétés dépendantes
    de la température (interpolation linéaire par morceaux sur des points
    de référence physiquement plausibles).
    """

    name: str
    solidus_c: float          # Température de début de fusion (°C)
    liquidus_c: float         # Température de fin de fusion (°C)
    latent_heat_fusion: float  # J/kg
    density_solid: float      # kg/m^3 (à température ambiante)
    density_liquid: float     # kg/m^3 (état liquide)

    # Points de référence (température °C -> valeur) pour interpolation
    specific_heat_points: Dict[float, float] = field(default_factory=dict)      # J/(kg.K)
    resistivity_points: Dict[float, float] = field(default_factory=dict)        # ohm.m
    thermal_conductivity_points: Dict[float, float] = field(default_factory=dict)  # W/(m.K)
    relative_permeability: float = 1.000021  # Aluminium: quasi non-magnétique
    emissivity_points: Dict[float, float] = field(default_factory=dict)         # sans unité
    viscosity_liquid_points: Dict[float, float] = field(default_factory=dict)   # Pa.s (au-dessus du liquidus)

    def _interp(self, points: Dict[float, float], temp_c: float) -> float:
        xs = sorted(points.keys())
        if temp_c <= xs[0]:
            return points[xs[0]]
        if temp_c >= xs[-1]:
            return points[xs[-1]]
        for x0, x1 in zip(xs, xs[1:]):
            if x0 <= temp_c <= x1:
                y0, y1 = points[x0], points[x1]
                ratio = (temp_c - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return points[xs[-1]]

    def specific_heat(self, temp_c: float) -> float:
        return self._interp(self.specific_heat_points, temp_c)

    def resistivity(self, temp_c: float) -> float:
        return self._interp(self.resistivity_points, temp_c)

    def thermal_conductivity(self, temp_c: float) -> float:
        return self._interp(self.thermal_conductivity_points, temp_c)

    def emissivity(self, temp_c: float) -> float:
        return self._interp(self.emissivity_points, temp_c)

    def viscosity(self, temp_c: float) -> float:
        if temp_c < self.liquidus_c:
            return float("inf")  # solide : viscosité non définie
        return self._interp(self.viscosity_liquid_points, temp_c)

    def density(self, temp_c: float) -> float:
        if temp_c < self.solidus_c:
            return self.density_solid
        if temp_c > self.liquidus_c:
            return self.density_liquid
        # Interpolation dans la zone pâteuse
        ratio = (temp_c - self.solidus_c) / (self.liquidus_c - self.solidus_c)
        return self.density_solid + ratio * (self.density_liquid - self.density_solid)

    def melt_fraction(self, temp_c: float) -> float:
        """Fraction liquide (0 à 1) pour la gestion de la chaleur latente."""
        if temp_c <= self.solidus_c:
            return 0.0
        if temp_c >= self.liquidus_c:
            return 1.0
        return (temp_c - self.solidus_c) / (self.liquidus_c - self.solidus_c)


def _default_alloy_1050() -> AluminumAlloy:
    """Aluminium quasi pur (série 1000), typique du recyclage de câbles/tôles."""
    return AluminumAlloy(
        name="AL1050",
        solidus_c=646.0,
        liquidus_c=657.0,
        latent_heat_fusion=397000.0,
        density_solid=2705.0,
        density_liquid=2380.0,
        specific_heat_points={20: 900.0, 300: 1010.0, 600: 1160.0, 660: 1180.0, 900: 1220.0},
        resistivity_points={20: 2.65e-8, 300: 5.9e-8, 600: 10.2e-8, 660: 24.0e-8, 900: 28.0e-8},
        thermal_conductivity_points={20: 229.0, 300: 210.0, 600: 180.0, 660: 100.0, 900: 95.0},
        emissivity_points={20: 0.09, 300: 0.15, 600: 0.20, 660: 0.30, 900: 0.35},
        viscosity_liquid_points={660: 0.0013, 750: 0.0011, 900: 0.0008},
        relative_permeability=1.000021,
    )


def _default_alloy_6061() -> AluminumAlloy:
    """Alliage Al-Mg-Si, courant en aéronautique/automobile recyclé."""
    return AluminumAlloy(
        name="AL6061",
        solidus_c=582.0,
        liquidus_c=652.0,
        latent_heat_fusion=390000.0,
        density_solid=2700.0,
        density_liquid=2375.0,
        specific_heat_points={20: 896.0, 300: 1020.0, 580: 1150.0, 655: 1170.0, 900: 1210.0},
        resistivity_points={20: 3.99e-8, 300: 7.4e-8, 580: 11.5e-8, 655: 24.5e-8, 900: 28.5e-8},
        thermal_conductivity_points={20: 167.0, 300: 180.0, 580: 175.0, 655: 95.0, 900: 92.0},
        emissivity_points={20: 0.11, 300: 0.17, 580: 0.22, 655: 0.31, 900: 0.36},
        viscosity_liquid_points={655: 0.0014, 750: 0.0012, 900: 0.0009},
        relative_permeability=1.000022,
    )


def _default_alloy_380() -> AluminumAlloy:
    """Alliage de fonderie Al-Si (série 3xx / A380), typique des pièces moulées."""
    return AluminumAlloy(
        name="AL380",
        solidus_c=540.0,
        liquidus_c=595.0,
        latent_heat_fusion=389000.0,
        density_solid=2710.0,
        density_liquid=2380.0,
        specific_heat_points={20: 963.0, 300: 1040.0, 540: 1140.0, 595: 1165.0, 900: 1200.0},
        resistivity_points={20: 5.2e-8, 300: 8.9e-8, 540: 12.3e-8, 595: 25.0e-8, 900: 29.0e-8},
        thermal_conductivity_points={20: 96.0, 300: 120.0, 540: 140.0, 595: 90.0, 900: 88.0},
        emissivity_points={20: 0.12, 300: 0.19, 540: 0.24, 595: 0.32, 900: 0.37},
        viscosity_liquid_points={595: 0.0016, 750: 0.0012, 900: 0.0009},
        relative_permeability=1.000020,
    )


ALLOY_DATABASE: Dict[str, AluminumAlloy] = {
    alloy.name: alloy
    for alloy in (_default_alloy_1050(), _default_alloy_6061(), _default_alloy_380())
}


def register_alloy(alloy: AluminumAlloy) -> None:
    """Permet d'ajouter facilement un nouvel alliage au catalogue global."""
    ALLOY_DATABASE[alloy.name] = alloy


def get_alloy(name: str) -> AluminumAlloy:
    try:
        return ALLOY_DATABASE[name]
    except KeyError as exc:
        raise ValueError(
            f"Alliage inconnu: {name}. Disponibles: {list(ALLOY_DATABASE)}"
        ) from exc
