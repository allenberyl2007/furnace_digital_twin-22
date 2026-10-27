"""
Module refroidissement
========================

Simule le circuit de refroidissement à eau de la bobine et de
l'électronique de puissance : débit, températures d'entrée/sortie,
pression, capacité de refroidissement et pertes thermiques évacuées.
"""

from __future__ import annotations

from dataclasses import dataclass

WATER_SPECIFIC_HEAT = 4186.0  # J/(kg.K)
WATER_DENSITY = 1000.0        # kg/m^3


@dataclass
class CoolingState:
    flow_rate_lpm: float
    inlet_temp_c: float
    outlet_temp_c: float
    pressure_bar: float
    heat_removed_w: float
    cooling_capacity_w: float
    is_sufficient: bool


class CoolingSystem:
    """Circuit de refroidissement en boucle (échangeur + pompe)."""

    def __init__(
        self,
        nominal_flow_rate_lpm: float = 40.0,
        nominal_pressure_bar: float = 3.5,
        inlet_temp_c: float = 18.0,
        max_cooling_capacity_w: float = 60000.0,
    ) -> None:
        self.nominal_flow_rate_lpm = nominal_flow_rate_lpm
        self.nominal_pressure_bar = nominal_pressure_bar
        self.inlet_temp_c = inlet_temp_c
        self.max_cooling_capacity_w = max_cooling_capacity_w

    def step(
        self,
        heat_to_remove_w: float,
        flow_fraction: float = 1.0,
        inlet_temp_c: float | None = None,
    ) -> CoolingState:
        """Calcule l'état du circuit de refroidissement pour une charge
        thermique donnée (pertes IGBT + pertes bobine à évacuer).

        `flow_fraction` (0..1) permet de simuler une perte partielle ou
        totale de refroidissement (défaut pompe, colmatage, fuite...).
        """
        flow_fraction = max(0.0, min(1.0, flow_fraction))
        flow_lpm = self.nominal_flow_rate_lpm * flow_fraction
        inlet = inlet_temp_c if inlet_temp_c is not None else self.inlet_temp_c

        cooling_capacity_w = self.max_cooling_capacity_w * flow_fraction
        heat_removed_w = min(heat_to_remove_w, cooling_capacity_w)

        mass_flow_kg_s = (flow_lpm / 60000.0) * WATER_DENSITY  # L/min -> m^3/s -> kg/s
        if mass_flow_kg_s > 1e-6:
            delta_t = heat_removed_w / (mass_flow_kg_s * WATER_SPECIFIC_HEAT)
        else:
            delta_t = 0.0
        outlet_temp_c = inlet + delta_t

        pressure_bar = self.nominal_pressure_bar * (flow_fraction**1.8)

        is_sufficient = heat_to_remove_w <= cooling_capacity_w

        return CoolingState(
            flow_rate_lpm=flow_lpm,
            inlet_temp_c=inlet,
            outlet_temp_c=outlet_temp_c,
            pressure_bar=pressure_bar,
            heat_removed_w=heat_removed_w,
            cooling_capacity_w=cooling_capacity_w,
            is_sufficient=is_sufficient,
        )
