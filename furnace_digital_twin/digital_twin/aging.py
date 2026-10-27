"""
Module vieillissement
=======================

Fait évoluer progressivement l'état de santé (0 = HS, 1 = neuf) des
composants critiques du four en fonction du temps de fonctionnement, du
nombre de cycles thermiques et des conditions de sollicitation (température,
courant). Ces indices de santé sont ensuite utilisés par les autres modules
(électronique de puissance, électromagnétique) pour dégrader les
performances de manière réaliste, et servent de labels pour la maintenance
prédictive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComponentHealth:
    refractory: float = 1.0
    coil: float = 1.0
    capacitors: float = 1.0
    igbt: float = 1.0

    def as_dict(self) -> dict:
        return {
            "health_refractory": self.refractory,
            "health_coil": self.coil,
            "health_capacitors": self.capacitors,
            "health_igbt": self.igbt,
        }


class AgingModel:
    """Calcule la dégradation cumulative des composants au fil du temps."""

    def __init__(
        self,
        refractory_wear_rate_per_hour: float = 2.5e-5,
        coil_wear_rate_per_hour: float = 1.2e-5,
        capacitor_wear_rate_per_hour: float = 3.0e-5,
        igbt_wear_rate_per_hour: float = 1.8e-5,
        thermal_cycle_extra_wear: float = 5e-4,
    ) -> None:
        self.refractory_wear_rate_per_hour = refractory_wear_rate_per_hour
        self.coil_wear_rate_per_hour = coil_wear_rate_per_hour
        self.capacitor_wear_rate_per_hour = capacitor_wear_rate_per_hour
        self.igbt_wear_rate_per_hour = igbt_wear_rate_per_hour
        self.thermal_cycle_extra_wear = thermal_cycle_extra_wear
        self.health = ComponentHealth()
        self._previous_heating_phase: bool | None = None

    def step(
        self,
        dt_s: float,
        metal_temp_c: float,
        igbt_temp_c: float,
        capacitor_stress_factor: float,
        coil_current_a: float,
        coil_current_nominal_a: float,
        is_heating: bool,
    ) -> ComponentHealth:
        dt_h = dt_s / 3600.0

        # Usure du réfractaire accélérée par la température de la charge
        temp_factor_refractory = max(0.2, metal_temp_c / 700.0)
        self.health.refractory -= self.refractory_wear_rate_per_hour * temp_factor_refractory * dt_h

        # Usure de la bobine accélérée par le rapport de courant au nominal
        overcurrent_factor = max(0.5, (coil_current_a / max(coil_current_nominal_a, 1e-6)) ** 2)
        self.health.coil -= self.coil_wear_rate_per_hour * overcurrent_factor * dt_h

        # Usure des condensateurs liée à leur sollicitation (ripple)
        stress_factor = max(0.3, capacitor_stress_factor * 3.0)
        self.health.capacitors -= self.capacitor_wear_rate_per_hour * stress_factor * dt_h

        # Usure des IGBT accélérée par la température de jonction
        temp_factor_igbt = max(0.3, (igbt_temp_c / 100.0) ** 2)
        self.health.igbt -= self.igbt_wear_rate_per_hour * temp_factor_igbt * dt_h

        # Fatigue thermique supplémentaire à chaque transition
        # chauffage <-> maintien/refroidissement (cyclage thermique)
        if self._previous_heating_phase is not None and is_heating != self._previous_heating_phase:
            self.health.refractory -= self.thermal_cycle_extra_wear
            self.health.coil -= self.thermal_cycle_extra_wear * 0.5
        self._previous_heating_phase = is_heating

        # Clamp dans [0, 1]
        self.health.refractory = max(0.0, min(1.0, self.health.refractory))
        self.health.coil = max(0.0, min(1.0, self.health.coil))
        self.health.capacitors = max(0.0, min(1.0, self.health.capacitors))
        self.health.igbt = max(0.0, min(1.0, self.health.igbt))

        return self.health
