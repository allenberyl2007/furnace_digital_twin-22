"""
Module capteurs
================

Capteurs virtuels qui transforment une grandeur physique "vraie" (issue des
autres modules) en mesure réaliste : bruit aléatoire, dérive lente dans le
temps, et possibilité de panne (gelé, décroché, hors plage).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class SensorFaultMode(Enum):
    NONE = "none"
    FROZEN = "frozen"          # renvoie toujours la dernière valeur
    DISCONNECTED = "disconnected"  # renvoie None / NaN
    OUT_OF_RANGE = "out_of_range"  # sort de la plage physique
    BIAS_SPIKE = "bias_spike"      # décalage brutal et persistant


@dataclass
class VirtualSensor:
    name: str
    noise_std: float                 # écart-type du bruit blanc, en unité du capteur
    drift_per_hour: float = 0.0      # dérive linéaire lente (unité/h)
    min_value: float = float("-inf")
    max_value: float = float("inf")
    fault_mode: SensorFaultMode = SensorFaultMode.NONE
    _elapsed_hours: float = field(default=0.0, init=False)
    _accumulated_drift: float = field(default=0.0, init=False)
    _last_value: float | None = field(default=None, init=False)
    _bias_spike_value: float = field(default=0.0, init=False)

    def set_fault(self, mode: SensorFaultMode, bias_spike_value: float = 0.0) -> None:
        self.fault_mode = mode
        self._bias_spike_value = bias_spike_value

    def clear_fault(self) -> None:
        self.fault_mode = SensorFaultMode.NONE

    def read(self, true_value: float, dt_s: float, rng: random.Random | None = None) -> float | None:
        rng = rng or random
        self._elapsed_hours += dt_s / 3600.0
        self._accumulated_drift = self.drift_per_hour * self._elapsed_hours

        if self.fault_mode == SensorFaultMode.DISCONNECTED:
            return None

        if self.fault_mode == SensorFaultMode.FROZEN and self._last_value is not None:
            return self._last_value

        noise = rng.gauss(0.0, self.noise_std)
        measured = true_value + noise + self._accumulated_drift

        if self.fault_mode == SensorFaultMode.BIAS_SPIKE:
            measured += self._bias_spike_value

        if self.fault_mode == SensorFaultMode.OUT_OF_RANGE:
            measured = self.max_value + abs(noise) * 10 + 1.0

        measured = max(self.min_value, min(self.max_value, measured))
        self._last_value = measured
        return measured


def make_standard_sensor_suite() -> dict[str, VirtualSensor]:
    """Crée le jeu de capteurs virtuels standard du four."""
    return {
        "thermocouple_metal": VirtualSensor(
            name="thermocouple_metal", noise_std=1.5, drift_per_hour=0.05,
            min_value=-50, max_value=1500,
        ),
        "pyrometer_surface": VirtualSensor(
            name="pyrometer_surface", noise_std=3.0, drift_per_hour=0.08,
            min_value=0, max_value=2000,
        ),
        "current_sensor": VirtualSensor(
            name="current_sensor", noise_std=0.8, drift_per_hour=0.01,
            min_value=0, max_value=5000,
        ),
        "voltage_sensor": VirtualSensor(
            name="voltage_sensor", noise_std=1.0, drift_per_hour=0.01,
            min_value=0, max_value=1200,
        ),
        "power_sensor": VirtualSensor(
            name="power_sensor", noise_std=200.0, drift_per_hour=1.0,
            min_value=0, max_value=2_000_000,
        ),
        "frequency_sensor": VirtualSensor(
            name="frequency_sensor", noise_std=5.0, drift_per_hour=0.02,
            min_value=0, max_value=50_000,
        ),
        "flow_sensor": VirtualSensor(
            name="flow_sensor", noise_std=0.3, drift_per_hour=0.02,
            min_value=0, max_value=200,
        ),
        "pressure_sensor": VirtualSensor(
            name="pressure_sensor", noise_std=0.05, drift_per_hour=0.005,
            min_value=0, max_value=20,
        ),
        "igbt_temp_sensor": VirtualSensor(
            name="igbt_temp_sensor", noise_std=1.0, drift_per_hour=0.03,
            min_value=-20, max_value=200,
        ),
        "refractory_temp_sensor": VirtualSensor(
            name="refractory_temp_sensor", noise_std=2.0, drift_per_hour=0.04,
            min_value=0, max_value=1800,
        ),
        "water_temp_sensor": VirtualSensor(
            name="water_temp_sensor", noise_std=0.4, drift_per_hour=0.01,
            min_value=0, max_value=120,
        ),
    }
