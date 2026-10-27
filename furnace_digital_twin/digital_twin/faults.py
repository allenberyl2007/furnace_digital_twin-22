"""
Module défauts
================

Injecteur de défauts pour la simulation : permet de déclencher des pannes de
manière aléatoire (probabilité par heure) ou selon un scénario explicite
(liste d'événements programmés), afin de générer des données d'anomalies
étiquetées pour l'entraînement d'une IA de détection/diagnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class FaultType(Enum):
    GRID_VOLTAGE_SAG = "grid_voltage_sag"
    COOLING_LOSS = "cooling_loss"
    SENSOR_FAILURE = "sensor_failure"
    CAPACITOR_DEGRADED = "capacitor_degraded"
    COIL_DAMAGED = "coil_damaged"
    METAL_IMPURITIES = "metal_impurities"
    IGBT_OVERHEAT = "igbt_overheat"
    PARTIAL_SHORT_CIRCUIT = "partial_short_circuit"


@dataclass
class ActiveFault:
    fault_type: FaultType
    start_time_s: float
    duration_s: float
    severity: float  # 0..1
    target_sensor: Optional[str] = None

    def is_active(self, t_s: float) -> bool:
        return self.start_time_s <= t_s < self.start_time_s + self.duration_s


@dataclass
class ScheduledFault:
    fault_type: FaultType
    start_time_s: float
    duration_s: float
    severity: float = 0.5
    target_sensor: Optional[str] = None


class FaultInjector:
    """Gère l'apparition de défauts, aléatoire ou scénarisée."""

    def __init__(
        self,
        random_fault_probability_per_hour: float = 0.05,
        rng: random.Random | None = None,
        sensor_names: list[str] | None = None,
    ) -> None:
        self.random_fault_probability_per_hour = random_fault_probability_per_hour
        self.rng = rng or random.Random()
        self.sensor_names = sensor_names or []
        self.scheduled: list[ScheduledFault] = []
        self.active_faults: list[ActiveFault] = []

    def schedule(self, fault: ScheduledFault) -> None:
        self.scheduled.append(fault)

    def _maybe_trigger_random_fault(self, t_s: float, dt_s: float) -> None:
        probability_this_step = self.random_fault_probability_per_hour * (dt_s / 3600.0)
        if self.rng.random() < probability_this_step:
            fault_type = self.rng.choice(list(FaultType))
            duration_s = self.rng.uniform(30, 900)
            severity = self.rng.uniform(0.2, 1.0)
            target_sensor = (
                self.rng.choice(self.sensor_names)
                if fault_type == FaultType.SENSOR_FAILURE and self.sensor_names
                else None
            )
            self.active_faults.append(
                ActiveFault(fault_type, t_s, duration_s, severity, target_sensor)
            )

    def step(self, t_s: float, dt_s: float) -> list[ActiveFault]:
        # Active les défauts programmés dont l'heure de départ est atteinte
        for sched in list(self.scheduled):
            if sched.start_time_s <= t_s < sched.start_time_s + dt_s:
                self.active_faults.append(
                    ActiveFault(
                        sched.fault_type, sched.start_time_s, sched.duration_s,
                        sched.severity, sched.target_sensor,
                    )
                )
                self.scheduled.remove(sched)

        self._maybe_trigger_random_fault(t_s, dt_s)

        # Nettoie les défauts expirés
        self.active_faults = [f for f in self.active_faults if f.is_active(t_s)]
        return self.active_faults

    def get_severity(self, fault_type: FaultType) -> float:
        """Sévérité maximale actuelle pour un type de défaut donné (0 si absent)."""
        active = [f.severity for f in self.active_faults if f.fault_type == fault_type]
        return max(active) if active else 0.0

    def active_types(self) -> set[FaultType]:
        return {f.fault_type for f in self.active_faults}
