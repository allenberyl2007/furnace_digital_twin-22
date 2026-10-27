"""
Module électronique de puissance
==================================

Simule la chaîne de conversion : redresseur réseau -> bus continu (avec
condensateurs) -> onduleur IGBT -> circuit résonant série alimentant la
bobine d'induction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .electromagnetic import ElectromagneticState


@dataclass
class PowerElectronicsState:
    dc_bus_voltage_v: float
    dc_bus_ripple_v: float
    inverter_output_current_a: float
    active_power_w: float
    reactive_power_var: float
    apparent_power_va: float
    power_factor: float
    igbt_junction_temp_c: float
    igbt_losses_w: float
    capacitor_stress_factor: float  # 0..1, sollicitation relative des condensateurs


class PowerElectronicsModel:
    """Chaîne redresseur / bus DC / onduleur / circuit résonant."""

    def __init__(
        self,
        grid_voltage_v: float = 400.0,
        rectifier_efficiency: float = 0.98,
        dc_bus_capacitance_f: float = 0.01,
        switching_frequency_hz: float = 20000.0,
        igbt_thermal_resistance_k_per_w: float = 0.12,  # jonction -> ambiant
        igbt_on_resistance_ohm: float = 0.0025,
        ambient_temp_c: float = 35.0,
    ) -> None:
        self.grid_voltage_v = grid_voltage_v
        self.rectifier_efficiency = rectifier_efficiency
        self.dc_bus_capacitance_f = dc_bus_capacitance_f
        self.switching_frequency_hz = switching_frequency_hz
        self.igbt_thermal_resistance_k_per_w = igbt_thermal_resistance_k_per_w
        self.igbt_on_resistance_ohm = igbt_on_resistance_ohm
        self.ambient_temp_c = ambient_temp_c

    def rectified_dc_voltage_v(self, grid_voltage_v: float | None = None) -> float:
        """Redressement triphasé double alternance : Vdc ~ 1.35 * Vligne."""
        v = grid_voltage_v if grid_voltage_v is not None else self.grid_voltage_v
        return 1.35 * v * self.rectifier_efficiency

    def dc_bus_ripple_v(self, power_w: float, dc_voltage_v: float) -> float:
        """Ondulation du bus DC en fonction de la charge et de la capacité."""
        if dc_voltage_v <= 0 or self.dc_bus_capacitance_f <= 0:
            return 0.0
        load_current = power_w / dc_voltage_v
        ripple_period = 1.0 / (6 * 50.0)  # redressement 6 pulses, réseau 50 Hz
        return load_current * ripple_period / self.dc_bus_capacitance_f

    def compute_state(
        self,
        target_power_w: float,
        em_state: ElectromagneticState,
        grid_voltage_v: float | None = None,
        capacitor_health: float = 1.0,
        igbt_health: float = 1.0,
    ) -> PowerElectronicsState:
        """Calcule l'état électrique instantané pour une consigne de
        puissance active donnée, compte tenu de l'impédance du circuit
        résonant (issue du module électromagnétique) et de la santé des
        composants (0..1, 1 = neuf)."""

        dc_voltage = self.rectified_dc_voltage_v(grid_voltage_v)
        # Le condensateur dégradé réduit la tension de bus effective utile
        effective_dc_voltage = dc_voltage * (0.7 + 0.3 * capacitor_health)

        r_total = max(em_state.total_resistance_ohm, 1e-6)
        # Courant nécessaire pour obtenir la puissance active demandée dans
        # la résistance équivalente (P = R * I^2 côté charge+bobine)
        current_rms = math.sqrt(max(target_power_w, 0.0) / r_total)

        # Tension de sortie onduleur nécessaire (loi d'Ohm complexe, module)
        v_out = current_rms * em_state.impedance_magnitude_ohm
        v_out = min(v_out, effective_dc_voltage)  # limité par le bus DC
        # Recalcule le courant réellement atteignable si on est saturé en tension
        if v_out < current_rms * em_state.impedance_magnitude_ohm:
            current_rms = v_out / max(em_state.impedance_magnitude_ohm, 1e-6)

        active_power = current_rms**2 * em_state.total_resistance_ohm
        reactive_power = current_rms**2 * abs(
            em_state.impedance_magnitude_ohm * math.sin(em_state.impedance_phase_rad)
        )
        apparent_power = math.sqrt(active_power**2 + reactive_power**2)
        power_factor = active_power / apparent_power if apparent_power > 0 else 1.0

        # Pertes IGBT : conduction (I^2 R) + commutation (proportionnelles à f et V)
        r_on = self.igbt_on_resistance_ohm / max(igbt_health, 0.05)
        conduction_losses = current_rms**2 * r_on
        switching_losses = 1e-6 * self.switching_frequency_hz * effective_dc_voltage * current_rms
        igbt_losses = conduction_losses + switching_losses
        igbt_junction_temp = self.ambient_temp_c + igbt_losses * self.igbt_thermal_resistance_k_per_w / max(
            igbt_health, 0.05
        )

        ripple = self.dc_bus_ripple_v(active_power, effective_dc_voltage) / max(capacitor_health, 0.05)
        capacitor_stress = min(1.0, ripple / (0.05 * effective_dc_voltage + 1e-9))

        return PowerElectronicsState(
            dc_bus_voltage_v=effective_dc_voltage,
            dc_bus_ripple_v=ripple,
            inverter_output_current_a=current_rms,
            active_power_w=active_power,
            reactive_power_var=reactive_power,
            apparent_power_va=apparent_power,
            power_factor=power_factor,
            igbt_junction_temp_c=igbt_junction_temp,
            igbt_losses_w=igbt_losses,
            capacitor_stress_factor=capacitor_stress,
        )
