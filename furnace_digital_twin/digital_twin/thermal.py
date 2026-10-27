"""
Module thermique
=================

Simule l'évolution de la température de la charge métallique en fonction de
la puissance injectée et des pertes thermiques (convection, rayonnement,
conduction vers le réfractaire), en tenant compte de la chaleur latente de
fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .materials import AluminumAlloy

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2.K^4)


@dataclass
class ThermalState:
    temperature_c: float
    melt_fraction: float = 0.0
    energy_accumulated_latent_j: float = 0.0


class ThermalModel:
    """Modèle thermique de la charge métallique dans le creuset."""

    def __init__(
        self,
        alloy: AluminumAlloy,
        mass_kg: float,
        exposed_surface_m2: float = 0.5,
        ambient_temp_c: float = 25.0,
        convection_coeff: float = 15.0,       # W/(m^2.K), convection naturelle four ouvert
        refractory_conductance: float = 8.0,  # W/K, conduction vers le réfractaire/creuset
        initial_temp_c: float = 25.0,
    ) -> None:
        self.alloy = alloy
        self.mass_kg = mass_kg
        self.exposed_surface_m2 = exposed_surface_m2
        self.ambient_temp_c = ambient_temp_c
        self.convection_coeff = convection_coeff
        self.refractory_conductance = refractory_conductance
        self.state = ThermalState(temperature_c=initial_temp_c)

    def _radiative_loss_w(self) -> float:
        eps = self.alloy.emissivity(self.state.temperature_c)
        t_surface_k = self.state.temperature_c + 273.15
        t_ambient_k = self.ambient_temp_c + 273.15
        return (
            eps
            * STEFAN_BOLTZMANN
            * self.exposed_surface_m2
            * (t_surface_k**4 - t_ambient_k**4)
        )

    def _convective_loss_w(self) -> float:
        return self.convection_coeff * self.exposed_surface_m2 * (
            self.state.temperature_c - self.ambient_temp_c
        )

    def _conductive_loss_w(self, refractory_temp_c: float) -> float:
        return self.refractory_conductance * max(
            0.0, self.state.temperature_c - refractory_temp_c
        )

    def losses_w(self, refractory_temp_c: float | None = None) -> dict:
        refractory_temp_c = (
            refractory_temp_c if refractory_temp_c is not None else self.ambient_temp_c
        )
        radiative = self._radiative_loss_w()
        convective = self._convective_loss_w()
        conductive = self._conductive_loss_w(refractory_temp_c)
        return {
            "radiative_w": radiative,
            "convective_w": convective,
            "conductive_w": conductive,
            "total_w": radiative + convective + conductive,
        }

    def step(self, power_in_w: float, dt_s: float, refractory_temp_c: float | None = None) -> ThermalState:
        """Fait avancer la simulation thermique d'un pas de temps dt_s (s).

        Gère la fusion via un bilan enthalpique : tant que la charge est dans
        la zone pâteuse (solidus-liquidus), l'énergie nette sert d'abord à
        faire fondre le métal (chaleur latente) avant de faire monter la
        température.
        """
        loss = self.losses_w(refractory_temp_c)
        net_power_w = power_in_w - loss["total_w"]
        energy_in_j = net_power_w * dt_s

        temp = self.state.temperature_c
        in_mushy_zone = self.alloy.solidus_c <= temp <= self.alloy.liquidus_c

        if in_mushy_zone and energy_in_j > 0:
            latent_needed_j = self.mass_kg * self.alloy.latent_heat_fusion
            remaining_latent_j = latent_needed_j - self.state.energy_accumulated_latent_j
            energy_to_latent = min(energy_in_j, max(0.0, remaining_latent_j))
            self.state.energy_accumulated_latent_j += energy_to_latent
            energy_in_j -= energy_to_latent
            self.state.melt_fraction = min(
                1.0, self.state.energy_accumulated_latent_j / latent_needed_j
            )

        cp = self.alloy.specific_heat(temp)
        delta_t = energy_in_j / (self.mass_kg * cp) if self.mass_kg > 0 else 0.0
        new_temp = temp + delta_t

        # Empêche de "sauter" au-dessus du plateau de fusion sans avoir
        # apporté toute la chaleur latente nécessaire.
        if temp < self.alloy.solidus_c and new_temp > self.alloy.solidus_c and energy_in_j > 0:
            new_temp = self.alloy.solidus_c

        self.state.temperature_c = new_temp
        if new_temp >= self.alloy.liquidus_c:
            self.state.melt_fraction = 1.0
        elif new_temp < self.alloy.solidus_c:
            self.state.melt_fraction = 0.0
            self.state.energy_accumulated_latent_j = 0.0

        return self.state
