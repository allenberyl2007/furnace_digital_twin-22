"""
Module électromagnétique
=========================

Calcule les grandeurs électromagnétiques du couple bobine/charge : effet de
peau, résistance et inductance équivalentes ramenées au primaire,
fréquence de résonance du circuit LC, et rendement de couplage.

Les formules utilisées sont des approximations d'ingénierie usuelles pour
le chauffage par induction (modèle à une spire équivalente, couplage
transformateur bobine/charge).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .materials import AluminumAlloy

MU0 = 4 * math.pi * 1e-7  # Perméabilité du vide, H/m


@dataclass
class ElectromagneticState:
    skin_depth_m: float
    coil_resistance_ohm: float
    load_resistance_ohm: float
    total_resistance_ohm: float
    inductance_h: float
    resonance_frequency_hz: float
    coupling_efficiency: float
    impedance_magnitude_ohm: float
    impedance_phase_rad: float


class InductionCoilModel:
    """Modèle simplifié du couplage bobine à induction / charge métallique."""

    def __init__(
        self,
        coil_turns: int = 8,
        coil_radius_m: float = 0.25,
        coil_length_m: float = 0.35,
        load_radius_m: float = 0.20,
        coil_resistivity_ohm_m: float = 1.68e-8,  # cuivre
        coil_wire_radius_m: float = 0.006,
        capacitance_f: float = 200e-6,
        matching_transformer_ratio: float = 22.0,
        coil_to_load_coupling: float = 0.85,
    ) -> None:
        self.coil_turns = coil_turns
        self.coil_radius_m = coil_radius_m
        self.coil_length_m = coil_length_m
        self.load_radius_m = load_radius_m
        self.coil_resistivity_ohm_m = coil_resistivity_ohm_m
        self.coil_wire_radius_m = coil_wire_radius_m
        self.capacitance_f = capacitance_f
        # Rapport de transformation du transformateur d'adaptation
        # d'impédance placé entre l'onduleur et le circuit résonant
        # bobine/charge. Il ramène l'impédance très faible du tank
        # (milliohms) à un niveau exploitable côté onduleur (quelques
        # ohms), comme sur une installation industrielle réelle.
        self.matching_transformer_ratio = matching_transformer_ratio
        # Rendement de couplage bobine/charge fixé par construction
        # (0..1). Remplace l'ancien facteur empirique de "concentration de
        # flux" par une valeur fixe et explicite représentant le
        # pourcentage de la puissance transférée qui se retrouve
        # effectivement dissipée dans la charge (le reste est dissipé
        # dans la résistance de la bobine). Valeur typique observée sur
        # des installations industrielles bien conçues : 0.80-0.90.
        self.coil_to_load_coupling = coil_to_load_coupling

    @staticmethod
    def skin_depth(resistivity_ohm_m: float, frequency_hz: float, relative_permeability: float) -> float:
        if frequency_hz <= 0:
            return float("inf")
        return math.sqrt(
            resistivity_ohm_m / (math.pi * frequency_hz * MU0 * relative_permeability)
        )

    def coil_inductance_h(self) -> float:
        """Inductance d'un solénoïde multicouche (formule de Wheeler)."""
        r = self.coil_radius_m
        l = self.coil_length_m
        n = self.coil_turns
        return (r**2 * n**2) / (9 * r + 10 * l) * 1e-3 * 25.4  # approximation empirique (H)

    def coil_ac_resistance_ohm(self, frequency_hz: float) -> float:
        """Résistance AC du bobinage cuivre, incluant l'effet de peau."""
        skin = self.skin_depth(self.coil_resistivity_ohm_m, frequency_hz, 1.0)
        wire_r = self.coil_wire_radius_m
        effective_radius = wire_r if skin >= wire_r else skin
        # Section effective annulaire utilisée par le courant HF
        area_eff = math.pi * (wire_r**2 - max(0.0, wire_r - effective_radius) ** 2)
        area_eff = max(area_eff, 1e-9)
        length_wire = self.coil_turns * 2 * math.pi * self.coil_radius_m
        return self.coil_resistivity_ohm_m * length_wire / area_eff

    def load_resistance_ohm(
        self,
        alloy: AluminumAlloy,
        temp_c: float,
        frequency_hz: float,
        melt_fraction: float = 0.0,
    ) -> float:
        """Résistance équivalente de la charge ramenée au circuit, dérivée
        de l'effet de peau et de la résistivité de l'alliage.

        La résistivité utilisée n'est plus lue uniquement via la courbe
        continue `alloy.resistivity(temp_c)` : dans la zone pâteuse
        (solidus <= temp_c <= liquidus), elle est interpolée entre la
        résistivité à l'état solide (prise au solidus) et à l'état
        liquide (prise au liquidus) en fonction de `melt_fraction`
        (0 = entièrement solide, 1 = entièrement liquide). Ceci reflète le
        saut de résistivité réel associé au changement de phase, plutôt
        qu'une simple interpolation linéaire en température qui le
        lisserait artificiellement.

        Args:
            alloy: Alliage considéré.
            temp_c: Température courante de la charge (°C).
            frequency_hz: Fréquence de fonctionnement (Hz).
            melt_fraction: Fraction liquide de la charge (0..1), utilisée
                uniquement si `temp_c` est dans la zone pâteuse de
                l'alliage. Valeur par défaut 0.0 pour rester
                rétrocompatible avec les appels existants.

        Returns:
            Résistance équivalente de la charge (ohm).
        """
        if alloy.solidus_c <= temp_c <= alloy.liquidus_c:
            rho_solid = alloy.resistivity(alloy.solidus_c)
            rho_liquid = alloy.resistivity(alloy.liquidus_c)
            melt_fraction = max(0.0, min(1.0, melt_fraction))
            rho = rho_solid + melt_fraction * (rho_liquid - rho_solid)
        else:
            rho = alloy.resistivity(temp_c)

        mu_r = alloy.relative_permeability
        skin = self.skin_depth(rho, frequency_hz, mu_r)
        skin = min(skin, self.load_radius_m)  # l'effet de peau ne dépasse pas le rayon
        surface_eff = 2 * math.pi * self.load_radius_m * skin
        surface_eff = max(surface_eff, 1e-9)
        length_eff = 2 * math.pi * self.load_radius_m * self.coil_turns
        r_load = rho * length_eff / surface_eff
        return r_load

    def coupling_efficiency(self, r_coil: float, r_load: float) -> float:
        """Rendement électrique du transfert bobine -> charge."""
        total = r_coil + r_load
        if total <= 0:
            return 0.0
        return r_load / total

    def resonance_frequency_hz(self, inductance_h: float) -> float:
        if inductance_h <= 0 or self.capacitance_f <= 0:
            return 0.0
        return 1.0 / (2 * math.pi * math.sqrt(inductance_h * self.capacitance_f))

    def compute_state(
        self,
        alloy: AluminumAlloy,
        temp_c: float,
        frequency_hz: float,
        auto_tune: bool = True,
        melt_fraction: float = 0.0,
    ) -> ElectromagneticState:
        """Calcule l'état électromagnétique.

        `auto_tune=True` reproduit le fonctionnement réel d'un four à
        induction industriel : la batterie de condensateurs (ou le
        pilotage de l'onduleur en poursuite de résonance) est ajustée pour
        que le circuit fonctionne à (ou très près de) sa fréquence de
        résonance, afin de maximiser le transfert de puissance. Dans ce
        cas la réactance nette est quasi nulle et l'impédance est dominée
        par la résistance équivalente (bobine + charge).

        `melt_fraction` (0..1) est transmis à `load_resistance_ohm()` pour
        interpoler la résistivité de la charge entre ses valeurs solide et
        liquide lorsque la température est dans la zone pâteuse.
        """
        skin = self.skin_depth(alloy.resistivity(temp_c), frequency_hz, alloy.relative_permeability)
        r_coil_secondary = self.coil_ac_resistance_ohm(frequency_hz)

        # La résistance de charge "utile" est déduite du rendement de
        # couplage fixe (coil_to_load_coupling) plutôt que calculée
        # directement depuis la géométrie brute :
        #   coupling = r_load / (r_coil + r_load)  =>  r_load = r_coil * coupling / (1 - coupling)
        coupling = max(0.0, min(0.999, self.coil_to_load_coupling))
        r_load_secondary_base = r_coil_secondary * coupling / (1 - coupling)

        # La fusion (melt_fraction) continue à moduler légèrement ce
        # couplage : la résistivité brute solide sert de référence, et le
        # rapport résistivité(melt_fraction)/résistivité(solide) vient
        # amplifier ou atténuer r_load_secondary_base en conséquence
        # (résistivité plus élevée à l'état liquide => couplage amélioré).
        r_load_raw_solid = self.load_resistance_ohm(alloy, temp_c, frequency_hz, melt_fraction=0.0)
        r_load_raw_actual = self.load_resistance_ohm(alloy, temp_c, frequency_hz, melt_fraction=melt_fraction)
        melt_ratio = r_load_raw_actual / max(r_load_raw_solid, 1e-12)
        r_load_secondary = r_load_secondary_base * melt_ratio

        efficiency = self.coupling_efficiency(r_coil_secondary, r_load_secondary)
        inductance_secondary = self.coil_inductance_h()

        # Report des grandeurs du "tank" (bobine + charge) côté primaire de
        # l'onduleur à travers le transformateur d'adaptation d'impédance.
        ratio2 = self.matching_transformer_ratio ** 2
        r_coil = r_coil_secondary * ratio2
        r_load = r_load_secondary * ratio2
        r_total = r_coil + r_load
        inductance = inductance_secondary * ratio2

        omega = 2 * math.pi * frequency_hz
        if auto_tune and omega > 0 and inductance > 0:
            # Capacité de la batterie de compensation ajustée pour résonner
            # exactement à la fréquence de fonctionnement.
            tuned_capacitance_f = 1.0 / (omega**2 * inductance)
            self.capacitance_f = tuned_capacitance_f

        f_res = self.resonance_frequency_hz(inductance)
        x_l = omega * inductance
        x_c = 1.0 / (omega * self.capacitance_f) if omega > 0 and self.capacitance_f > 0 else 0.0
        reactance = x_l - x_c
        impedance_mag = math.sqrt(r_total**2 + reactance**2)
        impedance_phase = math.atan2(reactance, r_total) if r_total != 0 else 0.0

        return ElectromagneticState(
            skin_depth_m=skin,
            coil_resistance_ohm=r_coil,
            load_resistance_ohm=r_load,
            total_resistance_ohm=r_total,
            inductance_h=inductance,
            resonance_frequency_hz=f_res,
            coupling_efficiency=efficiency,
            impedance_magnitude_ohm=impedance_mag,
            impedance_phase_rad=impedance_phase,
        )
