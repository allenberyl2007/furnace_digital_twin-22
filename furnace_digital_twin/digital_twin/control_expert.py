"""
Module de commande experte (contrôle optimal)
================================================

Fournit un contrôleur prédictif simplifié (MPC "à horizon court" basé sur
une évaluation de quelques trajectoires candidates plutôt qu'une
optimisation continue complète, afin de rester dépendant uniquement de la
bibliothèque standard) qui calcule à chaque pas de temps la consigne de
puissance et la fraction de débit de refroidissement optimales, sous
contraintes de rampe (limitation de la vitesse de variation de la
consigne) et de température cible.

Ce module est optionnel : le moteur de simulation continue de fonctionner
avec sa logique de consigne historique (`_power_setpoint_w`) si
`use_optimal_control` n'est pas activé dans `FurnaceConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ControlAction:
    """Action de commande calculée par le contrôleur pour un pas de temps.

    Attributes:
        power_setpoint_w: Consigne de puissance active à appliquer (W).
        cooling_flow_fraction: Fraction du débit de refroidissement
            nominal à appliquer (0..1).
    """

    power_setpoint_w: float
    cooling_flow_fraction: float


class OptimalController:
    """Contrôleur prédictif simplifié (MPC) pour le four à induction.

    À chaque appel, le contrôleur évalue un petit nombre de trajectoires
    de puissance candidates sur un horizon court (quelques pas de temps),
    approxime l'évolution de température par une extrapolation linéaire
    locale (dT/dt courant), et retient la candidate qui minimise un coût
    pondérant :
      - l'écart à la température cible en fin d'horizon ;
      - la consommation de puissance (pénalité de sur-régime) ;
      - le dépassement thermique (température au-delà de la cible).

    La consigne retenue est ensuite bornée par une contrainte de rampe
    (variation maximale autorisée par seconde) pour éviter les à-coups
    électriques/thermiques irréalistes, puis la fraction de refroidissement
    est ajustée proportionnellement à la puissance demandée et à la marge
    de température des IGBT.
    """

    def __init__(
        self,
        max_power_w: float,
        target_temp_c: float,
        max_power_ramp_w_per_s: float = 5000.0,
        max_cooling_ramp_per_s: float = 0.2,
        horizon_steps: int = 5,
        igbt_temp_soft_limit_c: float = 120.0,
        igbt_temp_hard_limit_c: float = 150.0,
        initial_power_to_temp_gain_c_per_w_s: float = 1e-6,
    ) -> None:
        """Initialise le contrôleur.

        Args:
            max_power_w: Puissance maximale disponible de l'onduleur (W).
            target_temp_c: Température cible de la charge métallique (°C).
            max_power_ramp_w_per_s: Rampe maximale de puissance (W/s).
            max_cooling_ramp_per_s: Rampe maximale de la fraction de
                débit de refroidissement (unité/s).
            horizon_steps: Nombre de pas de temps évalués pour la
                prédiction à horizon court.
            igbt_temp_soft_limit_c: Seuil au-delà duquel le refroidissement
                est augmenté de manière préventive.
            igbt_temp_hard_limit_c: Seuil au-delà duquel la puissance est
                réduite pour protéger les IGBT.
            initial_power_to_temp_gain_c_per_w_s: Estimation initiale du
                gain thermique (°C par W appliqué, par seconde) utilisée
                pour la prédiction tant qu'aucune mesure fiable de ce gain
                n'est disponible (voir `_power_to_temp_gain` ci-dessous).
        """
        self.max_power_w = max_power_w
        self.target_temp_c = target_temp_c
        self.max_power_ramp_w_per_s = max_power_ramp_w_per_s
        self.max_cooling_ramp_per_s = max_cooling_ramp_per_s
        self.horizon_steps = horizon_steps
        self.igbt_temp_soft_limit_c = igbt_temp_soft_limit_c
        self.igbt_temp_hard_limit_c = igbt_temp_hard_limit_c

        self._previous_power_setpoint_w: float = 0.0
        self._previous_cooling_flow_fraction: float = 1.0

        # Gain thermique estimé (°C par W, par seconde), mis à jour en
        # ligne par moyenne mobile exponentielle dès que la puissance
        # appliquée est significative. Sert de substitut à
        # `dtemp_dt / current_power_w`, qui est indéfini/nul tant que le
        # four n'a pas encore appliqué de puissance non nulle : sans cette
        # estimation de repli, toutes les puissances candidates prédisent
        # la même température de fin d'horizon au démarrage, et le
        # contrôleur choisit alors systématiquement la puissance nulle
        # (moindre coût), restant bloqué à 0 W indéfiniment.
        self._power_to_temp_gain_c_per_w_s = max(1e-12, initial_power_to_temp_gain_c_per_w_s)

    def _candidate_power_fractions(self) -> list[float]:
        """Trajectoires candidates (fractions de la puissance max) évaluées
        à chaque appel. Volontairement discrétisées pour rester légères en
        calcul (pas d'optimisation continue)."""
        return [0.0, 0.25, 0.5, 0.75, 1.0]

    def _predict_end_temperature_c(
        self,
        current_temp_c: float,
        dtemp_dt_c_per_s: float,
        candidate_power_w: float,
        current_power_w: float,
        dt_s: float,
    ) -> float:
        """Extrapolation linéaire locale de la température en fin
        d'horizon, en supposant que le gradient de température actuel
        (`dtemp_dt_c_per_s`) est proportionnel à la puissance appliquée.

        Le facteur de proportionnalité (gain thermique) est estimé en
        ligne : lorsqu'une puissance significative a été appliquée au pas
        précédent, la mesure `dtemp_dt_c_per_s / current_power_w` vient
        rafraîchir `_power_to_temp_gain_c_per_w_s` par moyenne mobile
        exponentielle. Sinon (démarrage à froid, puissance nulle), la
        dernière estimation connue (ou la valeur initiale) est réutilisée
        telle quelle, ce qui évite le blocage à puissance nulle.
        """
        significant_power_threshold_w = 0.02 * self.max_power_w
        if current_power_w > significant_power_threshold_w:
            observed_gain = dtemp_dt_c_per_s / current_power_w
            ema_alpha = 0.3
            self._power_to_temp_gain_c_per_w_s = max(
                1e-12,
                (1 - ema_alpha) * self._power_to_temp_gain_c_per_w_s + ema_alpha * observed_gain,
            )

        predicted_gradient = self._power_to_temp_gain_c_per_w_s * candidate_power_w
        return current_temp_c + predicted_gradient * dt_s * self.horizon_steps

    def _cost(self, predicted_temp_c: float, candidate_power_w: float) -> float:
        """Fonction de coût : écart à la cible + pénalité de consommation
        + pénalité de dépassement thermique."""
        temp_error = predicted_temp_c - self.target_temp_c
        overshoot_penalty = max(0.0, temp_error) ** 2 * 3.0
        undershoot_penalty = max(0.0, -temp_error) ** 2 * 1.0
        power_penalty = (candidate_power_w / max(self.max_power_w, 1.0)) ** 2 * 0.05
        return overshoot_penalty + undershoot_penalty + power_penalty

    def _apply_ramp_limit(
        self, desired_value: float, previous_value: float, max_ramp_per_s: float, dt_s: float
    ) -> float:
        max_delta = max_ramp_per_s * dt_s
        delta = desired_value - previous_value
        if delta > max_delta:
            return previous_value + max_delta
        if delta < -max_delta:
            return previous_value - max_delta
        return desired_value

    def compute_action(
        self,
        current_temp_c: float,
        dtemp_dt_c_per_s: float,
        current_power_w: float,
        igbt_temp_c: float,
        phase_allows_heating: bool,
        dt_s: float,
    ) -> ControlAction:
        """Calcule l'action de commande optimale pour le pas de temps
        courant.

        Args:
            current_temp_c: Température actuelle de la charge (°C).
            dtemp_dt_c_per_s: Gradient de température observé au pas
                précédent (°C/s), utilisé pour l'extrapolation.
            current_power_w: Puissance active appliquée au pas précédent (W).
            igbt_temp_c: Température de jonction IGBT courante (°C), pour
                la protection thermique.
            phase_allows_heating: Indique si la phase de procédé actuelle
                autorise un chauffage actif (False en phase de refroidissement).
            dt_s: Pas de temps de la simulation (s).

        Returns:
            ControlAction: consigne de puissance et fraction de
            refroidissement à appliquer, après application des rampes.
        """
        if not phase_allows_heating:
            desired_power_w = 0.0
        else:
            best_power_w: Optional[float] = None
            best_cost = float("inf")
            for fraction in self._candidate_power_fractions():
                candidate_power_w = fraction * self.max_power_w
                predicted_temp_c = self._predict_end_temperature_c(
                    current_temp_c, dtemp_dt_c_per_s, candidate_power_w, current_power_w, dt_s
                )
                cost = self._cost(predicted_temp_c, candidate_power_w)
                if cost < best_cost:
                    best_cost = cost
                    best_power_w = candidate_power_w
            desired_power_w = best_power_w if best_power_w is not None else 0.0

            # Protection thermique IGBT : réduit la puissance si le seuil
            # dur est dépassé.
            if igbt_temp_c >= self.igbt_temp_hard_limit_c:
                desired_power_w = min(desired_power_w, self.max_power_w * 0.2)

        power_setpoint_w = self._apply_ramp_limit(
            desired_power_w, self._previous_power_setpoint_w, self.max_power_ramp_w_per_s, dt_s
        )
        power_setpoint_w = max(0.0, min(self.max_power_w, power_setpoint_w))

        # Fraction de refroidissement : nominale, augmentée de manière
        # préventive si l'IGBT approche de son seuil d'alerte.
        desired_cooling_fraction = 1.0
        if igbt_temp_c >= self.igbt_temp_soft_limit_c:
            desired_cooling_fraction = 1.0  # déjà au maximum, la marge vient de la puissance
        cooling_flow_fraction = self._apply_ramp_limit(
            desired_cooling_fraction,
            self._previous_cooling_flow_fraction,
            self.max_cooling_ramp_per_s,
            dt_s,
        )
        cooling_flow_fraction = max(0.0, min(1.0, cooling_flow_fraction))

        self._previous_power_setpoint_w = power_setpoint_w
        self._previous_cooling_flow_fraction = cooling_flow_fraction

        return ControlAction(
            power_setpoint_w=power_setpoint_w,
            cooling_flow_fraction=cooling_flow_fraction,
        )
