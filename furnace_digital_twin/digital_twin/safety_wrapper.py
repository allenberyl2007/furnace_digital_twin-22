"""
Module de garde-fous de sécurité (safety wrapper)
====================================================

Fournit des fonctions indépendantes de tout contrôleur qui bornent et
limitent en rampe les consignes envoyées au procédé (puissance, débit de
refroidissement), afin de garantir qu'aucune consigne — qu'elle provienne
de la logique historique ou du contrôleur optimal — ne puisse dépasser des
limites physiques/sûres ou varier trop brutalement.

Ces fonctions sont volontairement sans état (pas de classe) : l'appelant
est responsable de conserver la valeur précédente d'un pas à l'autre.
"""

from __future__ import annotations

import warnings


def safe_power_setpoint(
    desired_power_w: float,
    previous_power_w: float,
    dt_s: float,
    max_power_w: float,
    max_ramp_w_per_s: float = 5000.0,
    min_power_w: float = 0.0,
) -> float:
    """Retourne une consigne de puissance sécurisée.

    Applique dans l'ordre :
      1. une limite de rampe (variation maximale autorisée par seconde
         entre `previous_power_w` et `desired_power_w`) ;
      2. un écrêtage dans l'intervalle [`min_power_w`, `max_power_w`].

    Un avertissement (`warnings.warn`) est émis si la consigne demandée a
    dû être modifiée.

    Args:
        desired_power_w: Consigne de puissance souhaitée (W).
        previous_power_w: Consigne de puissance appliquée au pas précédent (W).
        dt_s: Pas de temps de la simulation (s).
        max_power_w: Puissance maximale autorisée (W).
        max_ramp_w_per_s: Rampe maximale autorisée (W/s).
        min_power_w: Puissance minimale autorisée (W), 0 par défaut.

    Returns:
        La consigne de puissance effectivement sûre à appliquer (W).
    """
    max_delta = max_ramp_w_per_s * dt_s
    delta = desired_power_w - previous_power_w

    ramped_power_w = desired_power_w
    if delta > max_delta:
        ramped_power_w = previous_power_w + max_delta
    elif delta < -max_delta:
        ramped_power_w = previous_power_w - max_delta

    clamped_power_w = max(min_power_w, min(max_power_w, ramped_power_w))

    if clamped_power_w != desired_power_w:
        warnings.warn(
            f"safe_power_setpoint: consigne de puissance modifiée "
            f"({desired_power_w:.1f} W demandé -> {clamped_power_w:.1f} W appliqué) "
            f"pour respecter la rampe ({max_ramp_w_per_s} W/s) ou les bornes "
            f"[{min_power_w}, {max_power_w}] W.",
            stacklevel=2,
        )

    return clamped_power_w


def safe_cooling_flow(
    desired_flow_fraction: float,
    previous_flow_fraction: float,
    dt_s: float,
    max_ramp_per_s: float = 0.5,
    min_flow_fraction: float = 0.0,
    max_flow_fraction: float = 1.0,
) -> float:
    """Retourne une fraction de débit de refroidissement sécurisée.

    Applique une limite de rampe puis un écrêtage dans
    [`min_flow_fraction`, `max_flow_fraction`], et avertit si la valeur
    demandée a dû être corrigée.

    Args:
        desired_flow_fraction: Fraction de débit souhaitée (0..1).
        previous_flow_fraction: Fraction de débit appliquée au pas précédent (0..1).
        dt_s: Pas de temps de la simulation (s).
        max_ramp_per_s: Variation maximale autorisée par seconde (unité/s).
        min_flow_fraction: Fraction minimale autorisée.
        max_flow_fraction: Fraction maximale autorisée.

    Returns:
        La fraction de débit effectivement sûre à appliquer.
    """
    max_delta = max_ramp_per_s * dt_s
    delta = desired_flow_fraction - previous_flow_fraction

    ramped_fraction = desired_flow_fraction
    if delta > max_delta:
        ramped_fraction = previous_flow_fraction + max_delta
    elif delta < -max_delta:
        ramped_fraction = previous_flow_fraction - max_delta

    clamped_fraction = max(min_flow_fraction, min(max_flow_fraction, ramped_fraction))

    if clamped_fraction != desired_flow_fraction:
        warnings.warn(
            f"safe_cooling_flow: consigne de débit modifiée "
            f"({desired_flow_fraction:.3f} demandé -> {clamped_fraction:.3f} appliqué) "
            f"pour respecter la rampe ({max_ramp_per_s}/s) ou les bornes "
            f"[{min_flow_fraction}, {max_flow_fraction}].",
            stacklevel=2,
        )

    return clamped_fraction
