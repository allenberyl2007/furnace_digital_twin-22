#!/usr/bin/env python3
"""
Point d'entrée du jumeau numérique du four à induction.

Exemples d'utilisation :

    # Générer un dataset de 5 épisodes (cycles de fusion complets), pas de temps 1 s
    python main.py dataset --episodes 5 --output data/dataset.csv

    # Mode exploration : masses 1-100 kg, profils de puissance 5-100 %
    python main.py dataset --exploration --episodes 1000 --output data/exploration.csv

    # Lancer une simulation unique et afficher un résumé
    python main.py run --alloy AL6061 --mass 400 --power 400000

"""

from __future__ import annotations

import argparse
import sys

from digital_twin import FurnaceDigitalTwin, FurnaceConfig, DatasetGenerator
from digital_twin.materials import ALLOY_DATABASE


def cmd_dataset(args: argparse.Namespace) -> None:
    generator = DatasetGenerator(
        dt_s=args.dt,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        with_optimal_control=args.with_optimal_control,
        export_format=args.format,
        exploration=args.exploration,
    )
    generator.export_csv(args.output, n_episodes=args.episodes)


def cmd_run(args: argparse.Namespace) -> None:
    config = FurnaceConfig(
        alloy_name=args.alloy,
        mass_kg=args.mass,
        max_power_w=args.power,
        frequency_hz=args.frequency,
        target_temp_c=args.target_temp,
        random_fault_probability_per_hour=args.fault_probability,
        seed=args.seed,
        use_optimal_control=args.optimal_control,
    )
    twin = FurnaceDigitalTwin(config)

    print(f"Alliage : {config.alloy_name} | Masse : {config.mass_kg:.1f} kg | "
          f"Puissance max : {config.max_power_w/1000:.0f} kW | Fréquence : {config.frequency_hz:.0f} Hz | "
          f"Contrôle optimal : {'oui' if config.use_optimal_control else 'non'}")
    print("-" * 90)

    step = 0
    last_print_time = -1e9
    while step < args.max_steps:
        record = twin.step(args.dt)
        if record.time_s - last_print_time >= args.print_every:
            print(
                f"t={record.time_s:7.0f}s  phase={record.phase:8s}  "
                f"T={record.temperature_metal_c:7.1f}°C  "
                f"P={record.active_power_w/1000:7.1f}kW  "
                f"rendement={record.efficiency*100:5.1f}%  "
                f"santé_min={min(record.health_refractory, record.health_coil, record.health_capacitors, record.health_igbt):.3f}  "
                f"alarme={record.alarm}"
            )
            last_print_time = record.time_s
        step += 1
        if twin.is_finished():
            print("-" * 90)
            print(f"Cycle terminé après {record.time_s:.0f} s. "
                  f"Énergie consommée : {record.energy_consumed_j/3.6e6:.2f} kWh")
            break


def cmd_list_alloys(_: argparse.Namespace) -> None:
    for name, alloy in ALLOY_DATABASE.items():
        print(f"{name}: solidus={alloy.solidus_c}°C liquidus={alloy.liquidus_c}°C "
              f"chaleur_latente={alloy.latent_heat_fusion} J/kg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jumeau numérique - four à induction (recyclage aluminium)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dataset = sub.add_parser("dataset", help="Générer un dataset multi-épisodes pour l'entraînement IA")
    p_dataset.add_argument("--episodes", type=int, default=5)
    p_dataset.add_argument("--output", type=str, default="data/dataset.csv")
    p_dataset.add_argument("--dt", type=float, default=1.0, help="Pas de temps en secondes")
    p_dataset.add_argument("--max-steps", dest="max_steps", type=int, default=20000)
    p_dataset.add_argument("--seed", type=int, default=None)
    p_dataset.add_argument(
        "--with-optimal-control", dest="with_optimal_control", action="store_true",
        help="Active le contrôleur prédictif (OptimalController) pour tous les épisodes générés.",
    )
    p_dataset.add_argument(
        "--format", dest="format", type=str, default="csv", choices=["csv", "sqlite"],
        help="Format d'export du dataset (csv ou sqlite).",
    )
    p_dataset.add_argument(
        "--exploration", dest="exploration", action="store_true",
        help=(
            "Active le mode exploration : masse initiale tirée dans "
            "[1, 100] kg et profil de puissance (constant ou par paliers "
            "aléatoires 5-100%%) appliqué via power_pct. Ajoute les "
            "colonnes timestamp, T_bain, m_alu, m_dross au dataset."
        ),
    )
    p_dataset.set_defaults(func=cmd_dataset)

    p_run = sub.add_parser("run", help="Lancer une simulation unique avec affichage console")
    p_run.add_argument("--alloy", type=str, default="AL1050", choices=list(ALLOY_DATABASE.keys()))
    p_run.add_argument("--mass", type=float, default=500.0)
    p_run.add_argument("--power", type=float, default=500_000.0)
    p_run.add_argument("--frequency", type=float, default=1000.0)
    p_run.add_argument("--target-temp", dest="target_temp", type=float, default=720.0)
    p_run.add_argument("--fault-probability", dest="fault_probability", type=float, default=0.05)
    p_run.add_argument("--dt", type=float, default=1.0)
    p_run.add_argument("--max-steps", dest="max_steps", type=int, default=20000)
    p_run.add_argument("--print-every", dest="print_every", type=float, default=30.0)
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument(
        "--optimal-control", dest="optimal_control", action="store_true",
        help="Active le contrôleur prédictif (OptimalController) au lieu de la logique de consigne historique.",
    )
    p_run.set_defaults(func=cmd_run)

    p_alloys = sub.add_parser("alloys", help="Lister les alliages disponibles")
    p_alloys.set_defaults(func=cmd_list_alloys)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
