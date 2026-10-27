# Jumeau numérique industriel — Four à induction (recyclage aluminium)

Jumeau numérique modulaire d'un four à induction pour le recyclage de
l'aluminium, écrit en Python pur (bibliothèque standard uniquement).
Il simule les couplages thermique / électromagnétique / électronique de
puissance / refroidissement, injecte du vieillissement et des défauts
réalistes, et génère un jeu de données de télémétrie exploitable pour
l'entraînement d'une IA supervisée (détection d'anomalies, maintenance
prédictive).

## Structure

```
furnace_digital_twin/
├── digital_twin/
│   ├── materials.py           # Alliages d'aluminium (propriétés f(T))
│   ├── thermal.py             # Chauffage / fusion / maintien / refroidissement
│   ├── electromagnetic.py     # Impédance, effet de peau, résonance, couplage
│   ├── power_electronics.py   # Redresseur / bus DC / onduleur IGBT
│   ├── cooling.py             # Circuit de refroidissement à eau
│   ├── sensors.py             # Capteurs virtuels (bruit, dérive, pannes)
│   ├── aging.py                # Vieillissement des composants
│   ├── faults.py               # Injection de défauts (aléatoire/scénario)
│   ├── simulation_engine.py    # Moteur de simulation (orchestrateur)
│   └── database_generator.py   # Génération du dataset multi-épisodes
├── tests/
│   └── test_basic.py
├── data/                        # Sorties CSV (créé à l'usage)
├── main.py                      # CLI
└── README.md
```

## Utilisation

```bash
# Lister les alliages disponibles
python3 main.py alloys

# Lancer une simulation unique avec affichage console
python3 main.py run --alloy AL6061 --mass 400 --power 400000 --frequency 1000

# Générer un dataset de N épisodes (cycles de fusion complets) pour l'IA
python3 main.py dataset --episodes 50 --output data/dataset.csv --dt 2
```

Le dataset CSV contient, pour chaque pas de temps de chaque épisode :
température (vraie et mesurée), puissance active/réactive, facteur de
puissance, fréquence, courant, tension, impédance, températures IGBT /
réfractaire / eau, débit, pression, rendement, énergie consommée, pertes,
indices de santé des composants (réfractaire, bobine, condensateurs,
IGBT), liste des défauts actifs, et deux labels prêts pour l'IA :
`anomaly` (0/1) et `maintenance_alert` (0/1).

## Ajouter un nouvel alliage

```python
from digital_twin.materials import AluminumAlloy, register_alloy

mon_alliage = AluminumAlloy(
    name="AL7075",
    solidus_c=477, liquidus_c=635,
    latent_heat_fusion=380000,
    density_solid=2810, density_liquid=2400,
    specific_heat_points={20: 960, 300: 1050, 477: 1140, 635: 1180},
    resistivity_points={20: 5.0e-8, 477: 12e-8, 635: 25e-8},
    thermal_conductivity_points={20: 130, 477: 155, 635: 95},
    emissivity_points={20: 0.1, 635: 0.3},
    viscosity_liquid_points={635: 0.0013, 900: 0.0009},
)
register_alloy(mon_alliage)
```

## Limites et hypothèses de modélisation

Ce jumeau numérique privilégie la cohérence physique qualitative et la
richesse des données générées (variabilité, corrélations réalistes entre
grandeurs, dynamiques de défauts) plutôt qu'une précision quantitative de
niveau bureau d'études. Notamment :

- le modèle électromagnétique utilise une approximation « une spire
  équivalente » avec un transformateur d'adaptation d'impédance et un
  facteur empirique de concentration de flux, calibrés pour obtenir des
  rendements de couplage réalistes (60-85 %) plutôt qu'une résolution
  de champ (FEM) ;
- le circuit résonant est supposé « auto-accordé » (poursuite de
  résonance), ce que font la plupart des onduleurs industriels réels ;
- les lois de vieillissement sont des lois d'usure paramétriques
  simplifiées (pas de fatigue mécanique/électrochimique détaillée).

Ces choix sont documentés dans le code (docstrings) afin de pouvoir être
raffinés module par module — l'architecture est justement conçue pour
cela (chaque module est indépendant et remplaçable).

## Contrôle optimal, export SQLite et garde-fous de sécurité

Trois modules optionnels, sans dépendance externe, viennent compléter le
jumeau numérique :

- **`digital_twin/control_expert.py`** — `OptimalController` : un
  contrôleur prédictif simplifié (MPC à horizon court, quelques
  trajectoires de puissance candidates évaluées à chaque pas) qui calcule
  la consigne de puissance et la fraction de débit de refroidissement en
  fonction de la température cible, sous contrainte de rampe et avec
  protection thermique des IGBT.
- **`digital_twin/safety_wrapper.py`** — `safe_power_setpoint()` /
  `safe_cooling_flow()` : garde-fous indépendants qui bornent et limitent
  en rampe toute consigne, quelle que soit son origine (logique
  historique ou `OptimalController`), et avertissent (`warnings.warn`) si
  la consigne demandée a dû être corrigée.
- **`digital_twin/data_exporter.py`** — `DataExporter` : export par lots
  (batch) en CSV ou SQLite (bibliothèque standard `sqlite3`, PRAGMA de
  performance appliqués pour l'écriture séquentielle massive).

Ces fonctionnalités sont **désactivées par défaut** : sans action de
l'utilisateur, le comportement du jumeau numérique est strictement
identique à la version précédente (consigne historique, export CSV
direct).

### Activer le contrôle optimal

```bash
# Simulation unique avec contrôleur prédictif
python3 main.py run --alloy AL6061 --mass 400 --power 400000 --optimal-control

# Dataset généré avec contrôle optimal sur chaque épisode
python3 main.py dataset --episodes 20 --with-optimal-control --output data/dataset.csv
```

En Python :

```python
from digital_twin import FurnaceDigitalTwin, FurnaceConfig

config = FurnaceConfig(mass_kg=400, max_power_w=400_000, use_optimal_control=True)
twin = FurnaceDigitalTwin(config)
record = twin.step(dt_s=1.0)
print(record.power_setpoint_w, record.optimal_power_setpoint_w)
```

Les colonnes `optimal_power_setpoint_w` et `optimal_cooling_flow_fraction`
du dataset contiennent la sortie *brute* du contrôleur (avant garde-fous) ;
`power_setpoint_w` et `water_flow_lpm` reflètent la valeur réellement
appliquée après passage par `safe_power_setpoint()` / `safe_cooling_flow()`.

### Exporter en SQLite plutôt qu'en CSV

```bash
python3 main.py dataset --episodes 20 --format sqlite --output data/dataset.db
```

```python
from digital_twin import DatasetGenerator

generator = DatasetGenerator(dt_s=2.0, export_format="sqlite")
generator.export_csv("data/dataset.db", n_episodes=20)  # le nom de la méthode est conservé pour compatibilité
```

### Utiliser `DataExporter` directement

```python
from digital_twin import DataExporter

with DataExporter("data/manual_export.db", export_format="sqlite", batch_size=500) as exporter:
    for row in mes_lignes_de_telemetrie:
        exporter.write_row(row)
```

## Puissance variable (5-100 %), coûts et rendement matière

`FurnaceDigitalTwin.step()` accepte un paramètre optionnel `power_pct`
(fraction de la puissance nominale `config.max_power_w`, entre 0.05 et
1.00) qui remplace pour ce pas-ci la logique de consigne historique ou le
contrôleur optimal. Une valeur hors bornes est ramenée dans l'intervalle
avec un avertissement. **Sécurité thermique absolue**, toujours active
quelle que soit l'origine de la consigne : l'injection de puissance est
coupée dès que le bain atteint 720 °C
(`FurnaceDigitalTwin.ABSOLUTE_MAX_BATH_TEMP_C`).

Chaque pas calcule aussi, de façon cumulée depuis le début du cycle :
`m_dross_kg` (calamine formée, modèle empirique fonction de la fraction
fondue et de la puissance appliquée), `energy_kwh`, `cost_usd`
(`energy_kwh * 0.25 + m_dross_kg * 0.60`) et `yield_pct`
(`(m_initiale - m_dross_kg) / m_initiale * 100`), `m_initiale` étant
`config.mass_kg`.

```python
from digital_twin import FurnaceDigitalTwin, FurnaceConfig

config = FurnaceConfig(mass_kg=25, max_power_w=150_000)
twin = FurnaceDigitalTwin(config)
record = twin.step(dt_s=1.0, power_pct=0.6)  # 60 % de la puissance nominale
print(record.p_effective_w, record.cost_usd, record.yield_pct)
```

### Mode exploration (`--exploration`)

Génère des épisodes avec masse initiale aléatoire dans [1, 100] kg et un
profil de puissance par épisode, soit constant, soit par paliers
aléatoires (nouvelle valeur de `power_pct` entre 5 % et 100 % à des
instants aléatoires). Le dataset gagne les colonnes `timestamp`, `T_bain`,
`power_pct`, `p_effective_w` (= `P_effective`), `m_alu`, `m_dross`,
`energy_kwh`, `cost_usd`, `yield_pct`.

```bash
python3 main.py dataset --exploration --episodes 1000 --output data/exploration.csv
```

Désactivé par défaut (`exploration=False` dans `DatasetGenerator`) : sans
`--exploration`, le comportement (échantillonnage 150-1200 kg, pas de
profil de puissance) est strictement inchangé.

## Qualité et fiabilité du dataset généré

`DatasetGenerator` garantit que **tout dataset produit est directement
exploitable pour l'entraînement d'une IA, sans nettoyage manuel** :

- **Aucun épisode incomplet n'est jamais inclus.** `generate()` écarte et
  remplace automatiquement tout épisode qui n'atteint pas la fin de son
  cycle (chauffage → fusion → maintien → coulée) avant
  `max_steps_per_episode`, jusqu'à obtenir exactement `n_episodes`
  épisodes complets (`RuntimeError` explicite si le plafond de tentatives
  est atteint, plutôt que de livrer un dataset plus petit que demandé
  silencieusement).
- **Aucune colonne critique n'est jamais entièrement NaN.** `power_pct`
  et les deux colonnes réservées au contrôleur optionnel
  (`optimal_power_setpoint_w`, `optimal_cooling_flow_fraction`, non nulles
  uniquement avec `--with-optimal-control`) sont toujours renseignées par
  une valeur numérique valide, y compris hors mode `--exploration`.
- **`export_csv` valide le dataset en continu** : schéma de colonnes
  stable d'une ligne à l'autre, colonnes obligatoires présentes (mode
  exploration), et nombre d'épisodes distincts écrits vérifié égal à
  `n_episodes` en fin de génération.
- **La sécurité thermique absolue (720 °C) est garantie côté génération** :
  `ParameterSampler.sample()` ne tire plus de température de maintien
  cible au-delà de cette limite (bug historique qui empêchait la
  transition vers la phase de maintien, bloquant l'épisode indéfiniment).
  `FurnaceDigitalTwin.is_finished()` considère un cycle terminé dès la
  coulée (entrée en phase `cooling`), plutôt que d'exiger un
  refroidissement passif complet jusqu'à l'ambiante — physiquement trop
  lent pour tenir dans le budget de pas d'un épisode.
- **Les cibles IA ont une variance réelle (audit qualité IA, calibration
  corrigée)** : `DROSS_RATE_COEFFICIENT_KG_M2_S` était calibré ~3000x trop
  bas (perte de charge ≈0,0006 % au lieu des 1-5 % documentés) —
  `yield_pct`/`cost_usd` étaient de fait quasi constants. Recalibré à
  `1e-2` : `yield_pct` couvre désormais ~90-100 % sur un dataset généré
  (std ≈ 0,7), utilisable comme cible de régression. De même, le
  multiplicateur d'usure initiale du réfractaire (`0.6` → `0.95`) rend
  `maintenance_alert=1` effectivement atteignable (auparavant
  structurellement impossible : plancher de santé à usure maximale au-dessus
  du seuil de déclenchement), et la borne de probabilité de pannes en
  exploration (`0.15` → `5.0` panne/heure) porte la proportion d'épisodes
  avec au moins une panne active de ~4 % à ~45-50 %.

```bash
python3 main.py dataset --exploration --episodes 1000 --output data/dataset.csv
```

Validé empiriquement : 1000/1000 épisodes complets, 0 valeur NaN/infinie,
0 dépassement de 720 °C, masses dans [1, 100] kg, `power_pct` toujours
dans {0} ∪ [0.05, 1.00].

## Tests

## Prochaines étapes suggérées pour une mise en production

1. Remplacer le modèle électromagnétique par un solveur FEM 2D
   axisymétrique (ou des abaques calibrés sur mesures réelles) pour
   fiabiliser impédance/rendement en fonction de la géométrie exacte.
2. Ajouter une interface de connexion à un four réel (OPC-UA/Modbus)
   pour rejouer le même schéma de données en usage « shadow twin »
   (comparaison temps réel simulation vs mesures).
3. Ajouter un module de calibration automatique (recalage des
   paramètres du modèle sur un historique de production).
