"""
Module d'export de données
=============================

Fournit `DataExporter`, une classe unique gérant l'export de la
télémétrie du jumeau numérique soit en CSV, soit en base SQLite, avec
écriture par lots (batch) pour limiter les accès disque/E-S et accélérer
la génération de gros volumes de données.

N'utilise que la bibliothèque standard (`csv`, `sqlite3`).
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Iterable


class DataExporter:
    """Exporte des lignes de télémétrie (dictionnaires) vers un fichier
    CSV ou une base SQLite, avec écriture différée par lots.

    Usage typique :
        exporter = DataExporter("data/dataset.db", export_format="sqlite", batch_size=500)
        for row in generateur_de_lignes:
            exporter.write_row(row)
        exporter.close()
    """

    def __init__(
        self,
        output_path: str | Path,
        export_format: str = "csv",
        batch_size: int = 1000,
        table_name: str = "telemetry",
    ) -> None:
        """Initialise l'exporteur.

        Args:
            output_path: Chemin du fichier de sortie (.csv ou .db/.sqlite).
            export_format: "csv" ou "sqlite".
            batch_size: Nombre de lignes accumulées avant écriture
                physique (flush) sur disque.
            table_name: Nom de la table SQLite utilisée (ignoré en CSV).

        Raises:
            ValueError: si `export_format` n'est ni "csv" ni "sqlite".
        """
        if export_format not in ("csv", "sqlite"):
            raise ValueError(f"export_format doit être 'csv' ou 'sqlite', reçu: {export_format}")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_format = export_format
        self.batch_size = batch_size
        self.table_name = table_name

        self._buffer: list[dict[str, Any]] = []
        self._fieldnames: list[str] | None = None
        self._row_count = 0

        # --- État spécifique CSV ---
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None

        # --- État spécifique SQLite ---
        self._sqlite_conn: sqlite3.Connection | None = None
        if self.export_format == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Ouvre la connexion SQLite et applique les PRAGMA de performance
        adaptés à une écriture séquentielle massive (import en lot, faible
        durabilité acceptable car il s'agit de données de simulation
        régénérables)."""
        self._sqlite_conn = sqlite3.connect(str(self.output_path))
        cursor = self._sqlite_conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = OFF;")
        cursor.execute("PRAGMA temp_store = MEMORY;")
        cursor.execute("PRAGMA cache_size = -64000;")  # ~64 Mo de cache
        self._sqlite_conn.commit()

    def _create_sqlite_table_if_needed(self, row: dict[str, Any]) -> None:
        assert self._sqlite_conn is not None
        columns_sql = ", ".join(f'"{k}" TEXT' for k in row.keys())
        self._sqlite_conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ({columns_sql})'
        )
        self._sqlite_conn.commit()

    def write_row(self, row: dict[str, Any]) -> None:
        """Ajoute une ligne au tampon interne et déclenche un flush
        automatique si `batch_size` est atteint.

        Args:
            row: Dictionnaire représentant un enregistrement de télémétrie.
        """
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            if self.export_format == "sqlite":
                self._create_sqlite_table_if_needed(row)

        self._buffer.append(row)
        self._row_count += 1

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        """Ajoute plusieurs lignes (confort d'utilisation)."""
        for row in rows:
            self.write_row(row)

    def _flush(self) -> None:
        """Écrit physiquement le contenu du tampon sur disque, puis le vide."""
        if not self._buffer:
            return

        if self.export_format == "csv":
            self._flush_csv()
        else:
            self._flush_sqlite()

        self._buffer.clear()

    def _flush_csv(self) -> None:
        if self._csv_file is None:
            self._csv_file = self.output_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
        assert self._csv_writer is not None
        self._csv_writer.writerows(self._buffer)

    def _flush_sqlite(self) -> None:
        assert self._sqlite_conn is not None
        assert self._fieldnames is not None
        placeholders = ", ".join("?" for _ in self._fieldnames)
        columns = ", ".join(f'"{c}"' for c in self._fieldnames)
        sql = f'INSERT INTO "{self.table_name}" ({columns}) VALUES ({placeholders})'
        values = [
            tuple(str(row.get(field, "")) for field in self._fieldnames)
            for row in self._buffer
        ]
        self._sqlite_conn.executemany(sql, values)
        self._sqlite_conn.commit()

    @property
    def row_count(self) -> int:
        """Nombre total de lignes écrites (tampon inclus)."""
        return self._row_count

    def close(self) -> None:
        """Vide le tampon restant et ferme proprement les ressources
        (fichier CSV ou connexion SQLite)."""
        self._flush()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None

    def __enter__(self) -> "DataExporter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
