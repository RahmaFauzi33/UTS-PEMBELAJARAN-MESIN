from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# ======== Konfigurasi dasar ========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "movie_model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
DATA_PATH = BASE_DIR / "movie_statistic_dataset.csv"

FEATURE_COLUMNS: List[str] = [
    "genres",
    "runtime_minutes",
    "movie_averageRating",
    "movie_numerOfVotes",
    "approval_Index",
    "Production budget $",
    "Domestic gross $",
]

FIELD_LABELS: Dict[str, str] = {
    "genres": "Genre",
    "runtime_minutes": "Durasi (menit)",
    "movie_averageRating": "Skor rata-rata penonton",
    "movie_numerOfVotes": "Jumlah suara pengguna",
    "approval_Index": "Approval index",
    "Production budget $": "Anggaran produksi (USD)",
    "Domestic gross $": "Pendapatan domestik (USD)",
}

# ======== Inisialisasi aplikasi & resource ========
app = Flask(__name__)

model = joblib.load(MODEL_PATH)
transformer = joblib.load(ENCODER_PATH)
raw_df = pd.read_csv(DATA_PATH)

# Hilangkan baris yang tidak lengkap untuk fitur yang digunakan model
model_df = raw_df.dropna(subset=FEATURE_COLUMNS + ["Worldwide gross $"]).copy()


def _currency(value: float | int) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "-"


app.jinja_env.filters["currency"] = _currency


def _prepare_default_inputs() -> Dict[str, str]:
    defaults: Dict[str, str] = {
        "genres": model_df["genres"].mode().iat[0] if not model_df["genres"].mode().empty else "",
        "runtime_minutes": f"{model_df['runtime_minutes'].median():.0f}",
        "movie_averageRating": f"{model_df['movie_averageRating'].median():.1f}",
        "movie_numerOfVotes": f"{model_df['movie_numerOfVotes'].median():.0f}",
        "approval_Index": f"{model_df['approval_Index'].median():.2f}",
        "Production budget $": f"{model_df['Production budget $'].median():.0f}",
        "Domestic gross $": f"{model_df['Domestic gross $'].median():.0f}",
    }
    return defaults


DEFAULT_INPUTS = _prepare_default_inputs()
GENRE_OPTIONS = sorted(model_df["genres"].dropna().unique())


def _genre_highlights(top_n: int = 4) -> List[Dict[str, int]]:
    if model_df.empty:
        return []
    genre_counts = (
        model_df["genres"]
        .str.get_dummies(sep=",")
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return [{"name": name.strip(), "count": int(count)} for name, count in genre_counts.items()]


SUMMARY = {
    "total_movies": int(len(raw_df)),
    "avg_worldwide": model_df["Worldwide gross $"].mean(),
    "median_budget": model_df["Production budget $"].median(),
    "avg_rating": model_df["movie_averageRating"].mean(),
    "avg_runtime": model_df["runtime_minutes"].mean(),
    "top_genres": _genre_highlights(),
    "top_movies": (
        raw_df[["movie_title", "Worldwide gross $", "genres"]]
        .dropna(subset=["Worldwide gross $"])
        .nlargest(3, "Worldwide gross $")
        .to_dict("records")
    ),
}


def _prepare_features(form_data: Dict[str, str]):
    payload = {}

    for column in FEATURE_COLUMNS:
        value = (form_data.get(column) or "").strip()
        label = FIELD_LABELS.get(column, column)

        if not value:
            raise ValueError(f"Kolom '{label}' wajib diisi.")

        if column == "genres":
            payload[column] = value
            continue

        try:
            payload[column] = float(value)
        except ValueError as exc:  # pragma: no cover - validasi konversi
            raise ValueError(f"Nilai '{label}' harus berupa angka.") from exc

    input_df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    return transformer.transform(input_df)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_value = None
    prediction_label = None
    error_message = None
    form_data = DEFAULT_INPUTS.copy()

    if request.method == "POST":
        form_data = {key: request.form.get(key, "") for key in FEATURE_COLUMNS}
        try:
            features = _prepare_features(form_data)
            prediction_value = float(model.predict(features)[0])
            prediction_label = _currency(prediction_value)
        except ValueError as validation_error:
            error_message = str(validation_error)
        except Exception as unexpected_error:  # pragma: no cover - logging runtime issue
            app.logger.exception("Terjadi kesalahan saat melakukan prediksi: %s", unexpected_error)
            error_message = "Terjadi kesalahan pada sistem. Silakan coba kembali."

    return render_template(
        "index.html",
        title="Model Prediksi Pendapatan Film Berdasarkan Statistik Film Menggunakan Random Forest",
        summary=SUMMARY,
        genres_options=GENRE_OPTIONS,
        form_data=form_data,
        prediction=prediction_label,
        error=error_message,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

