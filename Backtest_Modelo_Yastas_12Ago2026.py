from __future__ import annotations
import base64
import io
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────────────────────────────────────
NARANJA = "#E37222"
MORADO  = "#652D86"

PALETA_ROC = [MORADO, NARANJA, "#9B59B6", "#F2A65A", "#3D1F4F"]

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
RUTA_MODELO_PKL = Path(r"D:\Users\lumonroy\OneDrive - Compartamos Banco\Escritorio\Yastas Modelo\1.Modelo Entrenado Yastas\modelos_entrenados.pkl")
CARPETA_SALIDA  = Path(r"D:\Users\lumonroy\OneDrive - Compartamos Banco\Escritorio\Yastas Modelo\3.Backtest Modelo Yastas")
RUTA_EXCEL = CARPETA_SALIDA / "Backtest_Junio_Julio.xlsx"
RUTA_HTML  = CARPETA_SALIDA / "Reporte_Backtest_Modelo_YASTAS.html"


CARPETA_LOGOS = Path(r"D:\Users\lumonroy\OneDrive - Compartamos Banco\Escritorio\Yastas Modelo")
LOGO_GENTERA_NOMBRE = "Gentera_logo"
LOGO_YASTAS_NOMBRE = "logo-yastas"

HOJAS_BACKTEST = ["Junio", "Julio"]


def cargar_modelos_y_cfg():
    with open(RUTA_MODELO_PKL, "rb") as f:
        data = pickle.load(f)
    return data["modelos"], data["cfg"], data["label"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE DATOS 
# ─────────────────────────────────────────────────────────────────────────────
def _normalizar_id(df, hoja, cfg):
    POSIBLES = [cfg.columna_id, "ID COB", "ID COMERCIO aux", "ID CMR aux"]
    enc = next((n for n in POSIBLES if n in df.columns), None)
    if enc is None:
        raise KeyError(f"Columna ID no encontrada en '{hoja}'")
    df = df.copy()
    if enc != "ID COB":
        if "ID COB" in df.columns:
            df.drop(columns=["ID COB"], inplace=True)
        df.rename(columns={enc: "ID COB"}, inplace=True)
    df["ID COB"] = df["ID COB"].astype(str).str.strip()
    return df[~df["ID COB"].isin(["0", "0.0", "nan"])].copy()


def cargar_hoja(cfg, hoja):
    xl = pd.ExcelFile(cfg.ruta_excel)
    hoja_real = next((h for h in xl.sheet_names if h.strip().lower() == hoja.strip().lower()), None)
    if hoja_real is None:
        raise KeyError(f"La hoja '{hoja}' no existe en el archivo Excel.")
    df = pd.read_excel(cfg.ruta_excel, sheet_name=hoja_real)
    df.columns = df.columns.str.strip()
    return _normalizar_id(df, hoja_real, cfg)


def to_numeric_all(df, excluir):
    df = df.copy()
    for col in df.columns:
        if col not in excluir and col != "ID COB":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def aplanar_y_crear_ratios(df):
    df = df.copy()
    txs = np.maximum(df["Transacciones"].values if "Transacciones" in df.columns else 1.0, 1.0)
    if "txs_borde_umbral" in df.columns:
        df["ratio_borde"] = df["txs_borde_umbral"] / txs
    if "velocity_max_1h" in df.columns:
        df["ratio_velocidad"] = df["velocity_max_1h"] / txs
    if "txs_rafaga_menor_60s" in df.columns:
        df["ratio_rafaga"] = df["txs_rafaga_menor_60s"] / txs
    if "max_txs_mismo_monto" in df.columns:
        df["ratio_mismo_monto"] = df["max_txs_mismo_monto"] / txs
    return df


def metricas_completas(y_true, y_pred, y_proba=None):
    has2 = len(np.unique(y_true)) > 1
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if has2 and y_proba is not None else float("nan")
    ap = average_precision_score(y_true, y_proba) if has2 and y_proba is not None else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": prec, "recall": rec, "f1": f1v, "auc": auc, "ap": ap,
        "vp": int(cm[1, 1]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "vn": int(cm[0, 0]),
        "cumple": prec >= 0.75 and rec >= 0.75,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOGOS
# ─────────────────────────────────────────────────────────────────────────────
def _codificar_imagen(carpeta: Path, nombre_base: str) -> str | None:
    """Busca un archivo 'nombre_base.*' en la carpeta y lo devuelve como data-URI base64."""
    candidatos = sorted(carpeta.glob(f"{nombre_base}.*"))
    if not candidatos:
        return None
    ruta = candidatos[0]
    ext = ruta.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_backtest(modelos, cfg, label_map):
    prob_col = {t: f"Prob_{label_map[t].replace(' ', '_').replace('/', '_')}_%" for t in modelos}
    resultados_backtest = {}
    dfs_backtest = {}
    roc_backtest = {}  # roc_backtest[hoja][target] = {"fpr", "tpr", "auc", "label"}

    for hoja in HOJAS_BACKTEST:
        print(f"Evaluando backtest en la hoja '{hoja}'...")
        df_hoja = cargar_hoja(cfg, hoja)
        df_hoja = to_numeric_all(df_hoja, set(cfg.no_features))
        df_hoja = aplanar_y_crear_ratios(df_hoja)

        rows_bt = []
        roc_hoja = {}
        for target, bloque in modelos.items():
            if not bloque:
                continue
            cols_disp = [c for c in bloque["cols"] if c in df_hoja.columns]
            if not cols_disp:
                continue

            X_val = bloque["scaler"].transform(df_hoja[cols_disp].values)
            proba = bloque["modelo"].predict_proba(X_val)[:, 1]
            pred = (proba >= bloque["umbral"]).astype(int)

            df_hoja[prob_col[target]] = (proba * 100).round(2)
            df_hoja[f"Pred_{target}"] = pred

            if target in df_hoja.columns:
                y_val = pd.to_numeric(df_hoja[target], errors="coerce").fillna(0).astype(int)
                met = metricas_completas(y_val.values, pred, proba)
                rows_bt.append({
                    "Hoja": hoja,
                    "Tipología": bloque["label"],
                    "Umbral": bloque["umbral"],
                    "Positivos reales": met.get("vp", 0) + met.get("fn", 0),
                    "Hits (VP)": met.get("vp", "—"),
                    "Falsas alarmas (FP)": met.get("fp", "—"),
                    "Perdidos (FN)": met.get("fn", "—"),
                    "Sin alerta ok (VN)": met.get("vn", "—"),
                    "Captura % (Recall)": round(met.get("recall", 0) * 100, 1),
                    "Precisión %": round(met.get("precision", 0) * 100, 1),
                    "F1 Score": round(met.get("f1", 0), 3),
                    "ROC-AUC": round(met.get("auc", 0), 3),
                    "Average Precision (AP)": round(met.get("ap", 0), 3),
                })

                if len(np.unique(y_val.values)) > 1:
                    fpr, tpr, _ = roc_curve(y_val.values, proba)
                    roc_hoja[target] = {"fpr": fpr, "tpr": tpr, "auc": met.get("auc", float("nan")), "label": bloque["label"]}

        resultados_backtest[hoja] = pd.DataFrame(rows_bt)
        dfs_backtest[hoja] = df_hoja
        roc_backtest[hoja] = roc_hoja

    return resultados_backtest, dfs_backtest, roc_backtest


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA ROC-AUC POR HOJA 
# ─────────────────────────────────────────────────────────────────────────────
def generar_graficas_roc(roc_backtest: dict) -> dict:
    """Devuelve {hoja: data-uri-base64} con una figura de curvas ROC por hoja de backtest."""
    imagenes = {}
    for hoja, roc_hoja in roc_backtest.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, (target, datos) in enumerate(roc_hoja.items()):
            color = PALETA_ROC[i % len(PALETA_ROC)]
            ax.plot(datos["fpr"], datos["tpr"], color=color, linewidth=2.2,
                     label=f"{datos['label']} (AUC={datos['auc']:.3f})")

        ax.plot([0, 1], [0, 1], color="#bbb", linestyle=":", linewidth=1)
        ax.set_title(f"Curvas ROC-AUC — Backtest {hoja}", fontsize=14, fontweight="bold", color=MORADO)
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=10)
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=10)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        imagenes[hoja] = f"data:image/png;base64,{b64}"

    return imagenes


def generar_html(res_backtest: dict, graficas_roc: dict, ruta_html: Path) -> None:
    secciones = ""
    for hoja, df_bt in res_backtest.items():
        filas = ""
        for _, r in df_bt.iterrows():
            filas += f"""
            <tr>
              <td class="tipologia">{r['Tipología']}</td>
              <td>{r['Umbral']}</td>
              <td>{r['Positivos reales']}</td>
              <td>{r['Hits (VP)']}</td>
              <td>{r['Falsas alarmas (FP)']}</td>
              <td>{r['Perdidos (FN)']}</td>
              <td>{r['Captura % (Recall)']}%</td>
              <td>{r['Precisión %']}%</td>
              <td>{r['F1 Score']}</td>
              <td>{r['ROC-AUC']}</td>
            </tr>"""

        grafica_html = ""
        if hoja in graficas_roc:
            grafica_html = f"""
        <div class="grafica">
          <img src="{graficas_roc[hoja]}" alt="Curvas ROC-AUC — {hoja}">
        </div>"""

        secciones += f"""
        <div class="seccion">
          <h2>Backtest — {hoja}</h2>
          {grafica_html}
          <table>
            <thead>
              <tr>
                <th>Tipología</th><th>Umbral</th><th>Positivos reales</th><th>Hits (VP)</th>
                <th>Falsas alarmas (FP)</th><th>Perdidos (FN)</th><th>Captura % (Recall)</th>
                <th>Precisión %</th><th>F1</th><th>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>{filas}</tbody>
          </table>
        </div>"""

    logo_gentera = _codificar_imagen(CARPETA_LOGOS, LOGO_GENTERA_NOMBRE)
    logo_yastas = _codificar_imagen(CARPETA_LOGOS, LOGO_YASTAS_NOMBRE)
    logos_html = ""
    if logo_gentera:
        logos_html += f'<img src="{logo_gentera}" class="logo" alt="Gentera">'
    if logo_yastas:
        logos_html += f'<img src="{logo_yastas}" class="logo" alt="Yastás">'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Backtest Junio–Julio — Modelo Yastás</title>
<style>
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background:#faf8fc; color:#2b2b2b; margin:0; padding:40px;
  }}
  .header {{
    background: linear-gradient(90deg, {MORADO}, {NARANJA});
    border-radius:14px; padding:24px 32px; color:#fff; margin-bottom:28px;
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;
  }}
  .header-texto h1 {{ margin:0; font-size:26px; }}
  .header-texto p {{ margin:6px 0 0; opacity:.9; font-size:14px; }}
  .logos {{ display:flex; align-items:center; gap:14px; }}
  .logo {{
    height:48px; max-width:150px; object-fit:contain;
    background:#fff; border-radius:8px; padding:6px 10px;
  }}
  .seccion {{
    background:#fff; border-radius:12px; padding:24px; margin-bottom:28px;
    box-shadow:0 2px 10px rgba(0,0,0,.06);
  }}
  .seccion h2 {{ color:{MORADO}; margin:0 0 16px; font-size:19px; border-left:5px solid {NARANJA}; padding-left:10px; }}
  .grafica {{ display:flex; justify-content:center; margin-bottom:20px; }}
  .grafica img {{ max-width:520px; width:100%; height:auto; border-radius:8px; }}
  table {{
    width:100%; border-collapse:collapse; background:#fff; border-radius:10px;
    overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,.06);
  }}
  thead th {{
    background:{MORADO}; color:#fff; text-align:left; padding:12px 14px;
    font-size:13px; text-transform:uppercase; letter-spacing:.03em;
  }}
  tbody td {{ padding:11px 14px; border-bottom:1px solid #eee; font-size:14px; }}
  tbody tr:nth-child(even) {{ background:#faf5ff; }}
  tbody tr:hover {{ background:#fdeee0; }}
  td.tipologia {{ font-weight:700; color:{MORADO}; }}
  .footer {{ margin-top:22px; font-size:12px; color:#888; text-align:right; }}
  .badge {{
    display:inline-block; background:{NARANJA}; color:#fff; padding:3px 10px;
    border-radius:20px; font-size:12px; margin-left:10px; vertical-align:middle;
  }}
</style>
</head>
<body>
  <div class="header">
    <div class="header-texto">
      <h1>Modelo Yastás — Backtest Junio y Julio<span class="badge"></span></h1>
      <p>Validación fuera de muestra — meses no usados en el entrenamiento</p>
    </div>
    <div class="logos">{logos_html}</div>
  </div>
  {secciones}
  <div class="footer">Generado automáticamente — Modelo Yastás</div>
</body>
</html>"""
    ruta_html.parent.mkdir(parents=True, exist_ok=True)
    ruta_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    modelos, cfg, label_map = cargar_modelos_y_cfg()
    res_backtest, dfs_backtest, roc_backtest = ejecutar_backtest(modelos, cfg, label_map)
    graficas_roc = generar_graficas_roc(roc_backtest)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(RUTA_EXCEL, engine="openpyxl") as writer:
        for hoja, df_pred in dfs_backtest.items():
            df_pred.to_excel(writer, sheet_name=f"Pred_{hoja}", index=False)
        for hoja, df_bt in res_backtest.items():
            df_bt.to_excel(writer, sheet_name=f"Metricas_{hoja}", index=False)

    generar_html(res_backtest, graficas_roc, RUTA_HTML)

    print(f"Excel guardado en: {RUTA_EXCEL}")
    print(f"HTML guardado en: {RUTA_HTML}")