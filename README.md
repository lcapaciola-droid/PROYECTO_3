# ⛏️ Productividad de Perforación Minera

Aplicación web interactiva para calcular, simular y analizar la **productividad de perforación** en minería subterránea y superficial.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🖥️ Equipos incluidos

| Equipo | Tipo | Ø (mm) | Aplicación |
|---|---|---|---|
| ⛏️ **Jumbo Hidráulico (Boomer)** | Rotopercutivo | 45 | Galerías, túneles, avances |
| 🔩 **Jack Leg (Neumático)** | Rotopercutivo | 38 | Pequeña minería, labores estrechas |
| 💥 **DTH (Martillo en Fondo)** | DTH | 115 | Taladros largos, chimeneas, cielo abierto |
| 🎯 **Simba (Perforadora Radial)** | Rotopercutivo | 64 | Tajeos, stope drilling |

---

## 📊 Funcionalidades

- ✅ Cálculo determinista: VPef, Tef, MP/guardia, Ch, Cm, TDC, C.Directo
- ✅ **Monte Carlo** hasta 50,000 iteraciones (distribución normal truncada)
- ✅ Histogramas KDE con P10 · P50 · P90
- ✅ Scatter VP vs Costo coloreado por productividad
- ✅ Análisis de sensibilidad ±30% tipo spider
- ✅ Diagrama de Tornado
- ✅ Comparación simultánea 4 equipos (barras + radar + scatter)
- ✅ Estructura de costos: Pie donut + Waterfall
- ✅ Curva Costo total vs metros/mes
- ✅ Superficie 3D VP vs UCS vs CAI
- ✅ Memoria de cálculo completa + exportar CSV

---

## 🚀 Instalación

```bash
git clone https://github.com/TU_USUARIO/drilling-productivity.git
cd drilling-productivity
pip install -r requirements.txt
streamlit run app.py
```

Abre automáticamente en `http://localhost:8501`

---

## 🌐 Deploy en Streamlit Cloud

1. Sube el repositorio a GitHub
2. Entra a [share.streamlit.io](https://share.streamlit.io)
3. **New app** → selecciona tu repo → `app.py` → **Deploy**

---

## 📐 Metodología

| Fórmula | Fuente |
|---|---|
| VPef = VP × U × DM | Llaique & Sánchez (2015) |
| TDC = C_broca/VU + Ch/VPef | Alfredo Camac Torres |
| VP vs UCS: modelo potencial | Bauer & Calder (1967) |
| Coef. Protodyakonov f = UCS/10 | Protodyakonov |
| Monte Carlo (normal truncada) | — |

---

## 📁 Estructura

```
drilling_productivity/
├── app.py            ← Todo el código en un solo archivo
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Autor

**Ing. de Minas — Universidad Nacional del Altiplano Puno**
