# =============================================================================
#  PRODUCTIVIDAD DE PERFORACIÓN MINERA — v3.0 (Motor Real por Equipo)
#  Jumbo Hidráulico · Jack Leg · DTH · Simba
#  Fórmulas empíricas reales · Monte Carlo · Análisis de Costos
#  Ref: Bauer & Calder (1967), Paone & Madson (1966), Tandanand (1973),
#       Hustrulid (1999), Jimeno et al. (1995), Camac Torres (2009)
#  Autor: Ing. de Minas — Universidad Nacional del Altiplano Puno
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import truncnorm
from scipy.stats import gaussian_kde as kde_scipy

st.set_page_config(
    page_title="Productividad de Perforación Minera",
    page_icon="⛏️", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{
    background:linear-gradient(135deg,#0a0a1a 0%,#12122a 50%,#0f3460 100%);
    padding:2rem;border-radius:14px;margin-bottom:1.5rem;
    text-align:center;border:1px solid rgba(233,69,96,0.25);
    box-shadow:0 6px 28px rgba(233,69,96,0.18);
}
.main-header h1{color:#e94560;font-size:2.1rem;margin:0;font-weight:700;}
.main-header p{color:#a8b2d8;font-size:0.93rem;margin-top:0.5rem;}
.section-title{
    color:#e94560;font-size:1.15rem;font-weight:600;
    border-left:4px solid #e94560;padding-left:0.8rem;
    margin:1.6rem 0 0.8rem 0;
}
.info-box{
    background:rgba(15,52,96,0.18);border-left:3px solid #e94560;
    padding:0.8rem 1rem;border-radius:0 8px 8px 0;
    color:#a8b2d8;font-size:0.87rem;margin-bottom:1rem;
}
.spec-box{
    background:rgba(22,33,62,0.9);border:1px solid rgba(0,212,170,0.3);
    border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;
}
[data-testid="stSidebar"]{background:#080812;}
.stTabs [data-baseweb="tab-list"]{background:#12122a;border-radius:8px;padding:4px;}
.stTabs [data-baseweb="tab"]{color:#a8b2d8;border-radius:6px;}
.stTabs [aria-selected="true"]{background:rgba(233,69,96,0.15)!important;
    color:#e94560!important;font-weight:600;}
div[data-testid="metric-container"]{
    background:#16213e;border-radius:10px;padding:0.8rem;
    border:1px solid rgba(233,69,96,0.2);
}
div[data-testid="metric-container"] label{color:#a8b2d8!important;font-size:0.8rem!important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
    color:#e94560!important;font-size:1.7rem!important;font-weight:700!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PALETA
# ─────────────────────────────────────────────────────────────────────────────
DARK="#0a0a1a"; CARD="#16213e"; GRID="#1e2a4a"; TEXT="#a8b2d8"
R="#e94560"; G="#00d4aa"; Y="#f39c12"; P="#9b59b6"; B="#3498db"
COLORS=[R,G,Y,P,B,"#e67e22","#1abc9c","#e74c3c"]

def hex2rgba(h,a=0.2):
    h=h.lstrip("#"); r,g,b=int(h[:2],16),int(h[2:4],16),int(h[4:],16)
    return f"rgba({r},{g},{b},{a})"

def _layout(title="",h=440):
    return dict(
        paper_bgcolor=DARK,plot_bgcolor=DARK,height=h,
        font=dict(color=TEXT,family="Arial",size=12),
        title=dict(text=title,font=dict(color="#fff",size=14),x=0.01),
        margin=dict(l=55,r=20,t=52,b=45),
        legend=dict(bgcolor=CARD,bordercolor=GRID,borderwidth=1,font=dict(size=11)),
        xaxis=dict(gridcolor=GRID,zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID,zerolinecolor=GRID),
    )

# ─────────────────────────────────────────────────────────────────────────────
#  ESPECIFICACIONES TÉCNICAS REALES POR EQUIPO
#  Fuentes: Atlas Copco (2022), Sandvik (2021), Epiroc (2023),
#           Jimeno et al. (1995), Hustrulid (1999)
# ─────────────────────────────────────────────────────────────────────────────
EQUIPOS = {
    "Jumbo Hidráulico (Boomer)": {
        "icono":"⛏️",
        "desc":"Equipo electrohidráulico de 1–2 brazos. Rotopercutivo hidráulico de alta frecuencia.",
        "modelo_ref":"Atlas Copco Boomer M2C / Epiroc E2C",
        "tipo":"Rotopercutivo Hidráulico",
        "n_brazos":2,"diam_mm":45,"long_barra_m":3.6,
        "P_perc_kW":18.0,          # Potencia de percusión por brazo (kW)
        "freq_golpes":60,           # Frecuencia de golpes (Hz)
        "presion_bar":200,          # Presión hidráulica (bar)
        "rot_rpm":300,              # Velocidad de rotación (RPM)
        "empuje_kN":18.0,           # Fuerza de empuje (kN)
        "consumo_kWh":55.0,         # Consumo eléctrico total (kWh)
        "consumo_agua_L":80.0,      # Consumo agua (L/min)
        # Parámetros operativos típicos
        "vp_ref":45.0,"ucs_ref":100,"cai_ref":2.0,
        "dm":90,"u":80,"t_pos":3.0,
        # Exponentes empíricos reales (Bauer-Calder/Paone-Madson calibrados)
        "n_ucs":0.42,   # exponente corrección UCS (mayor = más sensible a roca dura)
        "n_cai":0.18,   # exponente corrección CAI
        "n_rqd":0.12,   # factor corrección RQD
        "vp_min":15.0,"vp_max":90.0,
        # Costos reales (USD)
        "c_eq":75.0,"c_mo":18.0,"c_mant":12.0,"c_energia":8.0,
        "p_broca_ref":230.0,"vu_broca_ref":250.0,  # a CAI=2, UCS=100
        "p_barra":800.0,"vu_barra_ref":1000.0,
        # Exponentes desgaste aceros (Tandanand, 1973)
        "k_broca":1.6,  # exponente CAI para vida broca
        "k_ucs_b":0.35, # exponente UCS para vida broca
        "color":"#e94560",
        "app":"Galerías de desarrollo, túneles, avances en minería subterránea",
        "avance_tipo":"desarrollo",  # desarrollo = avance por disparo
        "eficiencia_voladura":0.90,  # eficiencia de avance por disparo
    },
    "Jack Leg (Neumático)": {
        "icono":"🔩",
        "desc":"Perforadora neumática manual tipo jackleg. Baja inversión, operador único.",
        "modelo_ref":"Chicago Pneumatic RH658 / Montabert HC50",
        "tipo":"Rotopercutivo Neumático",
        "n_brazos":1,"diam_mm":38,"long_barra_m":1.8,
        "P_perc_kW":3.5,
        "freq_golpes":38,
        "presion_bar":6,
        "rot_rpm":220,
        "empuje_kN":3.5,
        "consumo_kWh":0.0,         # neumático — costo en aire comprimido
        "consumo_aire_m3min":2.8,  # consumo de aire comprimido (m³/min)
        "consumo_agua_L":20.0,
        "vp_ref":18.0,"ucs_ref":80,"cai_ref":1.5,
        "dm":80,"u":70,"t_pos":8.0,
        "n_ucs":0.35,"n_cai":0.15,"n_rqd":0.20,
        "vp_min":5.0,"vp_max":38.0,
        "c_eq":12.0,"c_mo":8.0,"c_mant":3.0,"c_energia":4.0,  # costo aire
        "p_broca_ref":45.0,"vu_broca_ref":120.0,
        "p_barra":120.0,"vu_barra_ref":400.0,
        "k_broca":1.4,"k_ucs_b":0.30,
        "color":"#00d4aa",
        "app":"Labores estrechas, chiflones, pequeña y mediana minería",
        "avance_tipo":"desarrollo",
        "eficiencia_voladura":0.82,
    },
    "DTH (Martillo en Fondo)": {
        "icono":"💥",
        "desc":"Down-The-Hole. El martillo percute directamente la broca en el fondo. Alta eficiencia en roca dura.",
        "modelo_ref":"Sandvik QL60 / Epiroc Numa Commander 5",
        "tipo":"Percusión en Fondo (DTH)",
        "n_brazos":1,"diam_mm":115,"long_barra_m":6.0,
        "P_perc_kW":35.0,          # Potencia neumática efectiva
        "freq_golpes":25,
        "presion_bar":25,          # Presión de aire (bar)
        "rot_rpm":15,              # Rotación lenta — solo para avanzar
        "empuje_kN":10.0,
        "consumo_kWh":0.0,
        "consumo_aire_m3min":12.0, # gran consumo de aire a alta presión
        "consumo_agua_L":40.0,
        "vp_ref":30.0,"ucs_ref":150,"cai_ref":3.0,
        "dm":85,"u":75,"t_pos":5.0,
        # DTH tiene mayor exponente UCS — trabaja bien en roca muy dura
        "n_ucs":0.50,"n_cai":0.22,"n_rqd":0.08,
        "vp_min":10.0,"vp_max":65.0,
        "c_eq":55.0,"c_mo":15.0,"c_mant":10.0,"c_energia":12.0,
        "p_broca_ref":450.0,"vu_broca_ref":350.0,
        "p_barra":600.0,"vu_barra_ref":800.0,
        "k_broca":1.8,"k_ucs_b":0.40,
        "color":"#f39c12",
        "app":"Taladros largos de producción, chimeneas, piques, minería a cielo abierto",
        "avance_tipo":"produccion",
        "eficiencia_voladura":0.95,
    },
    "Simba (Perforadora Radial)": {
        "icono":"🎯",
        "desc":"Perforadora radial electrohidráulica. Taladros de producción de mediana y gran longitud.",
        "modelo_ref":"Epiroc Simba S7D / Atlas Copco Simba W469",
        "tipo":"Rotopercutivo Hidráulico Radial",
        "n_brazos":1,"diam_mm":64,"long_barra_m":5.0,
        "P_perc_kW":20.0,
        "freq_golpes":55,
        "presion_bar":200,
        "rot_rpm":260,
        "empuje_kN":22.0,
        "consumo_kWh":65.0,
        "consumo_agua_L":60.0,
        "vp_ref":35.0,"ucs_ref":120,"cai_ref":2.5,
        "dm":88,"u":82,"t_pos":4.0,
        "n_ucs":0.44,"n_cai":0.20,"n_rqd":0.10,
        "vp_min":12.0,"vp_max":70.0,
        "c_eq":60.0,"c_mo":16.0,"c_mant":9.0,"c_energia":9.0,
        "p_broca_ref":320.0,"vu_broca_ref":280.0,
        "p_barra":500.0,"vu_barra_ref":700.0,
        "k_broca":1.65,"k_ucs_b":0.38,
        "color":"#9b59b6",
        "app":"Tajeos largos, stope drilling, bench drilling de producción",
        "avance_tipo":"produccion",
        "eficiencia_voladura":0.95,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR DE CÁLCULO REAL
# ─────────────────────────────────────────────────────────────────────────────

def calcular_vp_real(cfg, vp_campo, ucs, cai, rqd):
    """
    VP efectiva corregida por condiciones de roca — fórmula empírica real.

    Modelo: VP_campo es la velocidad medida en condiciones de referencia.
    Se aplican factores de corrección por:
      - UCS: F_ucs = (UCS_ref / UCS)^n_ucs  [Bauer & Calder, 1967]
      - CAI: F_cai = (CAI_ref / CAI)^n_cai  [Tandanand & Unger, 1975]
      - RQD: F_rqd = 1 + n_rqd*(RQD-75)/100  [corrección discontinuidades]
    Cada equipo tiene exponentes calibrados para su tipo de percusión.
    """
    F_ucs = (cfg["ucs_ref"] / max(ucs, 10)) ** cfg["n_ucs"]
    F_cai = (cfg["cai_ref"] / max(cai, 0.1)) ** cfg["n_cai"]
    F_rqd = 1.0 + cfg["n_rqd"] * (rqd - 75) / 100.0
    F_rqd = max(0.70, min(1.25, F_rqd))
    vp_real = vp_campo * F_ucs * F_cai * F_rqd
    return round(float(np.clip(vp_real, cfg["vp_min"], cfg["vp_max"])), 3), \
           round(F_ucs,3), round(F_cai,3), round(F_rqd,3)

def calcular_vida_broca_real(cfg, ucs, cai):
    """
    Vida útil real de broca en función de UCS y CAI.
    Modelo: VU = VU_ref / ((CAI/CAI_ref)^k_broca * (UCS/UCS_ref)^k_ucs)
    Ref: Tandanand (1973), Thuro (1997)
    """
    f_cai = (cai / max(cfg["cai_ref"], 0.1)) ** cfg["k_broca"]
    f_ucs = (ucs / max(cfg["ucs_ref"], 10))  ** cfg["k_ucs_b"]
    vu = cfg["vu_broca_ref"] / (f_cai * f_ucs)
    return round(float(np.clip(vu, 20.0, cfg["vu_broca_ref"] * 3.0)), 1)

def calcular_vida_barra_real(cfg, ucs, cai):
    """Vida útil de barra — menor sensibilidad que la broca."""
    f_cai = (cai / max(cfg["cai_ref"], 0.1)) ** (cfg["k_broca"] * 0.5)
    f_ucs = (ucs / max(cfg["ucs_ref"], 10))  ** (cfg["k_ucs_b"] * 0.4)
    vu = cfg["vu_barra_ref"] / (f_cai * f_ucs)
    return round(float(np.clip(vu, 100.0, cfg["vu_barra_ref"] * 2.5)), 1)

def calcular_productividad(cfg, vp_campo, dm, u, t_guardia, t_pos,
                           c_eq, c_mo, c_mant, c_energia,
                           p_broca, vu_broca, p_barra, vu_barra,
                           ucs, cai, rqd):
    """
    Cálculo completo de productividad real.

    Para equipos de DESARROLLO (Jumbo, Jack Leg):
      T_taladro = L_barra/VP_real + t_pos/60  (h/taladro)
      N_taladros = Tef / T_taladro
      MP/g = N_taladros × L_barra × Eficiencia × N_brazos

    Para equipos de PRODUCCIÓN (DTH, Simba):
      MP/g = VP_real × Tef × N_brazos  (metros lineales)
    """
    vp_real, F_ucs, F_cai, F_rqd = calcular_vp_real(cfg, vp_campo, ucs, cai, rqd)
    vp_ef   = vp_real * u * dm
    t_ef    = t_guardia * u * dm

    if cfg["avance_tipo"] == "desarrollo":
        L  = cfg["long_barra_m"]
        t_tal = L / max(vp_ef, 0.01) + t_pos / 60.0  # h/taladro
        n_tal = t_ef / max(t_tal, 0.001)
        mp_g  = n_tal * L * cfg["eficiencia_voladura"] * cfg["n_brazos"]
        n_tal_g = round(n_tal, 1)
    else:
        mp_g    = vp_ef * t_ef * cfg["n_brazos"]
        n_tal_g = round(mp_g / cfg["long_barra_m"], 1)

    # Costos por metro
    ch      = c_eq + c_mo + c_mant + c_energia
    cm      = ch / max(vp_ef, 0.001)
    c_br_m  = p_broca  / max(vu_broca, 1.0)
    c_ba_m  = p_barra  / max(vu_barra, 1.0)
    c_ac_m  = c_br_m + c_ba_m
    tdc     = c_br_m + cm
    c_dir   = cm + c_ac_m

    # Métricas adicionales
    mp_turno_mes = mp_g * 3 * 26           # 3 guardias/día × 26 días/mes
    costo_mes    = c_dir * mp_turno_mes

    return dict(
        vp_real=vp_real, vp_ef=round(vp_ef,3),
        F_ucs=F_ucs, F_cai=F_cai, F_rqd=F_rqd,
        t_ef=round(t_ef,3), mp_g=round(mp_g,2),
        n_tal_g=n_tal_g,
        ch=round(ch,3), cm=round(cm,4),
        c_br_m=round(c_br_m,4), c_ba_m=round(c_ba_m,4),
        c_ac_m=round(c_ac_m,4), tdc=round(tdc,4),
        c_dir=round(c_dir,4),
        mp_mes=round(mp_turno_mes,1),
        costo_mes=round(costo_mes,0),
    )

# ─────────────────────────────────────────────────────────────────────────────
#  MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def _tnorm(mu, cv, n, lo=0.3, hi=2.5):
    s = mu * cv
    if s < 1e-9: return np.full(n, mu)
    a, b = (mu*lo - mu)/s, (mu*hi - mu)/s
    return truncnorm.rvs(a, b, loc=mu, scale=s, size=n)

def sim_mc(cfg, vp_campo, dm, u, tg, t_pos,
           c_eq, c_mo, c_mant, c_en,
           p_broca, vu_broca, p_barra, vu_barra,
           ucs, cai, rqd, n_sim, cv_vp, cv_dm, cv_u, cv_c, cv_roca):
    np.random.seed(42)
    # Variables aleatorias
    vp_s   = _tnorm(vp_campo, cv_vp, n_sim)
    dm_s   = np.clip(_tnorm(dm,   cv_dm, n_sim, 0.6, 1.1), 0.40, 0.99)
    u_s    = np.clip(_tnorm(u,    cv_u,  n_sim, 0.6, 1.1), 0.40, 0.99)
    ucs_s  = np.clip(_tnorm(ucs,  cv_roca, n_sim), 10, 400)
    cai_s  = np.clip(_tnorm(cai,  cv_roca, n_sim), 0.1, 6.0)
    rqd_s  = np.clip(_tnorm(rqd,  cv_roca*0.5, n_sim), 10, 100)
    ch_s   = (_tnorm(c_eq, cv_c,n_sim)+_tnorm(c_mo,cv_c,n_sim)+
              _tnorm(c_mant,cv_c,n_sim)+_tnorm(c_en,cv_c,n_sim))

    # Correcciones por roca — vectorizadas
    F_ucs_s = (cfg["ucs_ref"] / np.maximum(ucs_s, 10)) ** cfg["n_ucs"]
    F_cai_s = (cfg["cai_ref"] / np.maximum(cai_s, 0.1)) ** cfg["n_cai"]
    F_rqd_s = np.clip(1.0 + cfg["n_rqd"]*(rqd_s-75)/100, 0.70, 1.25)
    vp_real_s = np.clip(vp_s * F_ucs_s * F_cai_s * F_rqd_s,
                        cfg["vp_min"], cfg["vp_max"])
    vp_ef_s = vp_real_s * u_s * dm_s
    t_ef_s  = tg * u_s * dm_s

    if cfg["avance_tipo"] == "desarrollo":
        L = cfg["long_barra_m"]
        t_tal_s = L / np.maximum(vp_ef_s, 0.01) + t_pos/60
        n_tal_s = t_ef_s / np.maximum(t_tal_s, 0.001)
        mp_s = n_tal_s * L * cfg["eficiencia_voladura"] * cfg["n_brazos"]
    else:
        mp_s = vp_ef_s * t_ef_s * cfg["n_brazos"]

    # Vida broca simulada
    f_cai_b = (cai_s/max(cfg["cai_ref"],0.1))**cfg["k_broca"]
    f_ucs_b = (ucs_s/max(cfg["ucs_ref"],10))**cfg["k_ucs_b"]
    vu_br_s = np.clip(vu_broca/(f_cai_b*f_ucs_b), 20, vu_broca*3)

    cm_s  = np.where(vp_ef_s>0, ch_s/vp_ef_s, 0)
    cb_m  = p_broca / np.maximum(vu_br_s, 1.0)
    tdc_s = cb_m + cm_s

    return dict(vp_s=vp_ef_s, mp_s=mp_s, cm_s=cm_s, tdc_s=tdc_s,
                ucs_s=ucs_s, vu_br_s=vu_br_s)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros de Entrada")
    equipo_sel = st.selectbox("🔩 Tipo de Equipo", list(EQUIPOS.keys()))
    cfg = EQUIPOS[equipo_sel]
    st.caption(f"{cfg['icono']} **{cfg['modelo_ref']}**")
    st.caption(f"_{cfg['desc']}_")

    st.markdown("### 🪨 Propiedades de la Roca")
    ucs   = st.slider("UCS — Resistencia Compresión (MPa)", 20, 300, cfg["ucs_ref"], 5)
    cai   = st.slider("CAI — Índice Cerchar", 0.1, 5.0, float(cfg["cai_ref"]), 0.1)
    rqd   = st.slider("RQD (%)", 10, 100, 75, 5)
    f_pr  = round(ucs/10, 1)
    st.caption(f"f Protodyakonov = **{f_pr}** | Clase roca: **{'Muy dura' if ucs>200 else 'Dura' if ucs>100 else 'Media' if ucs>50 else 'Blanda'}**")

    st.markdown("### ⏱️ Parámetros Operativos")
    vp_campo = st.number_input("VP campo medida (m/h)", 2.0, 150.0,
                                float(cfg["vp_ref"]), 0.5,
                                help="Velocidad medida en campo en condiciones reales de la mina")
    dm       = st.slider("Disponibilidad Mecánica (%)", 60, 99, cfg["dm"])
    u_eq     = st.slider("Utilización del Equipo (%)", 50, 99, cfg["u"])
    t_guard  = st.selectbox("Duración de Guardia (h)", [8, 10, 12])
    t_pos    = st.number_input("Tiempo posicionam. (min/taladro)", 1.0, 30.0,
                                float(cfg["t_pos"]), 0.5)

    st.markdown(f"### 💰 Costos Operativos (USD/h)")
    c_eq   = st.number_input("Costo equipo (alquiler/deprec.)", 5.0, 500.0, float(cfg["c_eq"]), 1.0)
    c_mo   = st.number_input("Mano de obra (operador+ayud.)", 3.0, 100.0, float(cfg["c_mo"]), 0.5)
    c_mant = st.number_input("Mantenimiento preventivo/correctivo", 1.0, 80.0, float(cfg["c_mant"]), 0.5)
    c_en   = st.number_input(
        "Energía (electricidad / aire comprimido)",
        0.5, 80.0, float(cfg["c_energia"]), 0.5,
        help="Eléctrico para Jumbo/Simba · Aire comprimido para Jack Leg y DTH"
    )

    st.markdown("### 🔧 Aceros de Perforación")
    # Vida útil calculada automáticamente según roca
    vu_br_auto = calcular_vida_broca_real(cfg, ucs, cai)
    vu_ba_auto = calcular_vida_barra_real(cfg, ucs, cai)
    st.info(f"**Vida broca estimada:** {vu_br_auto} m  \n"
            f"**Vida barra estimada:** {vu_ba_auto} m  \n"
            f"_(calculada por UCS={ucs} MPa · CAI={cai})_")
    p_broca  = st.number_input("Precio broca ($)", 10.0, 800.0, float(cfg["p_broca_ref"]), 5.0)
    vu_broca = st.number_input("Vida útil broca (m) — ajustable", 10.0, 800.0, float(vu_br_auto), 5.0,
                                help="Auto-calculada por condición de roca, puede ajustarse")
    p_barra  = st.number_input("Precio barra ($)", 30.0, 2000.0, float(cfg["p_barra"]), 10.0)
    vu_barra = st.number_input("Vida útil barra (m) — ajustable", 50.0, 2000.0, float(vu_ba_auto), 10.0)

    st.markdown("### 🎲 Monte Carlo")
    n_sim   = st.selectbox("N° Simulaciones", [1000, 5000, 10000, 50000], index=2)
    cv_vp   = st.slider("CV Velocidad (%)",    2, 30, 12)
    cv_dm   = st.slider("CV Disponib. (%)",    2, 20,  5)
    cv_u    = st.slider("CV Utilización (%)",  2, 20,  5)
    cv_cost = st.slider("CV Costos (%)",       2, 25,  8)
    cv_roca = st.slider("CV Propied. Roca (%)",2, 25, 10,
                         help="Variabilidad de UCS, CAI, RQD por heterogeneidad del macizo")

# ─────────────────────────────────────────────────────────────────────────────
#  CÁLCULOS
# ─────────────────────────────────────────────────────────────────────────────
res = calcular_productividad(
    cfg=cfg, vp_campo=vp_campo, dm=dm/100, u=u_eq/100,
    t_guardia=t_guard, t_pos=t_pos,
    c_eq=c_eq, c_mo=c_mo, c_mant=c_mant, c_energia=c_en,
    p_broca=p_broca, vu_broca=vu_broca,
    p_barra=p_barra, vu_barra=vu_barra,
    ucs=ucs, cai=cai, rqd=rqd,
)

mc = sim_mc(
    cfg=cfg, vp_campo=vp_campo, dm=dm/100, u=u_eq/100,
    tg=t_guard, t_pos=t_pos,
    c_eq=c_eq, c_mo=c_mo, c_mant=c_mant, c_en=c_en,
    p_broca=p_broca, vu_broca=vu_broca,
    p_barra=p_barra, vu_barra=vu_barra,
    ucs=ucs, cai=cai, rqd=rqd,
    n_sim=n_sim, cv_vp=cv_vp/100, cv_dm=cv_dm/100,
    cv_u=cv_u/100, cv_c=cv_cost/100, cv_roca=cv_roca/100,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CABECERA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <h1>⛏️ Productividad de Perforación Minera</h1>
  <p>Motor de cálculo real por equipo · Monte Carlo · Análisis de Costos<br>
  <small>Ref: Bauer & Calder (1967) · Paone & Madson (1966) · Tandanand (1973) · Hustrulid (1999)</small></p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FACTORES DE CORRECCIÓN visibles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<p class="section-title">{cfg["icono"]} {equipo_sel} — Factores de Corrección por Roca</p>',
            unsafe_allow_html=True)
fc1,fc2,fc3,fc4 = st.columns(4)
fc1.metric("F_UCS (resistencia)", f"{res['F_ucs']:.3f}",
           f"UCS={ucs} MPa · ref={cfg['ucs_ref']} · n={cfg['n_ucs']}")
fc2.metric("F_CAI (abrasividad)", f"{res['F_cai']:.3f}",
           f"CAI={cai} · ref={cfg['cai_ref']} · n={cfg['n_cai']}")
fc3.metric("F_RQD (fracturación)", f"{res['F_rqd']:.3f}",
           f"RQD={rqd}% · ref=75% · n={cfg['n_rqd']}")
fc4.metric("VP Real corregida", f"{res['vp_real']:.2f} m/h",
           f"VP campo: {vp_campo} m/h → ×{res['F_ucs']*res['F_cai']*res['F_rqd']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
#  KPIs PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📊 Indicadores Clave de Productividad</p>', unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("VP Efectiva",      f"{res['vp_ef']:.3f} m/h",   f"Real×DM×U")
k2.metric("Tiempo Efectivo",  f"{res['t_ef']:.2f} h/g",    f"DM={dm}%  U={u_eq}%")
k3.metric("Metros/Guardia",   f"{res['mp_g']:.1f} m",      f"~{res['n_tal_g']} tal.")
k4.metric("Costo Horario",    f"${res['ch']:.2f}/h",       f"Eq+MO+Mant+Energ")
k5.metric("Costo/Metro",      f"${res['cm']:.4f}/m",       f"TDC: ${res['tdc']:.4f}")
k6.metric("Prod. Mensual",    f"{res['mp_mes']:,.0f} m/mes",f"≈${res['costo_mes']:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🎲 Monte Carlo","📈 Sensibilidad","⚙️ Comparación",
    "💰 Costos","📉 VP vs Roca","📋 Memoria de Cálculo",
])

# ════════════════════════════════════════════════════════════════════════
#  TAB 1 — MONTE CARLO
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">🎲 Simulación Monte Carlo</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    <b>{n_sim:,} iteraciones</b> — Distribución normal truncada.<br>
    Variables estocásticas: VP (CV={cv_vp}%), DM (CV={cv_dm}%), U (CV={cv_u}%),
    Costos (CV={cv_cost}%), <b>UCS+CAI+RQD (CV={cv_roca}%)</b> — incluye variabilidad geológica real del macizo.
    </div>""", unsafe_allow_html=True)

    def hist_kde(data, nombre, unidad, color):
        p10,p50,p90=np.percentile(data,[10,50,90])
        mu,sig=np.mean(data),np.std(data)
        counts,edges=np.histogram(data,bins=55)
        cx=(edges[:-1]+edges[1:])/2; bw=edges[1]-edges[0]
        try:
            kd=kde_scipy(data); kx=np.linspace(data.min(),data.max(),300)
            ky=kd(kx)*len(data)*bw
        except Exception:
            kx,ky=cx,counts.astype(float)
        fig=go.Figure()
        fig.add_trace(go.Bar(x=cx,y=counts,name="Frecuencia",width=bw*0.9,
                             marker=dict(color=color,opacity=0.72)))
        fig.add_trace(go.Scatter(x=kx,y=ky,mode="lines",name="KDE",
                                 line=dict(color="#fff",width=2)))
        for pv,pl,pc in [(p10,"P10",Y),(p50,"P50","#fff"),(p90,"P90",G)]:
            fig.add_vline(x=pv,line=dict(color=pc,dash="dash",width=1.5),
                          annotation=dict(text=f"{pl}={pv:.2f}",
                          font=dict(color=pc,size=10),yref="paper",y=1.05))
        fig.add_annotation(x=0.97,y=0.95,xref="paper",yref="paper",
                           text=f"μ={mu:.2f}  σ={sig:.2f}",showarrow=False,
                           font=dict(color=TEXT,size=10),
                           bgcolor=CARD,bordercolor=GRID,borderwidth=1)
        fig.update_layout(**_layout(f"Monte Carlo — {nombre}"))
        fig.update_xaxes(title_text=unidad); fig.update_yaxes(title_text="Frecuencia")
        return fig

    c1,c2=st.columns(2)
    with c1: st.plotly_chart(hist_kde(mc["mp_s"],"Metros/Guardia","m/guardia",R),use_container_width=True)
    with c2: st.plotly_chart(hist_kde(mc["cm_s"],"Costo/Metro","USD/m",G),use_container_width=True)
    c3,c4=st.columns(2)
    with c3: st.plotly_chart(hist_kde(mc["vp_s"],"VP Efectiva","m/h",Y),use_container_width=True)
    with c4: st.plotly_chart(hist_kde(mc["tdc_s"],"TDC — Total Drilling Cost","USD/m",P),use_container_width=True)

    # Scatter VP vs Costo coloreado por UCS simulado
    st.markdown('<p class="section-title">🔀 Dispersión VP vs Costo/m — coloreado por UCS simulado</p>', unsafe_allow_html=True)
    idx=np.random.choice(len(mc["vp_s"]),min(3000,n_sim),replace=False)
    fig_sc=go.Figure(go.Scatter(
        x=mc["vp_s"][idx],y=mc["cm_s"][idx],mode="markers",
        marker=dict(color=mc["ucs_s"][idx],colorscale="RdYlGn_r",size=4,opacity=0.55,
                    colorbar=dict(title="UCS (MPa)",thickness=12,x=1.01)),
    ))
    fig_sc.update_layout(**_layout("VP Efectiva vs Costo/m (color = UCS)",h=380))
    fig_sc.update_xaxes(title_text="VP Efectiva (m/h)")
    fig_sc.update_yaxes(title_text="Costo/Metro (USD/m)")
    st.plotly_chart(fig_sc,use_container_width=True)

    # Tabla percentiles
    st.markdown('<p class="section-title">📊 Tabla de Percentiles</p>', unsafe_allow_html=True)
    pcts=[5,10,25,50,75,90,95]
    df_pct=pd.DataFrame({
        "Percentil (%)":     pcts,
        "MP/Guardia (m)":    [round(float(np.percentile(mc["mp_s"],p)),2) for p in pcts],
        "Costo/m (USD)":     [round(float(np.percentile(mc["cm_s"],p)),4) for p in pcts],
        "VP Efectiva (m/h)":[round(float(np.percentile(mc["vp_s"],p)),3) for p in pcts],
        "TDC (USD/m)":       [round(float(np.percentile(mc["tdc_s"],p)),4) for p in pcts],
        "Vida Broca (m)":    [round(float(np.percentile(mc["vu_br_s"],p)),1) for p in pcts],
    })
    st.dataframe(df_pct,use_container_width=True,hide_index=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Media MP/g",    f"{np.mean(mc['mp_s']):.2f} m",   f"P90={np.percentile(mc['mp_s'],90):.2f}")
    c2.metric("Media Cm",      f"${np.mean(mc['cm_s']):.4f}",    f"P90={np.percentile(mc['cm_s'],90):.4f}")
    c3.metric("Media VP Ef.",  f"{np.mean(mc['vp_s']):.2f} m/h", f"σ={np.std(mc['vp_s']):.3f}")
    c4.metric("Media VU Broca",f"{np.mean(mc['vu_br_s']):.1f} m",f"P10={np.percentile(mc['vu_br_s'],10):.1f}")

# ════════════════════════════════════════════════════════════════════════
#  TAB 2 — SENSIBILIDAD
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">📈 Análisis de Sensibilidad ±30%</p>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Se varía cada parámetro ±30% manteniendo el resto constante.<br>
    Los exponentes de corrección son <b>reales y distintos por equipo</b>
    (mayor sensibilidad a UCS en DTH · mayor sensibilidad a RQD en Jack Leg).
    </div>""", unsafe_allow_html=True)

    PBASE=dict(cfg=cfg,vp_campo=vp_campo,dm=dm/100,u=u_eq/100,t_guardia=t_guard,t_pos=t_pos,
               c_eq=c_eq,c_mo=c_mo,c_mant=c_mant,c_energia=c_en,
               p_broca=p_broca,vu_broca=vu_broca,p_barra=p_barra,vu_barra=vu_barra,
               ucs=ucs,cai=cai,rqd=rqd)

    VARS=[
        ("VP Campo (m/h)",        "vp_campo",  vp_campo),
        ("Disponib. Mecánica",    "dm",        dm/100),
        ("Utilización",           "u",         u_eq/100),
        ("UCS (MPa)",             "ucs",       float(ucs)),
        ("CAI (Cerchar)",         "cai",       cai),
        ("RQD (%)",               "rqd",       float(rqd)),
        ("Costo Equipo",          "c_eq",      c_eq),
        ("Costo M.O.",            "c_mo",      c_mo),
        ("Precio Broca",          "p_broca",   p_broca),
        ("Vida Útil Broca",       "vu_broca",  vu_broca),
    ]
    pct_rng=np.linspace(-0.30,0.30,13)
    base_mp=res["mp_g"]; base_cm=res["cm"]

    fig_sp=go.Figure()
    for i,(lbl,key,base) in enumerate(VARS):
        ys=[]
        for pct in pct_rng:
            p=dict(PBASE); p[key]=base*(1+pct)
            try: ys.append(calcular_productividad(**p)["mp_g"])
            except: ys.append(base_mp)
        fig_sp.add_trace(go.Scatter(x=pct_rng*100,y=ys,mode="lines+markers",
                         name=lbl,line=dict(color=COLORS[i%len(COLORS)],width=2),
                         marker=dict(size=4)))
    fig_sp.add_hline(y=base_mp,line=dict(color="#fff",dash="dot",width=1.2))
    fig_sp.update_layout(**_layout(f"Sensibilidad MP/Guardia — {equipo_sel}"))
    fig_sp.update_xaxes(title_text="Variación (%)",ticksuffix="%")
    fig_sp.update_yaxes(title_text="Metros / Guardia")

    # Tornado
    imp=[]
    for lbl,key,base in VARS:
        p_lo=dict(PBASE); p_lo[key]=base*0.80
        p_hi=dict(PBASE); p_hi[key]=base*1.20
        try:
            lo_v=calcular_productividad(**p_lo)["mp_g"]
            hi_v=calcular_productividad(**p_hi)["mp_g"]
        except:
            lo_v=hi_v=base_mp
        imp.append((lbl,lo_v,hi_v,abs(hi_v-lo_v)))
    imp.sort(key=lambda x:x[3])
    lb=[x[0] for x in imp]; lo_t=[x[1] for x in imp]; hi_t=[x[2] for x in imp]

    fig_tor=go.Figure()
    fig_tor.add_trace(go.Bar(y=lb,x=[h-base_mp for h in hi_t],orientation="h",
                             name="+20%",marker_color=G,base=base_mp))
    fig_tor.add_trace(go.Bar(y=lb,x=[l-base_mp for l in lo_t],orientation="h",
                             name="-20%",marker_color=R,base=base_mp))
    fig_tor.add_vline(x=base_mp,line=dict(color="#fff",width=1.5,dash="dash"))
    fig_tor.update_layout(**_layout("Tornado — Impacto en MP/Guardia",h=480),barmode="overlay")
    fig_tor.update_xaxes(title_text="Metros / Guardia")

    c1,c2=st.columns(2)
    with c1: st.plotly_chart(fig_sp,use_container_width=True)
    with c2: st.plotly_chart(fig_tor,use_container_width=True)

    # Spider Costo/m
    st.markdown('<p class="section-title">📉 Sensibilidad — Costo por Metro</p>', unsafe_allow_html=True)
    fig_cm_s=go.Figure()
    for i,(lbl,key,base) in enumerate(VARS):
        ys=[]
        for pct in pct_rng:
            p=dict(PBASE); p[key]=base*(1+pct)
            try: ys.append(calcular_productividad(**p)["cm"])
            except: ys.append(base_cm)
        fig_cm_s.add_trace(go.Scatter(x=pct_rng*100,y=ys,mode="lines+markers",
                           name=lbl,line=dict(color=COLORS[i%len(COLORS)],width=2),
                           marker=dict(size=4)))
    fig_cm_s.add_hline(y=base_cm,line=dict(color="#fff",dash="dot",width=1.2))
    fig_cm_s.update_layout(**_layout("Sensibilidad — Costo/Metro (USD/m)"))
    fig_cm_s.update_xaxes(title_text="Variación (%)",ticksuffix="%")
    fig_cm_s.update_yaxes(title_text="USD/m")
    st.plotly_chart(fig_cm_s,use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
#  TAB 3 — COMPARACIÓN DE EQUIPOS
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">⚙️ Comparación Real de los 4 Equipos</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    <b>Misma condición de roca:</b> UCS={ucs} MPa · CAI={cai} · RQD={rqd}% · f={f_pr}<br>
    Cada equipo aplica sus propios factores de corrección reales (n_ucs, n_cai, n_rqd).
    Los resultados reflejan cómo cada tecnología responde diferente a la misma roca.
    </div>""", unsafe_allow_html=True)

    filas=[]
    for nombre,c in EQUIPOS.items():
        vu_br_c = calcular_vida_broca_real(c, ucs, cai)
        vu_ba_c = calcular_vida_barra_real(c, ucs, cai)
        rr=calcular_productividad(
            cfg=c,vp_campo=c["vp_ref"],dm=c["dm"]/100,u=c["u"]/100,
            t_guardia=t_guard,t_pos=c["t_pos"],
            c_eq=c["c_eq"],c_mo=c["c_mo"],c_mant=c["c_mant"],c_energia=c["c_energia"],
            p_broca=c["p_broca_ref"],vu_broca=vu_br_c,
            p_barra=c["p_barra"],vu_barra=vu_ba_c,
            ucs=ucs,cai=cai,rqd=rqd,
        )
        filas.append({**rr,"nombre":nombre,"color":c["color"],
                      "app":c["app"],"tipo":c["tipo"],"diam":c["diam_mm"],
                      "dm_c":c["dm"],"u_c":c["u"],"P_perc":c["P_perc_kW"],
                      "vu_br_real":vu_br_c,"icono":c["icono"]})

    nombres_e=[f["nombre"] for f in filas]
    mp_e=[f["mp_g"] for f in filas]; vp_e=[f["vp_ef"] for f in filas]
    cm_e=[f["cm"]   for f in filas]; td_e=[f["tdc"]   for f in filas]
    col_e=[f["color"] for f in filas]

    cc1,cc2=st.columns(2)
    for vals,tit,fmt,ylab,col_ in [
        (mp_e,"Metros / Guardia","{:.1f}","m/guardia",cc1),
        (vp_e,"VP Efectiva (m/h)","{:.2f}","m/h",cc2),
    ]:
        fig_b=go.Figure(go.Bar(x=nombres_e,y=vals,marker_color=col_e,
                               text=[fmt.format(v) for v in vals],
                               textposition="outside",textfont=dict(color="#fff")))
        fig_b.update_layout(**_layout(tit,h=380))
        fig_b.update_xaxes(title_text="Equipo"); fig_b.update_yaxes(title_text=ylab)
        with col_: st.plotly_chart(fig_b,use_container_width=True)

    cc3,cc4=st.columns(2)
    for vals,tit,fmt,ylab,col_ in [
        (cm_e,"Costo por Metro (USD/m)","${:.4f}","USD/m",cc3),
        (td_e,"TDC — Total Drilling Cost","${:.4f}","USD/m",cc4),
    ]:
        fig_b=go.Figure(go.Bar(x=nombres_e,y=vals,marker_color=col_e,
                               text=[fmt.format(v) for v in vals],
                               textposition="outside",textfont=dict(color="#fff")))
        fig_b.update_layout(**_layout(tit,h=380))
        fig_b.update_xaxes(title_text="Equipo"); fig_b.update_yaxes(title_text=ylab)
        with col_: st.plotly_chart(fig_b,use_container_width=True)

    # Scatter
    st.markdown('<p class="section-title">🎯 Posicionamiento Competitivo</p>', unsafe_allow_html=True)
    fig_sc2=go.Figure()
    for f in filas:
        fig_sc2.add_trace(go.Scatter(
            x=[f["cm"]],y=[f["mp_g"]],mode="markers+text",name=f["nombre"],
            text=[f"{f['icono']} {f['nombre']}"],textposition="top center",
            textfont=dict(color="#fff",size=11),
            marker=dict(color=f["color"],size=24,line=dict(color="#fff",width=2)),
        ))
    fig_sc2.update_layout(**_layout("Costo/m vs MP/Guardia — Posicionamiento competitivo",h=440),showlegend=False)
    fig_sc2.update_xaxes(title_text="Costo por Metro (USD/m)")
    fig_sc2.update_yaxes(title_text="Metros / Guardia")
    st.plotly_chart(fig_sc2,use_container_width=True)

    # Radar
    st.markdown('<p class="section-title">🕸️ Radar Comparativo (escala 0–10)</p>', unsafe_allow_html=True)
    cats=["MP/Guardia","VP Efectiva","Disponibilidad","Bajo Costo","Vida Broca"]
    mx_mp=max(mp_e); mx_vp=max(vp_e); mx_cm=max(cm_e)
    mx_vu=max(f["vu_br_real"] for f in filas)
    fig_rad=go.Figure()
    for f in filas:
        vals=[
            f["mp_g"]/mx_mp*10,
            f["vp_ef"]/mx_vp*10,
            f["dm_c"]/10,
            (1-f["cm"]/mx_cm)*10,
            f["vu_br_real"]/mx_vu*10,
        ]
        vals=vals+[vals[0]]
        fig_rad.add_trace(go.Scatterpolar(
            r=vals,theta=cats+[cats[0]],fill="toself",name=f["nombre"],
            line=dict(color=f["color"],width=2),
            fillcolor=hex2rgba(f["color"],0.18),
        ))
    fig_rad.update_layout(
        paper_bgcolor=DARK,font=dict(color=TEXT),height=500,
        margin=dict(l=50,r=50,t=60,b=50),
        title=dict(text="Radar Comparativo",font=dict(color="#fff",size=14),x=0.01),
        legend=dict(bgcolor=CARD,bordercolor=GRID,borderwidth=1),
        polar=dict(
            bgcolor=CARD,
            radialaxis=dict(visible=True,range=[0,10],gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID),
        ),
    )
    st.plotly_chart(fig_rad,use_container_width=True)

    # Tabla
    st.markdown('<p class="section-title">📋 Tabla Comparativa Completa</p>', unsafe_allow_html=True)
    df_comp=pd.DataFrame({
        "Equipo":            [f["icono"]+" "+f["nombre"] for f in filas],
        "Tipo":              [f["tipo"]    for f in filas],
        "Ø (mm)":            [f["diam"]   for f in filas],
        "P.Perc (kW)":       [f["P_perc"] for f in filas],
        "F_UCS":             [round(f["F_ucs"],3) for f in filas],
        "F_CAI":             [round(f["F_cai"],3) for f in filas],
        "VP Real (m/h)":     [round(f["vp_real"],2) for f in filas],
        "VP Ef. (m/h)":      [round(f["vp_ef"],3)  for f in filas],
        "MP/Guardia (m)":    [round(f["mp_g"],2)   for f in filas],
        "Costo/m (USD)":     [round(f["cm"],4)     for f in filas],
        "TDC (USD/m)":       [round(f["tdc"],4)    for f in filas],
        "VU Broca real (m)": [f["vu_br_real"]      for f in filas],
        "Aplicación":        [f["app"]              for f in filas],
    })
    st.dataframe(df_comp,use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════════
#  TAB 4 — COSTOS
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">💰 Desglose de Costos por Metro Perforado</p>', unsafe_allow_html=True)
    vp_ef_=res["vp_ef"]
    comp_lbl=["Equipo","Mano de Obra","Mantenimiento","Energía","Broca","Barra"]
    comp_v=[
        round(c_eq/max(vp_ef_,0.001),4),  round(c_mo/max(vp_ef_,0.001),4),
        round(c_mant/max(vp_ef_,0.001),4),round(c_en/max(vp_ef_,0.001),4),
        round(res["c_br_m"],4),            round(res["c_ba_m"],4),
    ]
    comp_col=[R,G,Y,P,B,"#e67e22"]
    total_=sum(comp_v)

    cc1,cc2=st.columns(2)
    with cc1:
        fig_pie=go.Figure(go.Pie(
            labels=comp_lbl,values=comp_v,hole=0.46,
            marker=dict(colors=comp_col,line=dict(color=DARK,width=2)),
            textinfo="label+percent",textfont=dict(size=11),
        ))
        fig_pie.add_annotation(x=0.5,y=0.5,showarrow=False,
                               text=f"Total<br><b>${total_:.4f}</b><br>/m",
                               font=dict(size=13,color="#fff"))
        fig_pie.update_layout(paper_bgcolor=DARK,font=dict(color=TEXT),height=420,
                              margin=dict(l=20,r=20,t=50,b=20),
                              title=dict(text="Estructura de Costos (USD/m)",
                              font=dict(color="#fff",size=14),x=0.01),
                              legend=dict(bgcolor=CARD,bordercolor=GRID,borderwidth=1))
        st.plotly_chart(fig_pie,use_container_width=True)
    with cc2:
        fig_wf=go.Figure(go.Waterfall(
            orientation="v",measure=["relative"]*len(comp_v)+["total"],
            x=comp_lbl+["TOTAL"],y=comp_v+[None],
            connector=dict(line=dict(color=GRID,width=1)),
            increasing=dict(marker=dict(color=R)),
            totals=dict(marker=dict(color=G)),
            text=[f"${v:.4f}" for v in comp_v]+[f"${total_:.4f}"],
            textposition="outside",textfont=dict(color="#fff",size=10),
        ))
        fig_wf.update_layout(**_layout("Cascada de Costos — USD/m",h=420))
        fig_wf.update_yaxes(title_text="USD/m")
        st.plotly_chart(fig_wf,use_container_width=True)

    # Curva costo acumulado
    st.markdown('<p class="section-title">📈 Costo Total de Perforación vs Metros/Mes</p>', unsafe_allow_html=True)
    metros_mes=np.arange(100,6001,100)
    fig_cu=go.Figure()
    for nombre_e,c in EQUIPOS.items():
        vu_br_c=calcular_vida_broca_real(c,ucs,cai)
        vu_ba_c=calcular_vida_barra_real(c,ucs,cai)
        rr=calcular_productividad(
            cfg=c,vp_campo=c["vp_ref"],dm=c["dm"]/100,u=c["u"]/100,
            t_guardia=t_guard,t_pos=c["t_pos"],
            c_eq=c["c_eq"],c_mo=c["c_mo"],c_mant=c["c_mant"],c_energia=c["c_energia"],
            p_broca=c["p_broca_ref"],vu_broca=vu_br_c,
            p_barra=c["p_barra"],vu_barra=vu_ba_c,
            ucs=ucs,cai=cai,rqd=rqd,
        )
        fig_cu.add_trace(go.Scatter(x=metros_mes,y=metros_mes*rr["c_dir"],mode="lines",
                         name=nombre_e,line=dict(color=c["color"],width=2.5)))
    fig_cu.update_layout(**_layout("Costo Acumulado Directo de Perforación vs Metros/Mes",h=400))
    fig_cu.update_xaxes(title_text="Metros Perforados / Mes")
    fig_cu.update_yaxes(title_text="Costo Total (USD)")
    st.plotly_chart(fig_cu,use_container_width=True)

    st.markdown('<p class="section-title">📊 Tabla de Costos Detallada</p>', unsafe_allow_html=True)
    df_cost=pd.DataFrame({
        "Componente":          comp_lbl,
        "Costo Horario ($/h)": [str(c_eq),str(c_mo),str(c_mant),str(c_en),"—","—"],
        "Costo/m (USD)":       comp_v,
        "% del Total":         [round(v/max(total_,0.0001)*100,1) for v in comp_v],
    })
    st.dataframe(df_cost,use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════════
#  TAB 5 — VP VS ROCA
# ════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">📉 VP vs Resistencia de Roca (UCS)</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    Modelo empírico Bauer-Calder calibrado por tipo de equipo.<br>
    <b>Exponentes reales:</b> Jumbo n={EQUIPOS['Jumbo Hidráulico (Boomer)']["n_ucs"]} ·
    Jack Leg n={EQUIPOS['Jack Leg (Neumático)']["n_ucs"]} ·
    DTH n={EQUIPOS['DTH (Martillo en Fondo)']["n_ucs"]} ·
    Simba n={EQUIPOS['Simba (Perforadora Radial)']["n_ucs"]}<br>
    El DTH tiene el mayor exponente — se degrada más rápido en roca muy dura pero es eficiente en un rango mayor.
    </div>""", unsafe_allow_html=True)

    ucs_x=np.linspace(20,300,200)
    fig_vpr=go.Figure()
    for nombre_e,c in EQUIPOS.items():
        # VP usando el modelo real de cada equipo
        F_u = (c["ucs_ref"]/np.maximum(ucs_x,10))**c["n_ucs"]
        vp_cur=np.clip(c["vp_ref"]*F_u, c["vp_min"], c["vp_max"])
        fig_vpr.add_trace(go.Scatter(x=ucs_x,y=vp_cur,mode="lines",name=nombre_e,
                          line=dict(color=c["color"],width=2.5)))
        if nombre_e==equipo_sel:
            fig_vpr.add_trace(go.Scatter(
                x=np.concatenate([ucs_x,ucs_x[::-1]]),
                y=np.concatenate([vp_cur*1.15,(vp_cur*0.85)[::-1]]),
                fill="toself",fillcolor=hex2rgba(c["color"],0.13),
                line=dict(color="rgba(0,0,0,0)"),showlegend=False,
            ))
            vp_pt=float(np.clip(c["vp_ref"]*(c["ucs_ref"]/max(ucs,10))**c["n_ucs"],
                                c["vp_min"],c["vp_max"]))
            fig_vpr.add_trace(go.Scatter(
                x=[ucs],y=[vp_pt],mode="markers+text",name="★ Actual",
                text=["★ Actual"],textposition="top right",textfont=dict(color="#fff"),
                marker=dict(color=R,size=14,symbol="star",line=dict(color="#fff",width=2)),
            ))
    fig_vpr.update_layout(**_layout("VP vs UCS por Tipo de Equipo — Modelos Empíricos Calibrados",h=480))
    fig_vpr.update_xaxes(title_text="UCS — Resistencia Compresión Uniaxial (MPa)")
    fig_vpr.update_yaxes(title_text="Velocidad de Penetración (m/h)")
    st.plotly_chart(fig_vpr,use_container_width=True)

    # VP vs CAI
    st.markdown('<p class="section-title">🔬 VP vs Índice Cerchar (CAI) — Efecto Abrasividad</p>', unsafe_allow_html=True)
    cai_x=np.linspace(0.1,5.0,100)
    fig_cai=go.Figure()
    for nombre_e,c in EQUIPOS.items():
        F_c=(c["cai_ref"]/np.maximum(cai_x,0.1))**c["n_cai"]
        vp_cai=np.clip(c["vp_ref"]*F_c,c["vp_min"],c["vp_max"])
        fig_cai.add_trace(go.Scatter(x=cai_x,y=vp_cai,mode="lines",name=nombre_e,
                          line=dict(color=c["color"],width=2.5)))
    fig_cai.add_vline(x=cai,line=dict(color="#fff",dash="dash",width=1.5),
                       annotation=dict(text=f"CAI actual={cai}",font=dict(color="#fff",size=11)))
    fig_cai.update_layout(**_layout("VP vs CAI — Índice Cerchar de Abrasividad",h=400))
    fig_cai.update_xaxes(title_text="CAI"); fig_cai.update_yaxes(title_text="VP (m/h)")
    st.plotly_chart(fig_cai,use_container_width=True)

    # Vida de broca vs UCS y CAI
    st.markdown('<p class="section-title">🔩 Vida Útil de Broca vs Condición de Roca</p>', unsafe_allow_html=True)
    cc1,cc2=st.columns(2)
    with cc1:
        fig_vu=go.Figure()
        for nombre_e,c in EQUIPOS.items():
            vu_x=[calcular_vida_broca_real(c,u_val,cai) for u_val in ucs_x]
            fig_vu.add_trace(go.Scatter(x=ucs_x,y=vu_x,mode="lines",name=nombre_e,
                             line=dict(color=c["color"],width=2.5)))
        fig_vu.add_vline(x=ucs,line=dict(color="#fff",dash="dash",width=1.2),
                          annotation=dict(text=f"UCS={ucs}",font=dict(color="#fff",size=10)))
        fig_vu.update_layout(**_layout("Vida Útil Broca vs UCS (CAI fijo)",h=400))
        fig_vu.update_xaxes(title_text="UCS (MPa)"); fig_vu.update_yaxes(title_text="Vida Broca (m)")
        st.plotly_chart(fig_vu,use_container_width=True)
    with cc2:
        fig_vu2=go.Figure()
        for nombre_e,c in EQUIPOS.items():
            vu_c=[calcular_vida_broca_real(c,ucs,c_val) for c_val in cai_x]
            fig_vu2.add_trace(go.Scatter(x=cai_x,y=vu_c,mode="lines",name=nombre_e,
                              line=dict(color=c["color"],width=2.5)))
        fig_vu2.add_vline(x=cai,line=dict(color="#fff",dash="dash",width=1.2),
                           annotation=dict(text=f"CAI={cai}",font=dict(color="#fff",size=10)))
        fig_vu2.update_layout(**_layout("Vida Útil Broca vs CAI (UCS fijo)",h=400))
        fig_vu2.update_xaxes(title_text="CAI"); fig_vu2.update_yaxes(title_text="Vida Broca (m)")
        st.plotly_chart(fig_vu2,use_container_width=True)

    # Superficie 3D
    st.markdown('<p class="section-title">🌐 Superficie 3D — VP vs UCS y CAI</p>', unsafe_allow_html=True)
    ucs_3=np.linspace(20,250,40); cai_3=np.linspace(0.1,5.0,40)
    UCS3,CAI3=np.meshgrid(ucs_3,cai_3)
    F_u3=(cfg["ucs_ref"]/np.maximum(UCS3,10))**cfg["n_ucs"]
    F_c3=(cfg["cai_ref"]/np.maximum(CAI3,0.1))**cfg["n_cai"]
    VP3=np.clip(cfg["vp_ref"]*F_u3*F_c3,cfg["vp_min"],cfg["vp_max"])
    fig_3d=go.Figure()
    fig_3d.add_trace(go.Surface(z=VP3,x=UCS3,y=CAI3,colorscale="RdYlGn",
                                colorbar=dict(title="VP m/h",thickness=12)))
    fig_3d.add_trace(go.Scatter3d(x=[ucs],y=[cai],z=[res["vp_real"]],mode="markers",name="Actual",
                                   marker=dict(color=R,size=7,line=dict(color="#fff",width=1))))
    fig_3d.update_layout(
        paper_bgcolor=DARK,font=dict(color=TEXT),height=520,
        margin=dict(l=0,r=0,t=40,b=0),
        title=dict(text=f"VP Real — {equipo_sel} (n_ucs={cfg['n_ucs']}, n_cai={cfg['n_cai']})",
                   font=dict(color="#fff",size=13),x=0.01),
        legend=dict(bgcolor=CARD,bordercolor=GRID,borderwidth=1),
        scene=dict(
            xaxis=dict(title="UCS (MPa)",backgroundcolor=DARK,gridcolor=GRID),
            yaxis=dict(title="CAI",backgroundcolor=DARK,gridcolor=GRID),
            zaxis=dict(title="VP (m/h)",backgroundcolor=DARK,gridcolor=GRID),
        ),
    )
    st.plotly_chart(fig_3d,use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
#  TAB 6 — MEMORIA DE CÁLCULO
# ════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">📋 Memoria de Cálculo Completa</p>', unsafe_allow_html=True)
    F_prod=res["F_ucs"]*res["F_cai"]*res["F_rqd"]
    tipo_avance="Avance por disparo (desarrollo)" if cfg["avance_tipo"]=="desarrollo" else "Metros lineales (producción)"

    st.markdown(f"""
## {cfg['icono']} Memoria: {equipo_sel}
**Modelo de referencia:** {cfg['modelo_ref']}  
**Tipo de avance:** {tipo_avance}

---
### 1. DATOS DE ENTRADA

| Parámetro | Símbolo | Valor | Unidad |
|---|---|---|---|
| VP medida en campo | VP_campo | {vp_campo} | m/h |
| UCS — Resistencia Compresión | UCS | {ucs} | MPa |
| CAI — Índice Cerchar | CAI | {cai} | — |
| RQD | RQD | {rqd} | % |
| f Protodyakonov | f | {f_pr} | adim |
| Disponibilidad Mecánica | DM | {dm} | % |
| Utilización del Equipo | U | {u_eq} | % |
| Duración de Guardia | TG | {t_guard} | h |
| Tiempo Posicionamiento | Tp | {t_pos} | min/taladro |
| N° de Brazos / Perforadores | n | {cfg['n_brazos']} | — |
| Diámetro de broca | Ø | {cfg['diam_mm']} | mm |
| Longitud de barra | L | {cfg['long_barra_m']} | m |
| Potencia de percusión | P_perc | {cfg['P_perc_kW']} | kW |

---
### 2. CORRECCIONES POR CONDICIÓN DE ROCA
*Modelo Bauer-Calder (1967) calibrado por tipo de equipo*

| Factor | Fórmula | Valor |
|---|---|---|
| F_UCS | (UCS_ref/UCS)^n = ({cfg['ucs_ref']}/{ucs})^{cfg['n_ucs']} | **{res['F_ucs']:.4f}** |
| F_CAI | (CAI_ref/CAI)^n = ({cfg['cai_ref']}/{cai})^{cfg['n_cai']} | **{res['F_cai']:.4f}** |
| F_RQD | 1 + n×(RQD-75)/100 = 1+{cfg['n_rqd']}×({rqd}-75)/100 | **{res['F_rqd']:.4f}** |
| **F_total** | F_UCS × F_CAI × F_RQD | **{F_prod:.4f}** |

---
### 3. VELOCIDAD DE PENETRACIÓN REAL
```
VP_real = VP_campo × F_UCS × F_CAI × F_RQD
VP_real = {vp_campo} × {res['F_ucs']} × {res['F_cai']} × {res['F_rqd']}
VP_real = {res['vp_real']} m/h
```

---
### 4. PARÁMETROS DE PRODUCCIÓN

```
VP_efectiva = VP_real × U × DM
VP_ef = {res['vp_real']} × {u_eq/100} × {dm/100} = {res['vp_ef']} m/h

T_efectivo = TG × U × DM
T_ef = {t_guard} × {u_eq/100} × {dm/100} = {res['t_ef']} h/guardia
```""")

    if cfg["avance_tipo"]=="desarrollo":
        L=cfg["long_barra_m"]; Ef=cfg["eficiencia_voladura"]
        t_tal=L/max(res["vp_ef"],0.001)+t_pos/60
        n_tal=res["t_ef"]/max(t_tal,0.001)
        st.markdown(f"""
```
[MODO DESARROLLO — avance por taladro]
T_taladro = L_barra/VP_ef + Tp
T_taladro = {L}/{res['vp_ef']} + {t_pos}/60 = {t_tal:.4f} h/taladro

N_taladros = T_ef / T_taladro
N_taladros = {res['t_ef']} / {t_tal:.4f} = {n_tal:.1f} taladros/guardia

MP/guardia = N_tal × L × Eficiencia × N_brazos
MP/g = {n_tal:.1f} × {L} × {Ef} × {cfg['n_brazos']} = {res['mp_g']:.2f} m/guardia
```""")
    else:
        st.markdown(f"""
```
[MODO PRODUCCIÓN — metros lineales]
MP/guardia = VP_ef × T_ef × N_brazos
MP/g = {res['vp_ef']} × {res['t_ef']} × {cfg['n_brazos']} = {res['mp_g']:.2f} m/guardia
```""")

    st.markdown(f"""
---
### 5. VIDA ÚTIL DE ACEROS — Modelo Tandanand (1973)
```
VU_broca = VU_ref / ((CAI/CAI_ref)^k_b × (UCS/UCS_ref)^k_u)
VU_broca = {cfg['vu_broca_ref']} / (({cai}/{cfg['cai_ref']})^{cfg['k_broca']} × ({ucs}/{cfg['ucs_ref']})^{cfg['k_ucs_b']})
VU_broca = {vu_broca} m  (ingresado/ajustado)

C_broca/m = P_broca / VU_broca = {p_broca} / {vu_broca} = {res['c_br_m']} USD/m
C_barra/m = P_barra / VU_barra = {p_barra} / {vu_barra} = {res['c_ba_m']} USD/m
```

---
### 6. COSTOS DE PERFORACIÓN
```
Ch (horario) = Ceq + Cmo + Cmant + Cenergia
Ch = {c_eq} + {c_mo} + {c_mant} + {c_en} = {res['ch']} USD/h

Cm (por metro) = Ch / VP_ef = {res['ch']} / {res['vp_ef']} = {res['cm']} USD/m

TDC = C_broca/m + Cm = {res['c_br_m']} + {res['cm']} = {res['tdc']} USD/m

C_directo = Cm + C_aceros = {res['cm']} + {res['c_ac_m']} = {res['c_dir']} USD/m
```

---
### 7. PRODUCCIÓN ESTIMADA
| Período | Metros Perforados | Costo Directo |
|---|---|---|
| Por guardia | {res['mp_g']:.1f} m | ${res['mp_g']*res['c_dir']:,.2f} |
| Por día (3 guardias) | {res['mp_g']*3:.1f} m | ${res['mp_g']*3*res['c_dir']:,.2f} |
| Por mes (26 días) | {res['mp_mes']:,.1f} m | ${res['costo_mes']:,.2f} |

---
### 8. RESULTADOS MONTE CARLO ({n_sim:,} iteraciones)
| Variable | P10 | P50 | P90 | Media | Desv.Est |
|---|---|---|---|---|---|
| VP Ef. (m/h) | {np.percentile(mc['vp_s'],10):.3f} | {np.percentile(mc['vp_s'],50):.3f} | {np.percentile(mc['vp_s'],90):.3f} | {np.mean(mc['vp_s']):.3f} | {np.std(mc['vp_s']):.3f} |
| MP/Guardia (m) | {np.percentile(mc['mp_s'],10):.2f} | {np.percentile(mc['mp_s'],50):.2f} | {np.percentile(mc['mp_s'],90):.2f} | {np.mean(mc['mp_s']):.2f} | {np.std(mc['mp_s']):.2f} |
| Costo/m (USD) | {np.percentile(mc['cm_s'],10):.4f} | {np.percentile(mc['cm_s'],50):.4f} | {np.percentile(mc['cm_s'],90):.4f} | {np.mean(mc['cm_s']):.4f} | {np.std(mc['cm_s']):.4f} |
| TDC (USD/m) | {np.percentile(mc['tdc_s'],10):.4f} | {np.percentile(mc['tdc_s'],50):.4f} | {np.percentile(mc['tdc_s'],90):.4f} | {np.mean(mc['tdc_s']):.4f} | {np.std(mc['tdc_s']):.4f} |
| VU Broca (m) | {np.percentile(mc['vu_br_s'],10):.1f} | {np.percentile(mc['vu_br_s'],50):.1f} | {np.percentile(mc['vu_br_s'],90):.1f} | {np.mean(mc['vu_br_s']):.1f} | {np.std(mc['vu_br_s']):.1f} |

---
### 9. ESPECIFICACIONES TÉCNICAS — {equipo_sel}
| Especificación | Valor |
|---|---|
| Tipo de percusión | {cfg['tipo']} |
| N° de brazos | {cfg['n_brazos']} |
| Diámetro de perforación | {cfg['diam_mm']} mm |
| Longitud de barra | {cfg['long_barra_m']} m |
| Potencia percusión | {cfg['P_perc_kW']} kW |
| Frecuencia de golpes | {cfg['freq_golpes']} Hz |
| Presión de trabajo | {cfg['presion_bar']} bar |
| Velocidad de rotación | {cfg['rot_rpm']} RPM |
| Fuerza de empuje | {cfg['empuje_kN']} kN |
| Aplicación principal | {cfg['app']} |
| Eficiencia de voladura | {cfg['eficiencia_voladura']*100:.0f}% |

---
### 10. BIBLIOGRAFÍA
- **Bauer & Calder (1967)** — *Open Pit Blast Design. Analysis for Blast Control*
- **Paone & Madson (1966)** — *Drillability Studies: Impregnated Diamond Bits*
- **Tandanand (1973)** — *Principles of Drilling. SME Mining Engineering Handbook*
- **Hustrulid & Bullock (1999)** — *Open Pit Mine Planning & Design*
- **Jimeno, Jimeno & Carcedo (1995)** — *Drilling and Blasting of Rocks*
- **Camac Torres, A. (2009)** — *Perforación y Voladura de Rocas en Minería*
- **Atlas Copco / Epiroc (2022)** — *Rock Drilling Tools — Technical Reference*
""")

    df_exp=pd.DataFrame({
        "Parámetro":[
            "Equipo","Modelo ref.","Tipo percusión",
            "VP campo (m/h)","UCS (MPa)","CAI","RQD (%)","f Protodyakonov",
            "F_UCS","F_CAI","F_RQD","VP Real (m/h)","VP Efectiva (m/h)",
            "DM (%)","U (%)","T.Guardia (h)","T.Efectivo (h)","MP/Guardia (m)",
            "Ch (USD/h)","Cm (USD/m)","TDC (USD/m)","C.Directo (USD/m)",
            "VU Broca real (m)","C.Broca/m (USD)","C.Barra/m (USD)",
            "Prod.Mes (m)","Costo Mes (USD)",
            "MC P10 MP/g","MC P50 MP/g","MC P90 MP/g",
            "MC P10 Cm","MC P50 Cm","MC P90 Cm",
        ],
        "Valor":[
            equipo_sel, cfg["modelo_ref"], cfg["tipo"],
            vp_campo,ucs,cai,rqd,f_pr,
            res["F_ucs"],res["F_cai"],res["F_rqd"],res["vp_real"],res["vp_ef"],
            dm,u_eq,t_guard,res["t_ef"],res["mp_g"],
            res["ch"],res["cm"],res["tdc"],res["c_dir"],
            vu_broca,res["c_br_m"],res["c_ba_m"],
            res["mp_mes"],res["costo_mes"],
            round(float(np.percentile(mc["mp_s"],10)),2),
            round(float(np.percentile(mc["mp_s"],50)),2),
            round(float(np.percentile(mc["mp_s"],90)),2),
            round(float(np.percentile(mc["cm_s"],10)),4),
            round(float(np.percentile(mc["cm_s"],50)),4),
            round(float(np.percentile(mc["cm_s"],90)),4),
        ],
    })
    st.download_button(
        label="📥 Descargar Memoria de Cálculo (CSV)",
        data=df_exp.to_csv(index=False,encoding="utf-8-sig"),
        file_name=f"memoria_{equipo_sel[:15].replace(' ','_')}.csv",
        mime="text/csv",use_container_width=True,
    )

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6b7faa;font-size:0.78rem;padding:0.3rem;">
⛏️ Productividad de Perforación Minera v3.0 &nbsp;|&nbsp;
Python · Streamlit · Plotly · Monte Carlo &nbsp;|&nbsp;
Ing. de Minas — Universidad Nacional del Altiplano Puno
</div>
""", unsafe_allow_html=True)
