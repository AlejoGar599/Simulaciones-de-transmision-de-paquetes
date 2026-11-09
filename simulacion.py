#!/usr/bin/env python3
"""
Optimización del tamaño de paquetes de información para maximizar la
eficiencia de transmisión en redes (modelo simplificado IPv4/TCP).

Modelo:
    P(S) = 1 - exp(-lambda * S)            probabilidad de corrupción
    E(S) = S/(S+H) * (1 - P(S))            eficiencia
         = S/(S+H) * exp(-lambda * S)

Condición de óptimo (dE/dS = 0):
    H / (S (S+H)) = lambda   ->   S* = ( -H + sqrt(H^2 + 4H/lambda) ) / 2

Simulaciones que reproduce este script:
  1. Barrido de E(S) y P(S) para el escenario base + tabla CSV.
  2. Máximo analítico vs. máximo numérico (SciPy) + 2ª derivada.
  3. Comparación de tres escenarios de red (p_b = 1e-7, 1e-6, 1e-5).
  4. Sensibilidad de S* respecto a H y lambda.
  5. Propagación de incertidumbres (analítica y Monte Carlo) para S*.
  6. Simulación estocástica de retransmisiones (Monte Carlo) con barras de error.

Uso:   python simulacion.py
Salida: carpeta ./resultados  (CSV + gráficas PNG)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ----------------------------------------------------------------------
# 0. PARÁMETROS
# ----------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

H = 40.0                    # overhead [bytes]: IPv4 (20) + TCP (20)
P_BIT_BASE = 1e-6           # prob. de error por bit (escenario base)
LAM_BASE = 8 * P_BIT_BASE   # lambda = 8 * p_b  [bytes^-1]

MTU = 1500                  # [bytes] Ethernet
MSS = MTU - 20 - 20         # 1460 bytes de datos útiles máximos por segmento

S_MIN, S_MAX, S_STEP = 10, 5000, 10   # barrido ampliado para ver el óptimo real
S = np.arange(S_MIN, S_MAX + 1, S_STEP, dtype=float)

ESCENARIOS = {                        # p_b por bit
    "Baja tasa de error (p_b=1e-7)": 1e-7,
    "Tasa moderada (p_b=1e-6)": 1e-6,
    "Alta tasa de error (p_b=1e-5)": 1e-5,
}

# Incertidumbres de los parámetros (VALORES SUPUESTOS: justifíquelos en el informe)
SIGMA_H = 4.0               # bytes (p. ej. ~10 % por opciones TCP/IP)
SIGMA_LAM_REL = 0.10        # 10 % de incertidumbre relativa en lambda

N_MC_UNC = 100_000          # muestras para propagación Monte Carlo
N_PAQUETES_SIM = 20_000     # paquetes por tamaño en la simulación de retransmisión
N_LOTES = 20                # lotes para estimar el error estadístico

OUT = "resultados"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------
# 1. FUNCIONES DEL MODELO
# ----------------------------------------------------------------------
def prob_error(S, lam):
    return 1.0 - np.exp(-lam * S)


def eficiencia(S, H, lam):
    return S / (S + H) * np.exp(-lam * S)


def dE_dS(S, H, lam):
    """Primera derivada: E' = E * [ H/(S(S+H)) - lambda ]"""
    return eficiencia(S, H, lam) * (H / (S * (S + H)) - lam)


def d2E_dS2(S, H, lam):
    """Segunda derivada de E = f*g con f=S/(S+H), g=exp(-lam S)."""
    f = S / (S + H)
    f1 = H / (S + H) ** 2
    f2 = -2 * H / (S + H) ** 3
    g = np.exp(-lam * S)
    return f2 * g - 2 * lam * f1 * g + lam ** 2 * f * g


def S_optimo(H, lam):
    return (-H + np.sqrt(H ** 2 + 4 * H / lam)) / 2


def S_optimo_numerico(H, lam, cota_sup=1e6):
    res = minimize_scalar(lambda s: -eficiencia(s, H, lam),
                          bounds=(1.0, cota_sup), method="bounded",
                          options={"xatol": 1e-9})
    return res.x


def dSopt_dH(H, lam):
    R = np.sqrt(H ** 2 + 4 * H / lam)
    return 0.5 * (-1 + (H + 2 / lam) / R)


def dSopt_dlam(H, lam):
    R = np.sqrt(H ** 2 + 4 * H / lam)
    return -H / (lam ** 2 * R)


# ----------------------------------------------------------------------
# 2. BARRIDO DEL ESCENARIO BASE
# ----------------------------------------------------------------------
def simulacion_barrido():
    P = prob_error(S, LAM_BASE)
    E = eficiencia(S, H, LAM_BASE)
    df = pd.DataFrame({"S [bytes]": S, "P(S)": P, "E(S)": E})
    df.to_csv(f"{OUT}/tabla_barrido_base.csv", index=False)

    # Tabla resumida para el informe
    tamanos = [100, 500, 1000, 1460, 1500, 2000, 3000, 5000]
    resumen = df[df["S [bytes]"].isin(tamanos)]
    resumen.to_csv(f"{OUT}/tabla_resumen_informe.csv", index=False)
    print("\n[1] Tabla resumen (escenario base)")
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return df


# ----------------------------------------------------------------------
# 3. ANALÍTICO vs NUMÉRICO
# ----------------------------------------------------------------------
def simulacion_analitico_vs_numerico():
    S_an = S_optimo(H, LAM_BASE)
    S_num = S_optimo_numerico(H, LAM_BASE)
    E_max = eficiencia(S_an, H, LAM_BASE)
    seg = d2E_dS2(S_an, H, LAM_BASE)
    comprobacion = dE_dS(S_an, H, LAM_BASE)

    tabla = pd.DataFrame({
        "Magnitud": ["S* analítico [B]", "S* numérico (SciPy) [B]",
                     "|diferencia| [B]", "E(S*)", "E'(S*) (debe ser ~0)",
                     "E''(S*) (debe ser <0)", "S* dentro del MSS (1460 B)?"],
        "Valor": [S_an, S_num, abs(S_an - S_num), E_max, comprobacion, seg,
                  "Sí" if S_an <= MSS else "No"],
    })
    tabla.to_csv(f"{OUT}/tabla_analitico_vs_numerico.csv", index=False)
    print("\n[2] Analítico vs numérico (escenario base)")
    print(tabla.to_string(index=False))

    # Óptimo restringido al límite físico (MSS)
    E_mss = eficiencia(MSS, H, LAM_BASE)
    print(f"\n    Con restricción S <= MSS={MSS} B: S*_restr = {MSS} B, "
          f"E = {E_mss:.6f}")

    # Gráfica 1: E(S) con el máximo
    E = eficiencia(S, H, LAM_BASE)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(S, E, label="E(S)")
    ax.plot(S_an, E_max, "ro", label=f"Máximo: S*={S_an:.0f} B, E={E_max:.4f}")
    ax.axvline(S_an, color="r", ls="--", lw=0.8)
    ax.axvline(MSS, color="gray", ls=":", label=f"MSS = {MSS} B")
    ax.set_xlabel("Tamaño del paquete S [bytes]")
    ax.set_ylabel("Eficiencia E(S) [adimensional]")
    ax.set_title("Eficiencia vs tamaño del paquete")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica1_eficiencia.png", dpi=200)
    plt.close(fig)

    # Gráfica 2: P(S)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(S, prob_error(S, LAM_BASE))
    ax.set_xlabel("Tamaño del paquete S [bytes]")
    ax.set_ylabel("Probabilidad de corrupción P(S)")
    ax.set_title("Probabilidad de error vs tamaño del paquete")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica2_probabilidad_error.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. COMPARACIÓN DE ESCENARIOS
# ----------------------------------------------------------------------
def simulacion_escenarios():
    filas = []
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for nombre, pb in ESCENARIOS.items():
        lam = 8 * pb
        s_an = S_optimo(H, lam)
        s_num = S_optimo_numerico(H, lam)
        e_max = eficiencia(s_an, H, lam)
        s_restr = min(s_an, MSS)
        filas.append({
            "Escenario": nombre, "p_b": pb, "lambda [1/B]": lam,
            "S* analítico [B]": s_an, "S* numérico [B]": s_num,
            "E(S*)": e_max,
            "S* restringido a MSS [B]": s_restr,
            "E(S* restr.)": eficiencia(s_restr, H, lam),
        })
        smax = max(S_MAX, 1.3 * s_an) if s_an < 20000 else S_MAX
        Sx = np.linspace(10, smax, 2000)
        l, = ax.plot(Sx, eficiencia(Sx, H, lam), label=nombre)
        ax.plot(s_an, e_max, "o", color=l.get_color())
    ax.axvline(MSS, color="gray", ls=":", label=f"MSS = {MSS} B")
    ax.set_xlabel("Tamaño del paquete S [bytes]")
    ax.set_ylabel("Eficiencia E(S) [adimensional]")
    ax.set_title("Comparación de escenarios de red")
    ax.set_xlim(0, S_MAX); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica3_escenarios.png", dpi=200)
    plt.close(fig)

    df = pd.DataFrame(filas)
    df.to_csv(f"{OUT}/tabla_escenarios.csv", index=False)
    print("\n[3] Escenarios")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


# ----------------------------------------------------------------------
# 5. SENSIBILIDAD
# ----------------------------------------------------------------------
def simulacion_sensibilidad():
    # S* vs lambda (log-log)
    lams = np.logspace(-7, -3.5, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(lams, S_optimo(H, lams))
    ax.axhline(MSS, color="gray", ls=":", label=f"MSS = {MSS} B")
    ax.set_xlabel(r"Parámetro de error $\lambda$ [bytes$^{-1}$]")
    ax.set_ylabel("Tamaño óptimo S* [bytes]")
    ax.set_title(r"S* en función de $\lambda$ (H = 40 B)")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica4_Sopt_vs_lambda.png", dpi=200)
    plt.close(fig)

    # S* vs H para tres lambdas
    Hs = np.linspace(10, 200, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nombre, pb in ESCENARIOS.items():
        ax.plot(Hs, S_optimo(Hs, 8 * pb), label=nombre)
    ax.set_xlabel("Overhead H [bytes]")
    ax.set_ylabel("Tamaño óptimo S* [bytes]")
    ax.set_title("S* en función del overhead")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica5_Sopt_vs_H.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# 6. PROPAGACIÓN DE INCERTIDUMBRES
# ----------------------------------------------------------------------
def simulacion_incertidumbre():
    filas = []
    print("\n[5] Propagación de incertidumbres "
          f"(sigma_H={SIGMA_H} B, sigma_lambda={SIGMA_LAM_REL*100:.0f} %)")
    for nombre, pb in ESCENARIOS.items():
        lam = 8 * pb
        sig_lam = SIGMA_LAM_REL * lam
        s0 = S_optimo(H, lam)

        # (a) Analítica: sigma^2 = (dS/dH)^2 sH^2 + (dS/dlam)^2 slam^2
        sig_an = np.sqrt((dSopt_dH(H, lam) * SIGMA_H) ** 2 +
                         (dSopt_dlam(H, lam) * sig_lam) ** 2)

        # (b) Monte Carlo
        Hm = rng.normal(H, SIGMA_H, N_MC_UNC)
        lm = rng.normal(lam, sig_lam, N_MC_UNC)
        mask = (Hm > 0) & (lm > 0)
        sm = S_optimo(Hm[mask], lm[mask])

        # Eficiencia máxima y su incertidumbre (MC)
        em = eficiencia(sm, Hm[mask], lm[mask])

        filas.append({
            "Escenario": nombre,
            "S* [B]": s0,
            "sigma analítica [B]": sig_an,
            "sigma Monte Carlo [B]": sm.std(ddof=1),
            "E(S*) media": em.mean(),
            "sigma E(S*)": em.std(ddof=1),
        })
        print(f"    {nombre}: S* = ({s0:.0f} ± {sig_an:.0f}) B  "
              f"[MC: ± {sm.std(ddof=1):.0f}]   "
              f"E = {em.mean():.4f} ± {em.std(ddof=1):.4f}")
    df = pd.DataFrame(filas)
    df.to_csv(f"{OUT}/tabla_incertidumbre.csv", index=False)

    # Histograma MC (escenario base)
    lam = LAM_BASE
    Hm = rng.normal(H, SIGMA_H, N_MC_UNC)
    lm = rng.normal(lam, SIGMA_LAM_REL * lam, N_MC_UNC)
    m = (Hm > 0) & (lm > 0)
    sm = S_optimo(Hm[m], lm[m])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(sm, bins=80, color="steelblue", alpha=0.8)
    ax.axvline(S_optimo(H, lam), color="r", label="S* nominal")
    ax.set_xlabel("S* [bytes]"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de S* (Monte Carlo, escenario base)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/grafica6_incertidumbre_MC.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# 7. SIMULACIÓN ESTOCÁSTICA DE RETRANSMISIONES (TCP simplificado)
# ----------------------------------------------------------------------
def simular_eficiencia_estocastica(S_val, H_val, lam, n_paq, n_lotes):
    """
    Para cada paquete se reintenta hasta que llega sin corrupción.
    Nº de intentos ~ Geométrica(p_exito = exp(-lam*S)).
    Eficiencia = bytes útiles entregados / bytes totales transmitidos
               = n_paq*S / sum(intentos*(S+H)).
    Devuelve media y error estándar entre lotes.
    """
    p_ok = np.exp(-lam * S_val)
    tam = n_paq // n_lotes
    efs = []
    for _ in range(n_lotes):
        intentos = rng.geometric(p_ok, size=tam)
        efs.append(tam * S_val / (intentos.sum() * (S_val + H_val)))
    efs = np.array(efs)
    return efs.mean(), efs.std(ddof=1) / np.sqrt(n_lotes)


def simulacion_montecarlo_retransmision():
    lam = 8 * ESCENARIOS["Alta tasa de error (p_b=1e-5)"]  # óptimo dentro del MTU
    tamanos = np.arange(50, 1501, 50, dtype=float)
    medias, errores = [], []
    for s in tamanos:
        m, e = simular_eficiencia_estocastica(s, H, lam, N_PAQUETES_SIM, N_LOTES)
        medias.append(m); errores.append(e)
    medias, errores = np.array(medias), np.array(errores)
    teorica = eficiencia(tamanos, H, lam)

    df = pd.DataFrame({"S [bytes]": tamanos, "E simulada": medias,
                       "error estándar": errores, "E teórica": teorica})
    df.to_csv(f"{OUT}/tabla_montecarlo_retransmision.csv", index=False)

    s_star = S_optimo(H, lam)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    Sx = np.linspace(50, 1500, 600)
    ax.plot(Sx, eficiencia(Sx, H, lam), "k-", label="Teórica E(S)")
    ax.errorbar(tamanos, medias, yerr=errores, fmt="o", ms=3, capsize=2,
                label="Simulación Monte Carlo")
    ax.axvline(s_star, color="r", ls="--", label=f"S* = {s_star:.0f} B")
    ax.set_xlabel("Tamaño del paquete S [bytes]")
    ax.set_ylabel("Eficiencia [adimensional]")
    ax.set_title(r"Simulación de retransmisiones ($p_b=10^{-5}$)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/grafica7_montecarlo.png", dpi=200)
    plt.close(fig)

    i = np.argmax(medias)
    print("\n[6] Monte Carlo de retransmisiones (p_b=1e-5)")
    print(f"    S* teórico = {s_star:.1f} B | máximo simulado en S = {tamanos[i]:.0f} B "
          f"(E = {medias[i]:.4f} ± {errores[i]:.4f})")
    chi = np.mean(np.abs(medias - teorica) / np.maximum(errores, 1e-12))
    print(f"    Desviación media simulación-teoría: {chi:.2f} errores estándar")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"H = {H} B | lambda_base = {LAM_BASE:g} 1/B | MSS = {MSS} B")
    simulacion_barrido()
    simulacion_analitico_vs_numerico()
    simulacion_escenarios()
    simulacion_sensibilidad()
    simulacion_incertidumbre()
    simulacion_montecarlo_retransmision()
    print(f"\nListo. Resultados guardados en ./{OUT}/")
