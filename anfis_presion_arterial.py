"""
Red Neurodifusa Takagi-Sugeno Orden 0
======================================
Fenómeno: Presión arterial sistólica → Riesgo cardiovascular
Entrada : presión arterial sistólica [80, 200] mmHg
Salida  : nivel de riesgo cardiovascular [0, 10]

Conjuntos difusos de la entrada:
  - Normal    : trapezoidal  [80,  80, 115, 125]
  - Elevada   : triangular   [115, 130, 145]
  - Alta      : triangular   [135, 150, 170]
  - Muy Alta  : trapezoidal  [160, 175, 200, 200]

Reglas Takagi-Sugeno orden 0 (salida constante):
  R1: Si presión es Normal    → riesgo = 1.5
  R2: Si presión es Elevada   → riesgo = 4.0
  R3: Si presión es Alta      → riesgo = 6.5
  R4: Si presión es Muy Alta  → riesgo = 9.0
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. FUNCIONES DE PERTENENCIA
# ─────────────────────────────────────────────

def trapezoidal(x, a, b, c, d):
    """Función de pertenencia trapezoidal."""
    y = np.zeros_like(x, dtype=float)
    mask = (x > a) & (x <= b)
    y[mask] = (x[mask] - a) / (b - a) if b != a else 1.0
    mask = (x > b) & (x <= c)
    y[mask] = 1.0
    mask = (x > c) & (x <= d)
    y[mask] = (d - x[mask]) / (d - c) if d != c else 1.0
    return y

def triangular(x, a, b, c):
    """Función de pertenencia triangular."""
    y = np.zeros_like(x, dtype=float)
    mask = (x > a) & (x <= b)
    y[mask] = (x[mask] - a) / (b - a)
    mask = (x > b) & (x <= c)
    y[mask] = (c - x[mask]) / (c - b)
    return y

# ─────────────────────────────────────────────
# 2. RED TAKAGI-SUGENO ORDEN 0
# ─────────────────────────────────────────────

def anfis_ts0(presion):
    """
    Capa 1 – Fuzzificación: grado de pertenencia μ_i(x)
    Capa 2 – Fuerza de disparo: w_i = μ_i(x)
    Capa 3 – Normalización: w_i_norm = w_i / Σ w_j
    Capa 4 – Consecuentes T-S: f_i = constante
    Capa 5 – Salida: y = Σ w_i_norm * f_i
    """
    x = np.atleast_1d(np.array(presion, dtype=float))

    # Constantes de salida por regla (orden 0)
    f = np.array([1.5, 4.0, 6.5, 9.0])

    # Capa 1: Fuzzificación
    mu_normal  = trapezoidal(x,  80,  80, 115, 125)
    mu_elevada = triangular (x, 115, 130, 145)
    mu_alta    = triangular (x, 135, 150, 170)
    mu_muyalta = trapezoidal(x, 160, 175, 200, 200)

    # Capa 2: Fuerzas de disparo
    W = np.stack([mu_normal, mu_elevada, mu_alta, mu_muyalta], axis=1)

    # Capa 3: Normalización
    W_sum = W.sum(axis=1, keepdims=True)
    W_sum = np.where(W_sum == 0, 1e-10, W_sum)
    W_norm = W / W_sum

    # Capas 4-5: Salida defuzzificada
    riesgo = (W_norm * f).sum(axis=1)

    return riesgo, W, W_norm

# ─────────────────────────────────────────────
# 3. UNIVERSO DE DISCURSO
# ─────────────────────────────────────────────
input("Acomoda las ventanas donde desees y presiona Enter para generar las gráficas...")

x_universe = np.linspace(80, 200, 500)
riesgo_out, W_all, W_norm_all = anfis_ts0(x_universe)

# ─────────────────────────────────────────────
# 4. GRÁFICA 1 – SEÑAL DE ENTRADA
# ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(x_universe, x_universe, color='royalblue', linewidth=2.5, label='Presión sistólica (mmHg)')
ax1.set_xlabel('Muestra / barrido del universo', fontsize=12)
ax1.set_ylabel('Presión arterial sistólica (mmHg)', fontsize=12)
ax1.set_title('Gráfica 1 – Señal de Entrada\nPresión Arterial Sistólica', fontsize=14, fontweight='bold')
ax1.set_xlim(80, 200)
ax1.set_ylim(70, 210)

ax1.axhspan(80,  120, alpha=0.08, color='green',  label='Normal (<120)')
ax1.axhspan(120, 130, alpha=0.08, color='yellow', label='Elevada (120–130)')
ax1.axhspan(130, 180, alpha=0.08, color='orange', label='Alta (130–180)')
ax1.axhspan(180, 200, alpha=0.08, color='red',    label='Crisis (≥180)')

ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.4)
fig1.tight_layout()
fig1.savefig('grafica1_entrada.png', dpi=150, bbox_inches='tight')
print("Gráfica 1 guardada")

# ─────────────────────────────────────────────
# 5. GRÁFICA 2 – SALIDA DE LA RED
# ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(x_universe, riesgo_out, color='crimson', linewidth=2.5, label='Riesgo cardiovascular')
ax2.set_xlabel('Presión arterial sistólica (mmHg)', fontsize=12)
ax2.set_ylabel('Nivel de riesgo cardiovascular (0–10)', fontsize=12)
ax2.set_title('Gráfica 2 – Salida de la Red Neurodifusa\nRiesgo Cardiovascular (T-S orden 0)', fontsize=14, fontweight='bold')
ax2.set_xlim(80, 200)
ax2.set_ylim(0, 11)

for yi, label, col in zip([1.5, 4.0, 6.5, 9.0],
                           ['Normal (1.5)', 'Elevada (4.0)', 'Alta (6.5)', 'Muy Alta (9.0)'],
                           ['green', 'goldenrod', 'darkorange', 'red']):
    ax2.axhline(yi, linestyle=':', color=col, linewidth=1.2, alpha=0.7, label=label)

ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.4)
fig2.tight_layout()
fig2.savefig('grafica2_salida.png', dpi=150, bbox_inches='tight')
print("Gráfica 2 guardada")

# ─────────────────────────────────────────────
# 6. GRÁFICA 3 – CONJUNTOS DIFUSOS
# ─────────────────────────────────────────────
colores   = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
etiquetas = ['Normal', 'Elevada', 'Alta', 'Muy Alta']

mu_normal_  = trapezoidal(x_universe,  80,  80, 115, 125)
mu_elevada_ = triangular (x_universe, 115, 130, 145)
mu_alta_    = triangular (x_universe, 135, 150, 170)
mu_muyalta_ = trapezoidal(x_universe, 160, 175, 200, 200)
mus = [mu_normal_, mu_elevada_, mu_alta_, mu_muyalta_]

fig3, ax3 = plt.subplots(figsize=(9, 5))
for mu, color, etiqueta in zip(mus, colores, etiquetas):
    ax3.plot(x_universe, mu, color=color, linewidth=2.5, label=etiqueta)
    ax3.fill_between(x_universe, mu, alpha=0.12, color=color)

ax3.set_xlabel('Presión arterial sistólica (mmHg)', fontsize=12)
ax3.set_ylabel('Grado de pertenencia μ', fontsize=12)
ax3.set_title('Gráfica 3 – Conjuntos Difusos\nFunciones de Pertenencia de la Variable Lingüística', fontsize=14, fontweight='bold')
ax3.set_xlim(80, 200)
ax3.set_ylim(-0.05, 1.15)
ax3.legend(fontsize=11, loc='upper center', ncol=4)
ax3.grid(True, linestyle='--', alpha=0.4)

constantes = [1.5, 4.0, 6.5, 9.0]
picos = [80, 130, 150, 200]
for p, const, color in zip(picos, constantes, colores):
    ax3.annotate(f'f={const}', xy=(p, 1.0), xytext=(p, 1.07),
                 ha='center', fontsize=9, color=color, fontweight='bold')

fig3.tight_layout()
fig3.savefig('grafica3_conjuntos.png', dpi=150, bbox_inches='tight')
print("Gráfica 3 guardada")

# ─────────────────────────────────────────────
# 7. PRUEBA PUNTUAL
# ─────────────────────────────────────────────
print("\n── Ejemplo de inferencia ──")
for p in [100, 128, 150, 185]:
    r, w, wn = anfis_ts0(p)
    print(f"  Presión = {p:3d} mmHg  →  Riesgo = {r[0]:.3f}  "
          f"| μ: Normal={w[0,0]:.2f}  Elevada={w[0,1]:.2f}  "
          f"Alta={w[0,2]:.2f}  MuyAlta={w[0,3]:.2f}")
print("\nScript completo.")

# ─────────────────────────────────────────────
# 8. MOSTRAR LAS TRES GRÁFICAS EN PANTALLA
# ─────────────────────────────────────────────
plt.show()
