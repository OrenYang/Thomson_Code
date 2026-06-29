import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== EDIT THIS =====================
csv_path = '07237_TS_shotn_results.csv'
# =======================================================

df = pd.read_csv(csv_path)

if 'r_um' in df.columns and df['r_um'].notna().any():
    df = df.sort_values('r_um')
    x = df['r_um'].values
    xlabel = 'r (um)'
else:
    df = df.sort_values('fiber')
    x = df['fiber'].values
    xlabel = 'Fiber number'

colors = plt.get_cmap('tab10')

def find_cols(prefix):
    """Find all columns starting with prefix_<index>, e.g. 'T_i_0_Ne-20+8',
    'T_i_1_Ne-200+', ... Returns list of (index, column_name, species_label)
    sorted by index."""
    matches = []
    for c in df.columns:
        if c.endswith('_err'):
            continue
        if c.startswith(prefix + '_'):
            rest = c[len(prefix) + 1:]
            idx_str = rest.split('_')[0]
            if idx_str.isdigit():
                idx = int(idx_str)
                label = rest[len(idx_str):].lstrip('_') or idx_str
                matches.append((idx, c, label))
    matches.sort(key=lambda t: t[0])
    return matches

def err(col):
    col_err = col + '_err'
    if col_err in df.columns:
        return df[col_err].fillna(0).values
    return np.zeros(len(df))

def save(fig, name):
    fig.savefig(csv_path.rsplit('.', 1)[0] + f'_{name}.png', dpi=150)

# --- Ion fractions ---
fig1, ax = plt.subplots(figsize=(8, 5))
for idx, col, label in find_cols('ifract'):
    ax.errorbar(x, df[col], yerr=err(col), fmt='o-', color=colors(idx), capsize=3, label=label)
ax.set_ylabel('Ion fraction')
ax.set_xlabel(xlabel)
ax.set_title('Ion fraction vs r')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()


# --- Electron temperature ---
fig2, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, df['T_e_0'], yerr=err('T_e_0'), fmt='o-', color='C3', capsize=3)
ax.set_ylabel('T_e (eV)')
ax.set_xlabel(xlabel)
ax.set_title('Electron temperature vs r')
ax.grid(True, alpha=0.3)
plt.tight_layout()


# --- Ion temperatures ---
fig3, ax = plt.subplots(figsize=(8, 5))
for idx, col, label in find_cols('T_i'):
    ax.errorbar(x, df[col], yerr=err(col), fmt='o-', color=colors(idx), capsize=3, label=label)
ax.set_ylabel('T_i (eV)')
ax.set_xlabel(xlabel)
ax.set_title('Ion temperature vs r')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()


# --- Electron speed ---
fig4, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, df['electron_speed_0'] / 1e3, yerr=err('electron_speed_0') / 1e3,
            fmt='o-', color='C3', capsize=3)
ax.set_ylabel('v_e (km/s)')
ax.set_xlabel(xlabel)
ax.set_title('Electron speed vs r')
ax.grid(True, alpha=0.3)
plt.tight_layout()


# --- Ion speeds ---
fig5, ax = plt.subplots(figsize=(8, 5))
for idx, col, label in find_cols('ion_speed'):
    ax.errorbar(x, df[col] / 1e3, yerr=err(col) / 1e3, fmt='o-', color=colors(idx),
                capsize=3, label=label)
ax.set_ylabel('v_i (km/s)')
ax.set_xlabel(xlabel)
ax.set_title('Ion speed vs r')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()


# --- Density ---
fig6, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, df['n'], yerr=err('n'), fmt='o-', color='C4', capsize=3)
ax.set_ylabel('n (m^-3)')
ax.set_yscale('log')
ax.set_xlabel(xlabel)
ax.set_title('Density vs r')
ax.grid(True, alpha=0.3)
plt.tight_layout()


plt.show()
