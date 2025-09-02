# Computing B22 from distance distributions

This repository contains a small toolkit and example workflow to compute and visualize the **second osmotic virial coefficient (B22)** from distance distributions obtained in MD/biased simulations. It includes:

- a Python module `b22_from_distances.py` with documented functions for B22,
- an example notebook workflow that prepares data from PLUMED `COLVAR` files,
- minimal and advanced plotting examples.

> **Short version:** You read distances from `COLVAR`, convert to a light data structure, choose radii `R1` and `R2`, compute B22 via block averaging, and (optionally) plot free-energy profiles to guide the `R1` scan.

---

## Installation

```bash
git clone https://github.com/IsabellGrothaus/B22_from_distance.git
cd B22_from_distance
pip install -r requirements.txt
```

**Requirements**
- Python ≥ 3.9
- `numpy`, `pandas`, `matplotlib`, `scipy`
- **PLUMED Python bindings** (for `plumed.read_as_pandas`) — install as per your PLUMED build.

---

## Data layout expected by the example

The example assumes a directory structure like:

```
./
├─ Man9/
│  ├─ TIP3P/TOPO0/COLVAR.0
│  ├─ TIP4PD/TOPO0/COLVAR.0
│  └─ TIP4PD--/TOPO0/COLVAR.0
└─ b22_from_distances.py
```

Each `COLVAR.0` must contain at least the columns `time` and `distance`.

---

## What the notebook code does (in short)

1. **Enumerate systems and conditions**:
   ```python
   systems = ["Man9"]
   conditions = ["TIP3P", "TIP4PD", "TIP4PD--"]
   ```
2. **Build a `data` dict** with labels, paths, temperature, bias flags, and spring constants.
3. **Read PLUMED output**:
   - `plumed.read_as_pandas(.../COLVAR.0)`
   - Keep columns `["time","distance"]`
   - Save `distance.dat` (whitespace, no header) for cheap reloads.
   - Convert time from ps → µs.
4. **Compute bias strength in kBT/nm²**:
   ```python
   data['k_in_kbT'][key] = b22.get_k( data['k_in_kJ_per_M_per_nm2'][key], T=data['temperature'][key] )
   ```
5. **Persist** the `keys` and `data` dicts as pickles for quick reloads.

---

## Minimal example (B22 for each system)

```python
import numpy as np
import pickle as pkl
import b22_tools as b22
import plumed

ipath = "./"
systems = ["Man9"]
conditions = ["TIP3P", "TIP4PD", "TIP4PD--"]

data = {k: {} for k in ["label","path","distance","temperature","is_restrained",
                        "k_in_kJ_per_M_per_nm2","k_in_kbT"]}
keys = {}

for s in systems:
    for c in conditions:
        keys[s] = conditions
        key_name = f"{s}/{c}"
        data['label'][key_name] = key_name
        data['path'][key_name] = f"{ipath}{s}/{c}"

        df = plumed.read_as_pandas(f"{data['path'][key_name]}/TOPO0/COLVAR.0")
        df = df.loc[:, ["time","distance"]]
        df.to_csv(f"{data['path'][key_name]}/distance.dat", sep=" ", header=False, index=False)
        data['distance'][key_name] = np.loadtxt(f"{data['path'][key_name]}/distance.dat")
        data['distance'][key_name][:,0] /= 1e6  # time: ps → µs
        data['temperature'][key_name] = 300.0
        data['is_restrained'][key_name] = True
        data['k_in_kJ_per_M_per_nm2'][key_name] = 500.0
        data['k_in_kbT'][key_name] = b22.get_k(
            data['k_in_kJ_per_M_per_nm2'][key_name],
            T=data['temperature'][key_name]
        )

keys["all"] = [f"{s}/{c}" for s in systems for c in conditions]

# Compute B22
R1 = 5.0   # inner radius (nm)
R2 = 15.0  # outer cutoff (nm)
n_blocks = 5

for key in keys["all"]:
    print(f"Processing system: {key}")
    distances = b22.apply_pbcs(data['distance'][key], R2)
    w = b22.get_weights(distances, data['k_in_kbT'][key])

    B22_blocks = b22.block_averaging(distances, w, R1, R2, n_blocks)
    mean, std, sem = b22.get_blocked_std_fixed_M(B22_blocks, n_blocks)

    to_L_per_mol = b22.unit_conv_fac  # nm^3 → L/mol
    print(f"B22 = {mean*to_L_per_mol:.3f} ± {sem*to_L_per_mol:.3f} L/mol")
    print("--------------------------------------------------")
```

**Key parameters**
- `R1` (nm): inner sphere radius for the subvolume probability.
- `R2` (nm): outer cutoff for distances; also defines the shell volume `dv = V(R2) - V(R1)`.
- `n_blocks`: number of contiguous blocks for block averaging (uncertainty).

---

## Advanced example (FEP guidance and model comparison)

**1) Identify flat free-energy regions to guide `R1`**

```python
color = ['black', 'red', 'maroon']
merged_ranges_models = {}
for key, c in zip(keys["all"], color):
    print(f"Processing system: {key}")
    merged_ranges = b22.calc_plot_FEP(
        data, key,
        fit_range_lower=14.0,
        fit_range_upper=15.9,
        threshold=0.5,   # kJ/mol band around average
        max_gap=0.2,     # merge subranges separated by ≤ this gap (nm)
        color=c,
        fontsize=16
    )
    merged_ranges_models[key] = merged_ranges
```

**2) Scan `R1` and plot B22 vs `R1` (per system)**

```python
R2 = {}
for key, c in zip(keys["all"], color):
    print(f"Processing system: {key}")
    R2[key] = data['distance'][key][:,1].max() * 0.90
    R1_test_range = np.arange(merged_ranges_models[key][0][0], R2[key]-1.0, 1.0)
    b22.plot_B22_for_R1_range(
        data, R1_test_range, R2, key,
        n_blocks=5, color=c, fontsize=16
    )
```

**3) Compare models with a weighted mean across `R1` ranges**

```python
R1_ranges = {
    "Man9/TIP3P":  np.arange(3, 4, 0.1),
    "Man9/TIP4PD": np.arange(5, 6, 0.1),
    "Man9/TIP4PD--": np.arange(11, 12, 0.1),
}

B22_means, B22_errs = b22.calculate_and_plot_B22(
    data, keys["all"], R1_ranges,
    colors=['black','red','maroon'],
    labels=[r"$\mathrm{GLYCAM06^{TIP3P}}$",
            r"$\mathrm{GLYCAM06^{TIP4P-D}}$",
            r"$\mathrm{GLYCAM06***^{TIP4P-D}}$"],
    ylim=(-700, 700)
)
```

---

## Persisting & reloading

```python
# Save
with open("Man9/keys_Man9.pkl", "wb") as fp:
    pkl.dump(keys, fp)
with open("Man9/data_Man9.pkl", "wb") as fp:
    pkl.dump(data, fp)

# Load
with open("Man9/keys_Man9.pkl", "rb") as fp:
    keys = pkl.load(fp)
with open("Man9/data_Man9.pkl", "rb") as fp:
    data = pkl.load(fp)
```

---

## Units & conventions (quick)

- Distances `r` in **nm** (column 1 of each `distance` array).
- Temperature in **K**.
- Spring constant input `k_in_kJ_per_M_per_nm2` converted to **kBT/nm²** via `b22.get_k`.
- B22 is computed in **nm³** and reported also in **L/mol** via `unit_conv_fac = 1e-24 * 6.022e23`.

---

## Tips & gotchas

- Ensure `distance.dat` has exactly two columns: `time distance` (whitespace-separated, **no header**).
- `R2` should be comfortably **below** the maximum observed distance (e.g., 90% of the max).

---

## Citation & license

- **Authors:** see header of `b22_tools.py`.
- **Versioning:** Semantic Versioning; current module `__version__` is updated per release.
- **License:** GNU GENERAL PUBLIC LICENSE 

---
