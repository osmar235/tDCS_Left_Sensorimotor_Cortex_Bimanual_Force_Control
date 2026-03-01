# Reproducible analysis: tDCS effects on bimanual force control

This repository contains code and data to reproduce the analyses and figures for the Bioengineering submission:

**Effects of Transcranial Direct Current Stimulation over the Left Sensorimotor Cortex on Bimanual Force Control: A Computa-tional and Experimental Investigation** *(submitted)*  
**Authors:** Vinicius de Moura Silva Lima¹, Eduarda Faria Arthur¹˒², Rafaela Rodrigues Dousseau Gonzaga¹˒², Luan Faria Diniz¹, Rodrigo Cunha de Mello Pedreiro³˒⁴, Osmar Pinto Neto¹˒²˒⁵\*

> If you use this repository, please cite the manuscript above (update with journal details/DOI when available).

---

## Repository contents

Expected repository structure:

```
.
├── tDCS_motorcontrol_03012026.py        # main reproducible analysis script
├── data_tdcs_12212025.csv               # summarized dataset used for statistics/figures
├── CTE FORCA/                           # raw force files (Vernier exports)  <-- add this folder
│   ├── <participant>_CONSTANTE_<date>_PRE.txt
│   └── <participant>_CONSTANTE_<date>_POS.txt
└── (generated) results/
```

### Raw data format (important)

Raw force files inside `CTE FORCA/` are Vernier exports with tab-separated columns (e.g., `Time`, `Force 1`, `Force 2`, `Forca Total`, `Alvo`).

**Decimal separator note:** raw data in `CTE FORCA/` may use **comma** as the decimal separator (e.g., `5,3034` instead of `5.3034`).  
Your loader must convert comma → dot **or** read with locale-aware parsing. This repository is configured with this in mind.

---

## Requirements

- Python 3.9+ recommended
- Packages:
  - numpy
  - pandas
  - scipy
  - matplotlib

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

---

## Quick start (minimal changes)

1. **Download or clone** the repository.

2. Ensure these files are present at the repository root:
   - `tDCS_motorcontrol_03012026.py`
   - `data_tdcs_12212025.csv`

   Optionally add the raw data folder:
   - `CTE FORCA/` (raw Vernier `.txt` files)

3. Open `tDCS_motorcontrol_03012026.py` and set the path variables near the top of the script, e.g.:

```python
DATA_PATH = r"/absolute/path/to/your/local/repo"
RAW_DATA_PATH = DATA_PATH  # or point to DATA_PATH + "/CTE FORCA"
```

4. Run:

```bash
python tDCS_motorcontrol_03012026.py
```

Outputs are written to a `results/` directory (created automatically if it does not exist).

---

## Raw data naming convention (CTE FORCA)

Raw Vernier files are expected to follow the pattern:

```
<participant>_CONSTANTE_<date>_PRE.txt
<participant>_CONSTANTE_<date>_POS.txt
```

Examples (illustrative):

```
DAS_CONSTANTE_21_10_PRE.txt
DAS_CONSTANTE_21_10_POS.txt
```

If raw files are not present or not found, the script should still run the summarized analyses and will skip any optional steps that require raw traces.

---

## Troubleshooting

### “Data file not found”
- Confirm `data_tdcs_12212025.csv` is located in `DATA_PATH`.
- Confirm `DATA_PATH` points to the repository root.

### Raw traces not loading
- Confirm `RAW_DATA_PATH` points to the folder containing the raw `.txt` files (usually `.../CTE FORCA`).
- Confirm filenames follow the naming convention.

### Decimal/comma issues in raw files
- Ensure your parser converts `,` → `.` before casting to float, or uses locale-aware parsing.

---

## Data and code availability

- **Code:** `tDCS_motorcontrol_03012026.py` (and any additional scripts you add)
- **Summarized dataset:** `data_tdcs_12212025.csv`
- **Raw data:** provided in `CTE FORCA/` (to be added), with comma decimal separators

---

## Licenses (dual)

This repository uses **two licenses**:

### 1) Code license — MIT
All **source code** in this repository is licensed under the **MIT License**.

**SPDX identifier:** MIT

### 2) Data license — Creative Commons Attribution 4.0 International (CC BY 4.0)
All **data files** in this repository (including `data_tdcs_12212025.csv` and the contents of `CTE FORCA/`) are licensed under **CC BY 4.0**.

**SPDX identifier:** CC-BY-4.0

In plain terms:
- You may share and adapt the code (MIT) with attribution and license notice.
- You may share and adapt the data (CC BY 4.0) **with appropriate attribution**.

> Tip: Add `LICENSE` (MIT) and `LICENSE-DATA` (CC BY 4.0) at the repository root for maximum clarity.

---

## How to cite

Please cite the manuscript (submitted):

Vinicius de Moura Silva Lima, Eduarda Faria Arthur, Rafaela Rodrigues Dousseau Gonzaga, Luan Faria Diniz, Rodrigo Cunha de Mello Pedreiro, **Osmar Pinto Neto**.  
*Effects of Transcranial Direct Current Stimulation over the Left Sensorimotor Cortex on Bimanual Force Control: A Computa-tional and Experimental Investigation.* Bioengineering (submitted).

---

## Contact

For questions or issues, please open a GitHub Issue or contact the corresponding author:

- **Osmar Pinto Neto** (\*) — arena235research@gmail.com or osmar@csusm.edu
