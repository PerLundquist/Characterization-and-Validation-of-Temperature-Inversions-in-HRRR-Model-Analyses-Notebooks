# Characterization and Validation of Temperature Inversions in HRRR Model Analyses

This repository contains Jupyter notebooks used for the methods presented in the poster titled "Characterization and Validation of Temperature Inversions in HRRR Model Analyses," which will be presented at the 104th AMS Annual Meeting.

## Poster Presentation Overview

The poster presentation explores mehods to detect and validate temperature inversions within upper air soundings produced by the High Resolution Rapid Refresh (HRRR) Numerical Weather Prediction (NWP) model. The work involves:

- **Vector_Method.ipynb**: Detection and parsing of major temperature inversions within an upper air sounding. This notebook illustrates the process of identifying inversion vectors and ranking them based on their distances in a scaled Temperature-Height space. The primary inversion of interest within a sounding is produced.

- **Inversion_Comparison.ipynb**: Building on the previous method, this notebook creates a method for determining when the HRRR correctly produces and inversion by comparing modeled sounding data from HRRR with observed sounding data. 

- **HRRR_Correctness_Logger.ipynb**: Extending the analysis, this notebook collects six years of observed and modeled upper air sounding data from four separate NWS offices. It quantifies the correctness of the HRRR in producing temperature inversions, no inversions, false positives, and false negatives.

## Notebooks Overview

Each notebook includes detailed explanations of the methods presented, citations for data sources and python libraries, and comparisons between observed and modeled data using various Python libraries and tools.

## Getting Started

To replicate the analysis or understand the methods used in the poster presentation, follow these steps:

1. Clone the repository: `git clone https://github.com/your_username/repository_name.git`
2. Open the desired notebook using Jupyter Notebook or JupyterLab.
3. Ensure the necessary Python libraries are installed (listed in each notebook).
4. Run the cells in sequential order to understand the methods and reproduce the analyses.

### Contact Information

- **Per Lundquist,** *M.S. Graduate Student*
- Atmospheric & Environmental Sciences
- South Dakota Mines
- 501 E. Saint Joseph St., Rapid City, SD 57701
- Phone: 605.951.3251 | Email: Per.Lundquist@mines.sdsmt.edu
