# ROS-Rabi-UPE Model

Repository scaffold for the manuscript:

**A Multiscale Quantum-Kinetic Model of Infection-Associated Ultraweak Photon Emission: Structural Uncertainty in the ROS-to-Photon Transfer Function**

## Overview

This project develops a multiscale quantum-kinetic framework linking infection dynamics, immune activation, reactive oxygen species (ROS), emission-relevant oxidative chemistry, chemiexcitation, microscopic electron transfer, open quantum-system population dynamics, and wavelength-resolved ultraweak photon emission (UPE).

The model distinguishes two physical levels:

1. **Microscopic reactive encounters** represented by a two-state electron-transfer model with a Rabi-like coupling scale.
2. **Biological ensemble dynamics** represented by an incoherent chemical-pumping process coupled to infection, immune, oxidative, and photophysical dynamics.

The manuscript evaluates structural uncertainty in the ROS-to-UPE transfer function using high-gain and saturating mappings, literature-constrained oxidative regimes, respiratory-burst inhibition consistency testing, global uncertainty propagation, Sobol sensitivity analysis, and alternative spectral line-shape models.

## Repository structure

```text
ROS-Rabi-UPE-Model/
├── README.md
├── CITATION.cff
├── .gitignore
├── requirements.txt
├── manuscript/
│   └── Manuscript_ROS_Rabi_UPE.docx
├── src/
│   └── README.md
├── data/
│   └── README.md
├── results/
│   └── README.md
├── figures/
│   └── README.md
└── docs/
    └── reproducibility_checklist.md
```

## Current status

The manuscript is included. The computational source code, machine-readable parameter tables, simulation outputs, and figure-generation scripts should be added to the corresponding folders before the repository is used as the permanent reproducibility archive for publication.

## Core manuscript outputs

The current manuscript reports:

- Literature-constrained relative oxidative regimes for healthy, severe COVID-19, and sepsis conditions.
- Microscopic ROS-mediated electron-transfer calculations using a Rabi-like coupling scale.
- Reduction from transient coherent encounter dynamics to an incoherent ensemble chemical pump.
- Comparison of high-gain and saturating ROS-to-UPE transfer structures.
- PMA/DPI respiratory-burst pathway-level consistency testing.
- Latin Hypercube uncertainty propagation and Sobol sensitivity analysis.
- Wavelength-resolved UPE predictions using Voigt, Gaussian, and log-normal spectral representations.

## Reproducibility

Before public archival release, add:

- Exact Python and package versions.
- Complete source code for all model components.
- Fixed random seeds.
- Parameter tables and uncertainty distributions.
- Raw and processed simulation outputs.
- Spectral templates and detector-response functions.
- Scripts that reproduce every figure and table.

See `docs/reproducibility_checklist.md`.

## Related publication

Crogman, H. T., Alvarez, G., Clement, P. U., Sonawane, R. B., Chidananda, R., Khan, H., Eshun, K., Joseph, E., & Erenso, D. B. (2026). *A Multiscale Mechanistic Framework Linking Infection Dynamics, Oxidative Chemiexcitation, and Ultraweak Photon Emission*. **Biophysica, 6**, 87. https://doi.org/10.3390/biophysica6050087

## License

No software or data license has been selected yet. Add an appropriate license before public release.

## v1.1 analysis update (October 2026)

The repository update adds empirical-basis spectra, an effective Franck-Condon carbonyl model, temporal metrics, detector-forward spectral analysis, revised Discussion material, and manuscript-wide terminology harmonization. See `docs/updates/UPDATE_NOTES_v1.1.md`.
