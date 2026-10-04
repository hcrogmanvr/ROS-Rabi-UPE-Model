import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conditions = ["Healthy", "COVID-19", "Sepsis"]
ros_fold = np.array([1.00, 1378 / 153, 2120 / 153])
wavelength = np.linspace(350, 900, 1000)

emitters = [
    ("Excited carbonyl / lipid peroxidation", 470, 45),
    ("Singlet oxygen-related red band", 634, 35),
    ("Singlet oxygen / near-red band", 703, 40),
    ("Near-IR oxidative component", 800, 55),
]

def gaussian(lam, lam0, sigma):
    return np.exp(-0.5 * ((lam - lam0) / sigma) ** 2)

def band_weights(ros):
    w_blue = 1.0 + 0.8 * ros / (2.0 + ros)
    w_red634 = 0.25 + 1.8 * ros / (5.0 + ros)
    w_red703 = 0.15 + 2.2 * ros / (7.0 + ros)
    w_nir = 0.05 + 1.7 * ros / (12.0 + ros)
    return np.array([w_blue, w_red634, w_red703, w_nir])

spectra = {}
rows = []
for condition, rf in zip(conditions, ros_fold):
    weights = band_weights(rf)
    total = np.zeros_like(wavelength)
    for weight, (_, lam0, sigma) in zip(weights, emitters):
        total += weight * gaussian(wavelength, lam0, sigma)
    total_norm = total / np.trapz(total, wavelength)
    centroid = np.trapz(wavelength * total_norm, wavelength) / np.trapz(total_norm, wavelength)
    peak_lam = wavelength[np.argmax(total_norm)]
    frac = weights / weights.sum()
    spectra[condition] = total_norm
    rows.append({
        "Condition": condition,
        "ROS fold-change": rf,
        "Predicted spectral centroid (nm)": centroid,
        "Predicted peak wavelength (nm)": peak_lam,
        "Carbonyl/lipid band fraction": frac[0],
        "634 nm band fraction": frac[1],
        "703 nm band fraction": frac[2],
        "Near-IR band fraction": frac[3],
    })

df = pd.DataFrame(rows)
print(df.round(2))

plt.figure(figsize=(7.8, 5.2))
for condition in conditions:
    plt.plot(wavelength, spectra[condition], linewidth=2, label=condition)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized spectral intensity")
plt.title("Model-predicted UPE spectral redistribution with infection severity")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_upe_spectral_shift.png", dpi=300)
plt.show()

plt.figure(figsize=(6.5, 4.8))
x = np.arange(len(conditions))
centroids = df["Predicted spectral centroid (nm)"].values
plt.bar(x, centroids)
plt.xticks(x, conditions)
plt.ylabel("Predicted spectral centroid (nm)")
plt.title("Predicted infection-associated UPE wavelength shift")
for i, c in enumerate(centroids):
    plt.text(i, c + 5, f"{c:.0f} nm", ha="center", fontsize=9)
plt.ylim(0, max(centroids) + 100)
plt.tight_layout()
plt.savefig("predicted_upe_centroid_shift.png", dpi=300)
plt.show()
