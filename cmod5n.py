import numpy as np


# ── Official CMOD5.N coefficients (Hersbach et al., 2007) ─────────────────────
# Source: ECMWF Technical Memorandum 601
C = [
    0.0,        # C0  — unused (1-indexed)
    -0.6878,    # C1
    -0.7957,    # C2
     0.3380,    # C3
    -0.1728,    # C4
     0.0000,    # C5
     0.0040,    # C6
     0.1103,    # C7
     0.0159,    # C8
     6.7329,    # C9
     2.7713,    # C10
    -2.2885,    # C11
     3.1584,    # C12
    -0.4249,    # C13
    -0.0161,    # C14
     0.0000,    # C15
     0.0000,    # C16
     0.0000,    # C17
    -0.4249,    # C18
     0.0000,    # C19
     0.0000,    # C20
]

# CMOD5.N neutral stability correction coefficients
CN = [
    0.0,        # CN0 — unused
     0.2,       # CN1
     0.4,       # CN2
    -0.0006,    # CN3
     0.0100,    # CN4
     0.0040,    # CN5
     0.1600,    # CN6
     0.0040,    # CN7
]


def cmod5n(v, phi_deg, theta_deg):
    """
    CMOD5.N Geophysical Model Function.
    Computes normalized radar cross section sigma0 from wind.

    Based on Hersbach (2010) ECMWF Technical Memorandum 601.
    This is the standard operational GMF used by ESA and ECMWF.

    Parameters
    ----------
    v         : wind speed in m/s (scalar or numpy array)
    phi_deg   : wind direction relative to radar look direction (degrees)
    theta_deg : incidence angle (degrees)

    Returns
    -------
    sigma0 in linear scale (not dB)
    """
    v      = np.asarray(v, dtype=np.float64)
    phi    = np.deg2rad(np.asarray(phi_deg,   dtype=np.float64))
    theta  = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))

    # ── Incidence angle function ──────────────────────────────────────────────
    # Normalised incidence angle centred at 40°
    fi = (theta - np.deg2rad(40.0)) / np.deg2rad(25.0)

    # ── B0 — isotropic term ───────────────────────────────────────────────────
    # B0 determines the mean level of sigma0
    b0_exp = (C[1]
              + C[2] * fi
              + C[3] * fi**2
              + (C[4] + C[5] * fi) * v)

    B0 = 10.0 ** b0_exp

    # ── B1 — upwind/downwind asymmetry ────────────────────────────────────────
    B1 = (C[6]
          + C[7] * fi
          + C[8] * fi**2)

    # ── B2 — crosswind modulation ─────────────────────────────────────────────
    B2 = ((C[9]
           + C[10] * fi
           + C[11] * fi**2
           + C[12] * fi**3)
          / (1.0 + np.exp(-0.5 * (v - 7.5))))

    # ── Neutral stability correction (CMOD5.N vs CMOD5) ──────────────────────
    # N suffix = neutral stability correction applied
    v_n = v + CN[1] * np.exp(CN[2] * v) * (CN[3] * theta**2 + CN[4] * theta)
    B0_n = B0 * (1.0 + CN[5] * v_n * np.exp(-CN[6] * v_n))

    # ── Final sigma0 ──────────────────────────────────────────────────────────
    cos_phi  = np.cos(phi)
    cos_2phi = np.cos(2.0 * phi)

    sigma0 = B0_n * np.abs(1.0 + B1 * cos_phi + B2 * cos_2phi) ** 1.6

    return float(sigma0) if sigma0.ndim == 0 else sigma0


def sigma0_db_to_linear(sigma0_db):
    """Convert backscatter from dB to linear scale."""
    return 10.0 ** (sigma0_db / 10.0)


def sigma0_linear_to_db(sigma0_linear):
    """Convert backscatter from linear to dB scale."""
    return 10.0 * np.log10(sigma0_linear)


def retrieve_wind_speed(sigma0_linear, phi_deg, theta_deg,
                        v_min=0.2, v_max=50.0, max_iter=50):
    """
    Invert CMOD5.N to retrieve wind speed from sigma0.

    Uses Newton-Raphson numerical inversion.

    Parameters
    ----------
    sigma0_linear : backscatter in LINEAR scale (not dB)
    phi_deg       : wind direction relative to radar look (degrees)
    theta_deg     : incidence angle (degrees)
    v_min         : minimum physical wind speed (m/s)
    v_max         : maximum physical wind speed (m/s)
    max_iter      : maximum Newton-Raphson iterations

    Returns
    -------
    wind speed in m/s
    """
    # Initial guess based on approximate inverse relationship
    v = np.clip(np.sqrt(sigma0_linear * 100.0), v_min, v_max)

    h = 0.01  # step for numerical derivative

    for _ in range(max_iter):
        f  = cmod5n(v, phi_deg, theta_deg) - sigma0_linear
        df = (cmod5n(v + h, phi_deg, theta_deg) -
              cmod5n(v - h, phi_deg, theta_deg)) / (2.0 * h)

        if abs(df) < 1e-12:
            break

        v_new = v - f / df
        v_new = np.clip(v_new, v_min, v_max)

        if abs(v_new - v) < 1e-5:
            return float(v_new)

        v = v_new

    return float(v)


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CMOD5.N implementation...\n")

    # Test cases from Hersbach (2007) Table 1
    test_cases = [
        {"v": 5.0,  "phi": 0.0,  "theta": 25.0, "desc": "Low wind, upwind"},
        {"v": 10.0, "phi": 0.0,  "theta": 35.0, "desc": "Medium wind, upwind"},
        {"v": 15.0, "phi": 0.0,  "theta": 40.0, "desc": "High wind, upwind"},
        {"v": 10.0, "phi": 90.0, "theta": 35.0, "desc": "Medium wind, crosswind"},
        {"v": 10.0, "phi": 45.0, "theta": 35.0, "desc": "Medium wind, 45deg"},
    ]

    print(f"{'Description':<30} {'Speed':>6} {'Phi':>6} {'Theta':>6} "
          f"{'sigma0':>10} {'sigma0_dB':>10} {'Inverted':>10}")
    print("-" * 80)

    for tc in test_cases:
        s0     = cmod5n(tc["v"], tc["phi"], tc["theta"])
        s0_db  = sigma0_linear_to_db(s0)
        v_inv  = retrieve_wind_speed(s0, tc["phi"], tc["theta"])
        print(f"{tc['desc']:<30} {tc['v']:>6.1f} {tc['phi']:>6.1f} "
              f"{tc['theta']:>6.1f} {s0:>10.6f} {s0_db:>10.2f} {v_inv:>10.3f}")

    print("\nInversion test — should match input speed:")
    sigma0_test = cmod5n(12.0, 45.0, 35.0)
    v_recovered = retrieve_wind_speed(sigma0_test, 45.0, 35.0)
    print(f"  Input:     12.000 m/s")
    print(f"  Recovered: {v_recovered:.3f} m/s")
    print(f"  Error:     {abs(v_recovered - 12.0):.6f} m/s")
