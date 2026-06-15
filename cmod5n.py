import numpy as np

def cmod5n(v, phi, theta):
    """
    CMOD5.N Geophysical Model Function
    Computes normalized radar cross section (sigma0) from wind.

    v     : wind speed in m/s
    phi   : wind direction relative to radar look direction in degrees
    theta : incidence angle in degrees

    Returns: sigma0 in linear scale (not dB)

    Reference: Hersbach et al. (2007), Journal of Geophysical Research
    """
    # Convert angles to radians
    phi_rad   = np.deg2rad(phi)
    theta_rad = np.deg2rad(theta)

    # CMOD5.N coefficients
    C = [0, -0.6878, -0.7957, 0.3380, -0.1728,
         0.0000, 0.0040, 0.1103, 0.0159,
         6.7329, 2.7713, -2.2885, 3.1584,
         -0.4249, -0.0161, 0.0, 0.0, 0.0,
         -0.4249, 0.0, 0.0]

    # Angle-dependent terms
    cos_phi  = np.cos(phi_rad)
    cos_2phi = np.cos(2 * phi_rad)

    # Incidence angle function
    x = (theta_rad - np.deg2rad(40)) / np.deg2rad(25)

    # B0 — isotropic term (upwind + downwind average)
    B0 = 10 ** (
        C[1]
        + C[2] * x
        + C[3] * x**2
        + C[4] * (v - 7.5)
        + C[5] * (v - 7.5)**2
    )

    # B1 — upwind/downwind asymmetry
    B1 = C[6] + C[7] * x + C[8] * x**2

    # B2 — crosswind term
    B2 = (
        C[9]
        + C[10] * x
        + C[11] * x**2
        + C[12] * x**3
    ) / (1 + v)

    # Final sigma0
    sigma0 = B0 * (1 + B1 * cos_phi + B2 * cos_2phi) ** 1.6

    return sigma0


def retrieve_wind_speed(sigma0_linear, phi_deg, theta_deg,
                        v_min=0.5, v_max=50.0):
    """
    Invert CMOD5.N to retrieve wind speed from sigma0.

    sigma0_linear : backscatter in LINEAR scale (not dB)
    phi_deg       : wind direction relative to radar look (degrees)
    theta_deg     : incidence angle (degrees)

    Returns: wind speed in m/s
    """
    v = 10.0  # initial guess — 10 m/s

    for _ in range(50):  # max 50 iterations
        # Forward model
        f = cmod5n(v, phi_deg, theta_deg) - sigma0_linear

        # Numerical derivative
        h  = 0.001
        df = (cmod5n(v + h, phi_deg, theta_deg) -
              cmod5n(v - h, phi_deg, theta_deg)) / (2 * h)

        if abs(df) < 1e-10:
            break  # avoid division by zero

        # Newton-Raphson update
        v_new = v - f / df

        # Clamp to physical range
        v_new = np.clip(v_new, v_min, v_max)

        # Check convergence
        if abs(v_new - v) < 1e-4:
            return float(v_new)

        v = v_new

    return float(v)


def sigma0_db_to_linear(sigma0_db):
    """Convert backscatter from dB to linear scale."""
    return 10 ** (sigma0_db / 10)


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test case: typical monsoon conditions over Gujarat
    # sigma0 around -10 dB is typical for ~8 m/s wind
    sigma0_db  = float(input("Enter sigma0 in dB (typical range -20 to -5): "))
    theta_deg  = float(input("Enter incidence angle in degrees (typical 20-45): "))
    phi_deg    = float(input("Enter wind direction relative to radar look (degrees): "))

    # Convert sigma0 from dB to linear
    sigma0_lin = sigma0_db_to_linear(sigma0_db)
    print(f"\nSigma0 linear: {sigma0_lin:.6f}")

    # Retrieve wind speed
    wind_speed = retrieve_wind_speed(sigma0_lin, phi_deg, theta_deg)
    print(f"Retrieved wind speed: {wind_speed:.2f} m/s")

    # Verify by running forward model
    sigma0_check = cmod5n(wind_speed, phi_deg, theta_deg)
    print(f"Verification — forward model sigma0: {sigma0_check:.6f}")
    print(f"Original sigma0:                     {sigma0_lin:.6f}")
    print(f"Difference:                           {abs(sigma0_check - sigma0_lin):.8f}")