from pathlib import Path

import numpy as np
import pyvista as pv


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_MESH = Path(
    r"C:\Users\Jaaps\Documents\Trabmap\trabmap_github\input"
    r"\StJF970_TIB_dist\mappings"
    r"\StJF970_TIB_dist_trabecular_tetra_mesh.vtk"
)

OUTPUT_MESH = INPUT_MESH.with_name(
    INPUT_MESH.stem + "_DA_comparison.vtk"
)

DA_ST_COLUMN = "DA_ST"
DA_MIL_COLUMN = "DA_MIL"

# Values at or below this absolute difference can optionally be classified
# as agreement. Adjust based on the range and biological relevance of DA.
AGREEMENT_TOLERANCE = 0.05


# =============================================================================
# HELPERS
# =============================================================================

def find_array_location(mesh: pv.DataSet, array_name: str) -> str:
    """Return whether an array is stored as cell data or point data."""
    if array_name in mesh.cell_data:
        return "cell"

    if array_name in mesh.point_data:
        return "point"

    available_cell = list(mesh.cell_data.keys())
    available_point = list(mesh.point_data.keys())

    raise KeyError(
        f"Could not find '{array_name}' in the mesh.\n"
        f"Cell-data arrays: {available_cell}\n"
        f"Point-data arrays: {available_point}"
    )


def safe_numeric_array(values: np.ndarray) -> np.ndarray:
    """Convert mesh values to a floating-point NumPy array."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional scalar array, got shape {values.shape}"
        )

    return values


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("Loading mesh:")
    print(INPUT_MESH)

    mesh = pv.read(INPUT_MESH)

    st_location = find_array_location(mesh, DA_ST_COLUMN)
    mil_location = find_array_location(mesh, DA_MIL_COLUMN)

    if st_location != mil_location:
        raise ValueError(
            f"{DA_ST_COLUMN} is stored as {st_location} data, but "
            f"{DA_MIL_COLUMN} is stored as {mil_location} data. "
            "They must be defined at the same mesh locations before direct "
            "subtraction is appropriate."
        )

    data = mesh.cell_data if st_location == "cell" else mesh.point_data

    da_st = safe_numeric_array(data[DA_ST_COLUMN])
    da_mil = safe_numeric_array(data[DA_MIL_COLUMN])

    if da_st.shape != da_mil.shape:
        raise ValueError(
            f"Array sizes differ: {DA_ST_COLUMN}={da_st.shape}, "
            f"{DA_MIL_COLUMN}={da_mil.shape}"
        )

    valid = np.isfinite(da_st) & np.isfinite(da_mil)

    da_diff = np.full(da_st.shape, np.nan, dtype=float)
    da_abs_diff = np.full(da_st.shape, np.nan, dtype=float)
    da_mean = np.full(da_st.shape, np.nan, dtype=float)
    da_agreement = np.full(da_st.shape, np.nan, dtype=float)

    da_diff[valid] = da_st[valid] - da_mil[valid]
    da_abs_diff[valid] = np.abs(da_diff[valid])
    da_mean[valid] = (da_st[valid] + da_mil[valid]) / 2.0

    # 1 = agreement within tolerance; 0 = disagreement.
    da_agreement[valid] = (
        da_abs_diff[valid] <= AGREEMENT_TOLERANCE
    ).astype(float)


    data["DA_diff"] = da_diff
    data["DA_abs_diff"] = da_abs_diff
    data["DA_mean_methods"] = da_mean
    data["DA_agreement"] = da_agreement



    # -------------------------------------------------------------------------
    # Linear-regression residuals: DA_MIL predicted from DA_ST
    # -------------------------------------------------------------------------

    da_mil_predicted = np.full(da_st.shape, np.nan, dtype=float)
    da_regression_residual = np.full(da_st.shape, np.nan, dtype=float)
    da_abs_regression_residual = np.full(da_st.shape, np.nan, dtype=float)

    if np.count_nonzero(valid) >= 2:

        slope, intercept = np.polyfit(
            da_st[valid],
            da_mil[valid],
            deg=1
        )

        da_mil_predicted[valid] = (
            intercept + slope * da_st[valid]
        )

        da_regression_residual[valid] = (
            da_mil[valid] - da_mil_predicted[valid]
        )

        da_abs_regression_residual[valid] = np.abs(
            da_regression_residual[valid]
        )

        residual_sum_squares = np.sum(
            da_regression_residual[valid] ** 2
        )

        total_sum_squares = np.sum(
            (da_mil[valid] - np.mean(da_mil[valid])) ** 2
        )

        regression_r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else np.nan
        )

        regression_rmse = np.sqrt(
            np.mean(da_regression_residual[valid] ** 2)
        )

        print("\nLinear regression: DA_MIL predicted from DA_ST")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  Slope: {slope:.6f}")
        print(f"  R-squared: {regression_r_squared:.6f}")
        print(f"  Residual RMSE: {regression_rmse:.6f}")

    else:
        print(
            "WARNING: fewer than two valid paired DA values; "
            "regression residuals could not be calculated."
        )

    data["DA_MIL_predicted_from_ST"] = da_mil_predicted
    data["DA_regression_residual"] = da_regression_residual
    data["DA_abs_regression_residual"] = da_abs_regression_residual

    # -------------------------------------------------------------------------
    # Quadratic-regression residuals: DA_MIL predicted from DA_ST
    # -------------------------------------------------------------------------

    da_mil_quadratic_predicted = np.full(
        da_st.shape,
        np.nan,
        dtype=float
    )

    da_quadratic_residual = np.full(
        da_st.shape,
        np.nan,
        dtype=float
    )

    da_abs_quadratic_residual = np.full(
        da_st.shape,
        np.nan,
        dtype=float
    )

    # A quadratic model has three coefficients, so require at least
    # three valid paired observations and variation in DA_ST.
    valid_count = np.count_nonzero(valid)
    valid_da_st = da_st[valid]
    valid_da_mil = da_mil[valid]

    if valid_count >= 3 and np.unique(valid_da_st).size >= 3:

        # np.polyfit returns coefficients in descending polynomial order:
        # quadratic_coefficient, linear_coefficient, intercept
        quadratic_coefficient, linear_coefficient, quadratic_intercept = (
            np.polyfit(
                valid_da_st,
                valid_da_mil,
                deg=2
            )
        )

        da_mil_quadratic_predicted[valid] = (
            quadratic_coefficient * da_st[valid] ** 2
            + linear_coefficient * da_st[valid]
            + quadratic_intercept
        )

        # Residual = observed DA_MIL - predicted DA_MIL
        da_quadratic_residual[valid] = (
            da_mil[valid]
            - da_mil_quadratic_predicted[valid]
        )

        da_abs_quadratic_residual[valid] = np.abs(
            da_quadratic_residual[valid]
        )

        # Quadratic model summary statistics
        quadratic_residual_sum_squares = np.sum(
            da_quadratic_residual[valid] ** 2
        )

        quadratic_total_sum_squares = np.sum(
            (
                da_mil[valid]
                - np.mean(da_mil[valid])
            ) ** 2
        )

        if quadratic_total_sum_squares > 0:
            quadratic_r_squared = (
                1.0
                - quadratic_residual_sum_squares
                / quadratic_total_sum_squares
            )
        else:
            quadratic_r_squared = np.nan

        quadratic_rmse = np.sqrt(
            np.mean(
                da_quadratic_residual[valid] ** 2
            )
        )

        print(
            "\nQuadratic regression: "
            "DA_MIL predicted from DA_ST"
        )
        print(
            "  Model: DA_MIL = "
            f"{quadratic_intercept:.6f} + "
            f"{linear_coefficient:.6f} * DA_ST + "
            f"{quadratic_coefficient:.6f} * DA_ST^2"
        )
        print(
            f"  Intercept: {quadratic_intercept:.6f}"
        )
        print(
            f"  Linear coefficient: {linear_coefficient:.6f}"
        )
        print(
            f"  Quadratic coefficient: "
            f"{quadratic_coefficient:.6f}"
        )
        print(
            f"  R-squared: {quadratic_r_squared:.6f}"
        )
        print(
            f"  Residual RMSE: {quadratic_rmse:.6f}"
        )

    else:
        print(
            "WARNING: quadratic regression could not be calculated. "
            "At least three valid paired values and three unique "
            "DA_ST values are required."
        )

    # Add quadratic-regression properties to the mesh.
    data["DA_MIL_predicted_quadratic_from_ST"] = (
        da_mil_quadratic_predicted
    )

    data["DA_quadratic_residual"] = (
        da_quadratic_residual
    )

    data["DA_abs_quadratic_residual"] = (
        da_abs_quadratic_residual
    )

    # Adjusted R-squared for the quadratic model.
    # n = number of observations
    # p = number of predictor terms, excluding the intercept
    n = valid_count
    p = 2

    if (
        np.isfinite(quadratic_r_squared)
        and n > p + 1
    ):
        quadratic_adjusted_r_squared = (
            1.0
            - (1.0 - quadratic_r_squared)
            * (n - 1)
            / (n - p - 1)
        )
    else:
        quadratic_adjusted_r_squared = np.nan

    print(
        f"  Adjusted R-squared: "
        f"{quadratic_adjusted_r_squared:.6f}"
    )

    OUTPUT_MESH.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(OUTPUT_MESH)

    print("\nSaved updated mesh:")
    print(OUTPUT_MESH)

    print(f"\nArrays were added to {st_location} data.")

    if np.any(valid):
        print("\nComparison summary:")
        print(f"  Valid mesh values: {int(np.count_nonzero(valid))}")
        print(f"  Mean signed difference: {float(np.mean(da_diff[valid])):.6f}")
        print(f"  Mean absolute difference: {float(np.mean(da_abs_diff[valid])):.6f}")
        print(f"  Maximum absolute difference: {float(np.max(da_abs_diff[valid])):.6f}")
        print(
            f"  Fraction agreeing within {AGREEMENT_TOLERANCE}: "
            f"{float(np.mean(da_agreement[valid])):.3f}"
        )
    else:
        print("WARNING: no mesh locations contained valid values for both methods.")


if __name__ == "__main__":
    main()