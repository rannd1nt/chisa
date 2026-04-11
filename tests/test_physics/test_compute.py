"""Tests for phaethon.linalg and phaethon.random."""

import numpy as np
import numpy.ma as ma
import pytest

import phaethon as ptn
import phaethon.units as u
from phaethon.exceptions import AxiomViolationError


def test_native_scientific_compute():
    """
    Integration test for the native compute ecosystem:
    array creation, linear algebra synthesis, vector norms, and stochastic physics.
    """
    raw_matrix = [[1.0, 2.0], [3.0, 4.0]]

    arr_f32 = ptn.array(raw_matrix, u.Meter, dtype=np.float32, ndmin=3, order="F")
    assert arr_f32.shape == (1, 2, 2), "Array reshaping (ndmin) failed."
    assert arr_f32.mag.dtype == np.float32, "Dtype casting failed."
    assert arr_f32.mag.flags["F_CONTIGUOUS"] is True, "Fortran memory order was not preserved."
    assert arr_f32.dimension == "length", "Dimension resolution failed."

    massive_data = np.arange(1000, dtype=np.float64)
    arr_zero_copy = ptn.asarray(massive_data, u.Joule)
    assert np.shares_memory(massive_data, arr_zero_copy.mag), "asarray must not copy memory."
    assert arr_zero_copy.dimension == "energy", "Dimension resolution failed."

    masked_raw = ma.masked_array([10.0, -999.0, 30.0], mask=[0, 1, 0])
    safe_temp = ptn.asanyarray(masked_raw, u.Kelvin)
    assert isinstance(safe_temp.mag, ma.MaskedArray), "Masked array identity lost during initialization."
    assert safe_temp.mag.mask[1] == True, "Masked values were corrupted."

    A_mat = ptn.array([[4.0, 7.0], [2.0, 6.0]], u.Meter)
    A_inv = ptn.linalg.inv(A_mat)
    assert A_inv.dimension == "linear_attenuation", "Inverse matrix dimension must be 1/L."
    
    identity = A_mat @ A_inv
    assert identity.dimension == "dimensionless", "Matrix multiplied by its inverse must be dimensionless."
    assert np.allclose(identity.mag, np.eye(2)), "Inverse calculation yielded incorrect magnitudes."

    A_3x3 = ptn.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], u.Meter)
    det_vol = ptn.linalg.det(A_3x3)
    assert det_vol.dimension == "volume", "Determinant of 3x3 Length matrix must yield Volume."
    assert np.isclose(det_vol.mag, 1.0), "Determinant magnitude calculation failed."

    M_mass = ptn.array([[10.0, 2.0], [3.0, 5.0]], u.Kilogram)
    F_force = ptn.array([50.0, 25.0], u.Newton)
    accel_x = ptn.linalg.solve(M_mass, F_force)
    assert accel_x.dimension == "acceleration", "Solving M*a = F must yield Acceleration."
    assert isinstance(accel_x, u.MeterPerSecondSquared), "Incorrect base unit resolution for acceleration."

    F_mat = ptn.array([[50.0, 10.0], [10.0, 25.0]], u.Newton)
    F_vec = ptn.array([100.0, 50.0], u.Newton)
    ratio_x = ptn.linalg.solve(F_mat, F_vec)
    assert ratio_x.dimension == "dimensionless", "Solving Force/Force must yield dimensionless scalar."

    vec_v = ptn.array([3.0, 4.0], u.MeterPerSecond)
    mag_v = ptn.linalg.norm(vec_v)
    assert mag_v.dimension == "speed", "Vector norm must preserve the original dimension."
    assert np.isclose(mag_v.mag, 5.0), "L2 Norm calculation failed."

    mat_norm = ptn.linalg.norm(M_mass, ord="fro", keepdims=True)
    assert mat_norm.dimension == "mass", "Frobenius norm must preserve the original dimension."
    assert mat_norm.shape == (1, 1), "keepdims argument was ignored in linalg.norm."

    ptn.random.seed(42)
    rep_1 = ptn.random.uniform(0, 100, size=5, unit=u.Volt)
    ptn.random.seed(42)
    rep_2 = ptn.random.uniform(0, 100, size=5, unit=u.Volt)
    assert np.allclose(rep_1.mag, rep_2.mag), "Isolated RNG seed failed to produce deterministic results."
    assert rep_1.dimension == "electric_potential", "Unit injection failed in random.uniform."

    p_uni = ptn.random.uniform(low=10.5, high=20.5, size=(5, 5), unit=u.Pascal)
    assert p_uni.shape == (5, 5), "Random tensor shape mismatch."
    assert p_uni.dimension == "pressure", "Dimension mapping failed."
    assert np.all((p_uni.mag >= 10.5) & (p_uni.mag <= 20.5)), "Uniform distribution bounds violated."

    mass_norm = ptn.random.normal(loc=100.0, scale=2.5, size=10, unit="kg")
    assert mass_norm.dimension == "mass", "String alias mapping failed in random.normal."
    assert mass_norm.shape == (10,), "Output shape mismatch in normal distribution."

    t_half = ptn.random.exponential(scale=1.5, size=(3, 3, 3), unit=u.Second)
    assert t_half.dimension == "time", "Dimension mapping failed."
    assert np.all(t_half.mag >= 0), "Exponential distribution generated impossible negative values."

    bq_decay = ptn.random.poisson(lam=50.0, size=(100,), unit=u.Becquerel)
    assert bq_decay.dimension == "radioactivity", "Dimension mapping failed."
    assert np.issubdtype(bq_decay.mag.dtype, np.integer), "Poisson events must be discrete integers."

    spin_states = ptn.random.randint(-1, 2, size=(50,), unit=u.Dimensionless)
    assert np.issubdtype(spin_states.mag.dtype, np.integer), "Randint failed to produce integers."
    assert np.all((spin_states.mag >= -1) & (spin_states.mag < 2)), "Randint boundary violation."
    
    allowed_velocities = [300.0, 400.0, 500.0]
    gas_particles = ptn.random.choice(allowed_velocities, size=100, unit=u.MeterPerSecond)
    assert gas_particles.dimension == "speed", "Choice dimension mapping failed."
    assert np.all(np.isin(gas_particles.mag, allowed_velocities)), "Choice generated illegal states."

    ptn.random.seed(99)
    ordered_energy = ptn.array(np.arange(100.0), u.Joule)
    ordered_copy = ordered_energy.mag.copy()
    
    ptn.random.shuffle(ordered_energy)
    assert not np.array_equal(ordered_energy.mag, ordered_copy), "Shuffle failed to randomize array."
    assert np.array_equal(np.sort(ordered_energy.mag), ordered_copy), "Shuffle destroyed array elements."
    assert ordered_energy.dimension == "energy", "Shuffle stripped physical dimensions."

    original_force = ptn.array([10.0, 20.0, 30.0], u.Newton)
    permuted_force = ptn.random.permutation(original_force)
    assert np.array_equal(original_force.mag, [10.0, 20.0, 30.0]), "Permutation mutated original tensor."
    assert np.array_equal(np.sort(permuted_force.mag), [10.0, 20.0, 30.0]), "Permutation lost elements."
    assert permuted_force.dimension == "force", "Permutation stripped physical dimensions."

    idx_range = ptn.random.permutation(10)
    assert idx_range.dimension == "dimensionless", "Integer permutation must return Dimensionless unit."
    assert set(idx_range.mag) == set(range(10)), "Integer permutation failed to generate valid range."

    with pytest.raises(np.linalg.LinAlgError, match="square"):
        ptn.linalg.inv(ptn.array([[1, 2, 3], [4, 5, 6]], u.Meter))

    with pytest.raises(np.linalg.LinAlgError, match="at least two-dimensional"):
        ptn.linalg.inv(ptn.array([1, 2, 3], u.Meter))

    mat_db = ptn.array([[30, 30], [30, 30]], u.Decibel)
    with pytest.raises(AxiomViolationError, match="You cannot exponentiate"):
        ptn.linalg.det(mat_db)

    with pytest.raises(ValueError, match="physical unit must be specified"):
        ptn.random.uniform(size=5)
        
    with pytest.raises(TypeError, match="must be a Phaethon BaseUnit"):
        ptn.random.shuffle([1, 2, 3])