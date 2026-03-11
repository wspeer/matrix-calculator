"""Tests for the Matrix class and operations."""

import math
import pytest
from matrix import Matrix, MatrixError, identity, zeros, ones


def mat(data):
    return Matrix(data)


# --- Construction ---

def test_construction_basic():
    m = mat([[1, 2], [3, 4]])
    assert m.shape == (2, 2)
    assert m[0, 0] == 1.0
    assert m[1, 1] == 4.0


def test_construction_empty_raises():
    with pytest.raises(MatrixError):
        mat([])


def test_construction_jagged_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2], [3]])


# --- Addition ---

def test_add():
    A = mat([[1, 2], [3, 4]])
    B = mat([[5, 6], [7, 8]])
    assert A + B == mat([[6, 8], [10, 12]])


def test_add_shape_mismatch():
    A = mat([[1, 2]])
    B = mat([[1, 2], [3, 4]])
    with pytest.raises(MatrixError):
        A + B


# --- Subtraction ---

def test_sub():
    A = mat([[5, 6], [7, 8]])
    B = mat([[1, 2], [3, 4]])
    assert A - B == mat([[4, 4], [4, 4]])


# --- Scalar multiply ---

def test_scalar_mul_right():
    A = mat([[1, 2], [3, 4]])
    assert A * 2 == mat([[2, 4], [6, 8]])


def test_scalar_mul_left():
    A = mat([[1, 2], [3, 4]])
    assert 3 * A == mat([[3, 6], [9, 12]])


def test_neg():
    A = mat([[1, -2], [3, 0]])
    assert -A == mat([[-1, 2], [-3, 0]])


# --- Matrix multiply ---

def test_matmul_square():
    A = mat([[1, 2], [3, 4]])
    B = mat([[2, 0], [1, 2]])
    assert A * B == mat([[4, 4], [10, 8]])


def test_matmul_non_square():
    A = mat([[1, 2, 3], [4, 5, 6]])   # 2x3
    B = mat([[7, 8], [9, 10], [11, 12]])  # 3x2
    result = A * B
    assert result.shape == (2, 2)
    assert result == mat([[58, 64], [139, 154]])


def test_matmul_dimension_mismatch():
    A = mat([[1, 2]])
    B = mat([[1, 2]])
    with pytest.raises(MatrixError):
        A * B


# --- Transpose ---

def test_transpose_square():
    A = mat([[1, 2], [3, 4]])
    assert A.transpose() == mat([[1, 3], [2, 4]])


def test_transpose_rectangular():
    A = mat([[1, 2, 3], [4, 5, 6]])
    T = A.transpose()
    assert T.shape == (3, 2)
    assert T == mat([[1, 4], [2, 5], [3, 6]])


# --- Determinant ---

def test_det_1x1():
    assert mat([[5]]).determinant() == pytest.approx(5.0)


def test_det_2x2():
    A = mat([[1, 2], [3, 4]])
    assert A.determinant() == pytest.approx(-2.0)


def test_det_3x3():
    A = mat([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    assert A.determinant() == pytest.approx(4.0)


def test_det_singular():
    A = mat([[1, 2], [2, 4]])
    assert A.determinant() == pytest.approx(0.0, abs=1e-9)


def test_det_non_square_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2, 3]]).determinant()


# --- Inverse ---

def test_inverse_2x2():
    A = mat([[4, 7], [2, 6]])
    inv = A.inverse()
    product = A * inv
    I = identity(2)
    for r in range(2):
        for c in range(2):
            assert product[r, c] == pytest.approx(I[r, c], abs=1e-9)


def test_inverse_3x3():
    A = mat([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    inv = A.inverse()
    product = A * inv
    I = identity(3)
    for r in range(3):
        for c in range(3):
            assert product[r, c] == pytest.approx(I[r, c], abs=1e-9)


def test_inverse_singular_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2], [2, 4]]).inverse()


def test_inverse_non_square_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2, 3]]).inverse()


# --- Trace ---

def test_trace():
    A = mat([[1, 2], [3, 4]])
    assert A.trace() == pytest.approx(5.0)


def test_trace_non_square_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2, 3]]).trace()


# --- Rank ---

def test_rank_full():
    A = mat([[1, 0], [0, 1]])
    assert A.rank() == 2


def test_rank_deficient():
    A = mat([[1, 2], [2, 4]])
    assert A.rank() == 1


def test_rank_rectangular():
    A = mat([[1, 2, 3], [4, 5, 6]])
    assert A.rank() == 2


def test_rank_zero_matrix():
    A = zeros(3, 3)
    assert A.rank() == 0


# --- RREF ---

def test_rref_identity():
    A = mat([[2, 4], [1, 3]])
    R = A.rref()
    I = identity(2)
    for r in range(2):
        for c in range(2):
            assert R[r, c] == pytest.approx(I[r, c], abs=1e-9)


def test_rref_rank_deficient():
    A = mat([[1, 2, 3], [2, 4, 6]])
    R = A.rref()
    assert R[0, 0] == pytest.approx(1.0)
    assert R[1, 0] == pytest.approx(0.0)
    assert R[1, 1] == pytest.approx(0.0)
    assert R[1, 2] == pytest.approx(0.0)


# --- Power ---

def test_power_zero():
    A = mat([[2, 3], [1, 4]])
    assert A.power(0) == identity(2)


def test_power_one():
    A = mat([[2, 3], [1, 4]])
    assert A.power(1) == A


def test_power_two():
    A = mat([[1, 1], [0, 1]])
    assert A.power(2) == mat([[1, 2], [0, 1]])


def test_power_three():
    A = mat([[1, 1], [0, 1]])
    assert A.power(3) == mat([[1, 3], [0, 1]])


def test_power_non_square_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2, 3]]).power(2)


def test_power_negative_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2], [3, 4]]).power(-1)


# --- Eigenvalues ---

def test_eigenvalues_diagonal():
    A = mat([[3, 0], [0, 5]])
    evals = sorted(A.eigenvalues())
    assert evals[0] == pytest.approx(3.0, abs=1e-6)
    assert evals[1] == pytest.approx(5.0, abs=1e-6)


def test_eigenvalues_symmetric():
    A = mat([[2, 1], [1, 2]])
    evals = sorted(A.eigenvalues())
    assert evals[0] == pytest.approx(1.0, abs=1e-6)
    assert evals[1] == pytest.approx(3.0, abs=1e-6)


def test_eigenvalues_3x3():
    A = mat([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    evals = sorted(A.eigenvalues())
    assert evals[0] == pytest.approx(1.0, abs=1e-6)
    assert evals[1] == pytest.approx(2.0, abs=1e-6)
    assert evals[2] == pytest.approx(3.0, abs=1e-6)


def test_eigenvectors_returns_correct_count():
    A = mat([[4, 1], [2, 3]])
    evals, evecs = A.eigenvectors()
    assert len(evals) == 2
    assert len(evecs) == 2
    for vec in evecs:
        assert vec.shape == (2, 1)


def test_eigenvectors_verify_Av_eq_lv():
    """Verify that A*v = lambda*v for each eigenpair."""
    A = mat([[2, 1], [1, 2]])
    evals, evecs = A.eigenvectors()
    for lam, v in zip(evals, evecs):
        Av = A * v
        lv = lam * v
        for r in range(A.rows):
            assert Av[r, 0] == pytest.approx(lv[r, 0], abs=1e-4)


def test_eigenvalues_complex_rotation():
    """A 90-degree rotation matrix has eigenvalues +/- i."""
    A = mat([[0, -1], [1, 0]])
    evals = A.eigenvalues()
    # Should be complex: +i and -i
    assert all(isinstance(e, complex) for e in evals)
    imags = sorted([e.imag for e in evals])
    assert imags[0] == pytest.approx(-1.0, abs=1e-6)
    assert imags[1] == pytest.approx(1.0, abs=1e-6)
    assert all(abs(e.real) < 1e-6 for e in evals)


def test_eigenvalues_complex_3x3():
    """A 3x3 with one real and two complex eigenvalues."""
    # [[2, -1, 0], [1, 2, 0], [0, 0, 3]] has eigenvalues 2+i, 2-i, 3
    A = mat([[2, -1, 0], [1, 2, 0], [0, 0, 3]])
    evals = A.eigenvalues()
    # Sort: reals first, then by imaginary
    reals = [e for e in evals if not isinstance(e, complex) or abs(e.imag) < 1e-6]
    cpx = [e for e in evals if isinstance(e, complex) and abs(e.imag) > 1e-6]
    assert len(reals) == 1
    real_val = reals[0].real if isinstance(reals[0], complex) else reals[0]
    assert real_val == pytest.approx(3.0, abs=1e-6)
    assert len(cpx) == 2
    cpx_sorted = sorted(cpx, key=lambda e: e.imag)
    assert cpx_sorted[0].real == pytest.approx(2.0, abs=1e-6)
    assert cpx_sorted[0].imag == pytest.approx(-1.0, abs=1e-6)
    assert cpx_sorted[1].real == pytest.approx(2.0, abs=1e-6)
    assert cpx_sorted[1].imag == pytest.approx(1.0, abs=1e-6)


def test_eigenvectors_complex_verify():
    """Verify A*v = λ*v for complex eigenpairs."""
    A = mat([[0, -1], [1, 0]])
    evals, evecs = A.eigenvectors()
    for lam, vec in zip(evals, evecs):
        # A*v manually (vec is a list of complex numbers)
        assert isinstance(vec, list)
        n = len(vec)
        for r in range(n):
            Av = sum(complex(A[r, c]) * vec[c] for c in range(n))
            lv = lam * vec[r]
            assert abs(Av - lv) < 1e-4


def test_eigenvalues_non_square_raises():
    with pytest.raises(MatrixError):
        mat([[1, 2, 3]]).eigenvalues()


# --- Helpers ---

def test_identity():
    I = identity(3)
    assert I.shape == (3, 3)
    for r in range(3):
        for c in range(3):
            assert I[r, c] == (1.0 if r == c else 0.0)


def test_zeros():
    Z = zeros(2, 3)
    assert Z.shape == (2, 3)
    for r in range(2):
        for c in range(3):
            assert Z[r, c] == 0.0


def test_ones():
    O = ones(2, 3)
    assert O.shape == (2, 3)
    for r in range(2):
        for c in range(3):
            assert O[r, c] == 1.0
