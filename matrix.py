"""Matrix class with core linear algebra operations."""

from fractions import Fraction


class MatrixError(Exception):
    pass


class Matrix:
    def __init__(self, data):
        """Create a matrix from a list of lists."""
        if not data or not data[0]:
            raise MatrixError("Matrix cannot be empty")
        rows = len(data)
        cols = len(data[0])
        if any(len(row) != cols for row in data):
            raise MatrixError("All rows must have the same length")
        self._data = [[float(x) for x in row] for row in data]
        self._rows = rows
        self._cols = cols

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def shape(self):
        return (self._rows, self._cols)

    def __getitem__(self, key):
        r, c = key
        return self._data[r][c]

    def __setitem__(self, key, value):
        r, c = key
        self._data[r][c] = float(value)

    def __repr__(self):
        col_widths = []
        for c in range(self._cols):
            width = max(len(self._fmt(self._data[r][c])) for r in range(self._rows))
            col_widths.append(width)
        lines = []
        for r in range(self._rows):
            row_str = "  ".join(
                self._fmt(self._data[r][c]).rjust(col_widths[c])
                for c in range(self._cols)
            )
            lines.append(f"[ {row_str} ]")
        return "\n".join(lines)

    def _fmt(self, value):
        if value == int(value):
            return str(int(value))
        return f"{value:.4g}"

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self.shape != other.shape:
            return False
        for r in range(self._rows):
            for c in range(self._cols):
                if abs(self._data[r][c] - other._data[r][c]) > 1e-9:
                    return False
        return True

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise MatrixError("Can only add Matrix to Matrix")
        if self.shape != other.shape:
            raise MatrixError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )
        result = [
            [self._data[r][c] + other._data[r][c] for c in range(self._cols)]
            for r in range(self._rows)
        ]
        return Matrix(result)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise MatrixError("Can only subtract Matrix from Matrix")
        if self.shape != other.shape:
            raise MatrixError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )
        result = [
            [self._data[r][c] - other._data[r][c] for c in range(self._cols)]
            for r in range(self._rows)
        ]
        return Matrix(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = [
                [self._data[r][c] * other for c in range(self._cols)]
                for r in range(self._rows)
            ]
            return Matrix(result)
        if isinstance(other, Matrix):
            if self._cols != other._rows:
                raise MatrixError(
                    f"Cannot multiply {self.shape} by {other.shape}: "
                    f"inner dimensions {self._cols} != {other._rows}"
                )
            result = []
            for r in range(self._rows):
                row = []
                for c in range(other._cols):
                    val = sum(
                        self._data[r][k] * other._data[k][c]
                        for k in range(self._cols)
                    )
                    row.append(val)
                result.append(row)
            return Matrix(result)
        raise MatrixError(f"Cannot multiply Matrix by {type(other)}")

    def __rmul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return self.__mul__(scalar)
        raise MatrixError(f"Cannot multiply {type(scalar)} by Matrix")

    def __neg__(self):
        return self * -1

    def transpose(self):
        """Return the transpose of this matrix."""
        result = [
            [self._data[r][c] for r in range(self._rows)]
            for c in range(self._cols)
        ]
        return Matrix(result)

    def is_square(self):
        return self._rows == self._cols

    def _copy_data(self):
        return [row[:] for row in self._data]

    def determinant(self):
        """Compute the determinant using Gaussian elimination."""
        if not self.is_square():
            raise MatrixError("Determinant requires a square matrix")
        n = self._rows
        data = self._copy_data()
        det = 1.0

        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if abs(data[row][col]) > 1e-12:
                    pivot_row = row
                    break
            if pivot_row is None:
                return 0.0
            if pivot_row != col:
                data[col], data[pivot_row] = data[pivot_row], data[col]
                det *= -1

            det *= data[col][col]
            pivot = data[col][col]
            for row in range(col + 1, n):
                factor = data[row][col] / pivot
                for k in range(col, n):
                    data[row][k] -= factor * data[col][k]

        return det

    def inverse(self):
        """Compute the matrix inverse using Gauss-Jordan elimination."""
        if not self.is_square():
            raise MatrixError("Inverse requires a square matrix")
        n = self._rows
        # Augment with identity
        aug = [self._data[r][:] + [1.0 if r == c else 0.0 for c in range(n)]
               for r in range(n)]

        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if abs(aug[row][col]) > 1e-12:
                    pivot_row = row
                    break
            if pivot_row is None:
                raise MatrixError("Matrix is singular and cannot be inverted")
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot = aug[col][col]
            aug[col] = [x / pivot for x in aug[col]]

            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][k] - factor * aug[col][k]
                                for k in range(2 * n)]

        result = [aug[r][n:] for r in range(n)]
        return Matrix(result)

    def trace(self):
        """Compute the trace (sum of diagonal elements)."""
        if not self.is_square():
            raise MatrixError("Trace requires a square matrix")
        return sum(self._data[i][i] for i in range(self._rows))

    def rref(self):
        """Return the reduced row echelon form (RREF)."""
        data = self._copy_data()
        rows, cols = self._rows, self._cols
        pivot_row = 0

        for col in range(cols):
            # Find pivot in this column
            found = None
            for row in range(pivot_row, rows):
                if abs(data[row][col]) > 1e-12:
                    found = row
                    break
            if found is None:
                continue

            data[pivot_row], data[found] = data[found], data[pivot_row]
            pivot = data[pivot_row][col]
            data[pivot_row] = [x / pivot for x in data[pivot_row]]

            for row in range(rows):
                if row != pivot_row:
                    factor = data[row][col]
                    data[row] = [data[row][k] - factor * data[pivot_row][k]
                                 for k in range(cols)]
            pivot_row += 1

        return Matrix(data)

    def rank(self):
        """Compute the rank of the matrix."""
        rref = self.rref()
        count = 0
        for r in range(rref.rows):
            if any(abs(rref[r, c]) > 1e-9 for c in range(rref.cols)):
                count += 1
        return count

    def power(self, n):
        """Raise the matrix to the nth power (n must be a non-negative integer)."""
        if not self.is_square():
            raise MatrixError("Matrix power requires a square matrix")
        if not isinstance(n, int) or n < 0:
            raise MatrixError("Exponent must be a non-negative integer")
        if n == 0:
            return identity(self._rows)
        result = identity(self._rows)
        base = Matrix(self._copy_data())
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        return result

    def eigenvalues(self, max_iter=1000):
        """Compute eigenvalues using the QR algorithm.

        Returns a list of eigenvalues (real or complex).
        Complex eigenvalues are returned as Python complex numbers.
        """
        if not self.is_square():
            raise MatrixError("Eigenvalues require a square matrix")
        import math, cmath
        n = self._rows
        if n == 1:
            return [self._data[0][0]]
        A = self._copy_data()

        for _ in range(max_iter):
            # QR decomposition via Gram-Schmidt
            Q = [[0.0] * n for _ in range(n)]
            R = [[0.0] * n for _ in range(n)]
            for j in range(n):
                v = [A[i][j] for i in range(n)]
                for i in range(j):
                    dot = sum(Q[k][i] * A[k][j] for k in range(n))
                    R[i][j] = dot
                    for k in range(n):
                        v[k] -= dot * Q[k][i]
                norm = math.sqrt(sum(x * x for x in v))
                R[j][j] = norm
                if norm < 1e-14:
                    norm = 1e-14
                for k in range(n):
                    Q[k][j] = v[k] / norm
            # A = R * Q
            new_A = [[0.0] * n for _ in range(n)]
            for r in range(n):
                for c in range(n):
                    new_A[r][c] = sum(R[r][k] * Q[k][c] for k in range(n))
            A = new_A
            # Check convergence: sub-diagonal elements should be zero
            # (except within 2x2 blocks for complex eigenvalues)
            converged = True
            i = 0
            while i < n - 1:
                if abs(A[i + 1][i]) > 1e-10:
                    # Check if this is part of a 2x2 block with complex eigenvalues
                    a, b = A[i][i], A[i][i + 1]
                    c, d = A[i + 1][i], A[i + 1][i + 1]
                    disc = (a + d) ** 2 - 4 * (a * d - b * c)
                    if disc < -1e-12:
                        # Complex eigenvalue pair - this block is converged
                        i += 2
                        continue
                    else:
                        converged = False
                        break
                i += 1
            if converged:
                break

        # Extract eigenvalues from quasi-upper-triangular form
        eigenvalues = []
        i = 0
        while i < n:
            if i == n - 1 or abs(A[i + 1][i]) <= 1e-10:
                # Real eigenvalue on diagonal
                eigenvalues.append(A[i][i])
                i += 1
            else:
                # 2x2 block: extract complex conjugate pair
                a, b = A[i][i], A[i][i + 1]
                c, d = A[i + 1][i], A[i + 1][i + 1]
                tr = a + d
                det_val = a * d - b * c
                disc = tr * tr - 4 * det_val
                if disc < 0:
                    sq = cmath.sqrt(disc)
                    eigenvalues.append((tr + sq) / 2)
                    eigenvalues.append((tr - sq) / 2)
                else:
                    sq = math.sqrt(disc)
                    eigenvalues.append((tr + sq) / 2)
                    eigenvalues.append((tr - sq) / 2)
                i += 2

        return eigenvalues

    def eigenvectors(self):
        """Compute eigenvalues and eigenvectors.

        Returns (eigenvalues, eigenvectors) where eigenvectors is a list.
        For real eigenvalues, eigenvectors are Matrix column vectors.
        For complex eigenvalues, eigenvectors are lists of complex numbers.
        """
        if not self.is_square():
            raise MatrixError("Eigenvectors require a square matrix")
        import math, cmath
        n = self._rows
        evals = self.eigenvalues()
        evecs = []

        for lam in evals:
            is_complex = isinstance(lam, complex) and abs(lam.imag) > 1e-10

            if is_complex:
                # Complex Gaussian elimination on (A - λI)
                aug = [
                    [complex(self._data[r][c]) - (lam if r == c else 0)
                     for c in range(n)]
                    for r in range(n)
                ]
                # Row reduce
                pivot_row = 0
                pivot_cols = []
                for col in range(n):
                    # Find pivot
                    found = None
                    for row in range(pivot_row, n):
                        if abs(aug[row][col]) > 1e-10:
                            found = row
                            break
                    if found is None:
                        continue
                    aug[found], aug[pivot_row] = aug[pivot_row], aug[found]
                    p = aug[pivot_row][col]
                    aug[pivot_row] = [x / p for x in aug[pivot_row]]
                    for row in range(n):
                        if row != pivot_row and abs(aug[row][col]) > 1e-10:
                            f = aug[row][col]
                            aug[row] = [aug[row][k] - f * aug[pivot_row][k]
                                        for k in range(n)]
                    pivot_cols.append(col)
                    pivot_row += 1

                # Find a free variable
                free_col = None
                for c in range(n - 1, -1, -1):
                    if c not in pivot_cols:
                        free_col = c
                        break

                vec = [complex(0)] * n
                if free_col is not None:
                    vec[free_col] = complex(1)
                    for r in range(len(pivot_cols)):
                        vec[pivot_cols[r]] = -aug[r][free_col]
                else:
                    vec[n - 1] = complex(1)

                # Normalize
                norm = math.sqrt(sum(abs(x) ** 2 for x in vec))
                if norm > 1e-14:
                    vec = [x / norm for x in vec]
                evecs.append(vec)
            else:
                real_lam = lam.real if isinstance(lam, complex) else lam
                # Solve (A - lambda*I)x = 0 via RREF
                shifted = [
                    [self._data[r][c] - (real_lam if r == c else 0.0)
                     for c in range(n)]
                    for r in range(n)
                ]
                rref_data = Matrix(shifted).rref()

                # Find free variable (last one without a pivot)
                pivot_cols = set()
                for r in range(n):
                    for c in range(n):
                        if abs(rref_data[r, c]) > 1e-9:
                            pivot_cols.add(c)
                            break

                free_col = None
                for c in range(n - 1, -1, -1):
                    if c not in pivot_cols:
                        free_col = c
                        break

                vec = [0.0] * n
                if free_col is not None:
                    vec[free_col] = 1.0
                    for r in range(n):
                        for c in range(n):
                            if abs(rref_data[r, c]) > 1e-9:
                                vec[c] = -rref_data[r, free_col]
                                break
                else:
                    # Fallback: use inverse iteration
                    shift = 1e-10
                    shifted_inv = Matrix([
                        [self._data[r][c] - (real_lam - shift if r == c else 0.0)
                         for c in range(n)]
                        for r in range(n)
                    ])
                    try:
                        inv = shifted_inv.inverse()
                        vec = [1.0 / math.sqrt(n)] * n
                        for _ in range(50):
                            new_vec = [
                                sum(inv[r, c] * vec[c] for c in range(n))
                                for r in range(n)
                            ]
                            norm = math.sqrt(sum(x * x for x in new_vec))
                            if norm < 1e-14:
                                break
                            vec = [x / norm for x in new_vec]
                    except MatrixError:
                        vec = [0.0] * n
                        vec[0] = 1.0

                # Normalize
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 1e-14:
                    vec = [x / norm for x in vec]
                evecs.append(Matrix([[x] for x in vec]))

        return evals, evecs

    def to_list(self):
        """Return the matrix data as a list of lists."""
        return self._copy_data()


def identity(n):
    """Return the n x n identity matrix."""
    return Matrix([[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)])


def zeros(rows, cols):
    """Return a rows x cols matrix of zeros."""
    return Matrix([[0.0] * cols for _ in range(rows)])


def ones(rows, cols):
    """Return a rows x cols matrix of ones."""
    return Matrix([[1.0] * cols for _ in range(rows)])
