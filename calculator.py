"""Interactive matrix calculator CLI."""

import sys
from matrix import Matrix, MatrixError, identity, zeros, ones


def parse_matrix(text):
    """Parse a matrix from text like '1 2; 3 4' or '[[1,2],[3,4]]'."""
    text = text.strip()
    # Handle nested list format [[1,2],[3,4]]
    if text.startswith("[["):
        text = text.strip("[]")
        rows = [r.strip().strip("[]") for r in text.split("],[")]
        data = [[float(x) for x in row.split(",")] for row in rows]
        return Matrix(data)
    # Handle semicolon-separated rows: "1 2; 3 4"
    rows = [r.strip() for r in text.split(";")]
    data = [[float(x) for x in row.split()] for row in rows if row]
    return Matrix(data)


def prompt_matrix(name=""):
    """Interactively prompt the user to enter a matrix."""
    label = f"Enter matrix {name}" if name else "Enter matrix"
    print(f"{label} (rows separated by ';', e.g. '1 2; 3 4'):")
    line = input("> ").strip()
    return parse_matrix(line)


MENU = """
Matrix Calculator
=================
Operations:
  1  Add two matrices          (A + B)
  2  Subtract two matrices     (A - B)
  3  Multiply two matrices     (A * B)
  4  Scalar multiply           (c * A)
  5  Transpose                 (A^T)
  6  Determinant               (det A)
  7  Inverse                   (A^-1)
  8  Trace                     (tr A)
  9  Rank                      (rank A)
  10 Matrix power              (A^n)
  11 Reduced row echelon form  (RREF)
  12 Identity matrix
  13 Eigenvalues & eigenvectors
  0  Quit
"""


def run_calculator():
    print(MENU)
    while True:
        try:
            choice = input("Select operation: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if choice == "0":
            print("Goodbye!")
            break

        try:
            if choice == "1":
                A = prompt_matrix("A")
                B = prompt_matrix("B")
                result = A + B
                print(f"\nA + B =\n{result}\n")

            elif choice == "2":
                A = prompt_matrix("A")
                B = prompt_matrix("B")
                result = A - B
                print(f"\nA - B =\n{result}\n")

            elif choice == "3":
                A = prompt_matrix("A")
                B = prompt_matrix("B")
                result = A * B
                print(f"\nA * B =\n{result}\n")

            elif choice == "4":
                try:
                    c = float(input("Enter scalar c: ").strip())
                except ValueError:
                    print("Invalid scalar.\n")
                    continue
                A = prompt_matrix("A")
                result = c * A
                print(f"\n{c} * A =\n{result}\n")

            elif choice == "5":
                A = prompt_matrix("A")
                result = A.transpose()
                print(f"\nA^T =\n{result}\n")

            elif choice == "6":
                A = prompt_matrix("A")
                det = A.determinant()
                print(f"\ndet(A) = {det:.6g}\n")

            elif choice == "7":
                A = prompt_matrix("A")
                result = A.inverse()
                print(f"\nA^-1 =\n{result}\n")

            elif choice == "8":
                A = prompt_matrix("A")
                t = A.trace()
                print(f"\ntr(A) = {t:.6g}\n")

            elif choice == "9":
                A = prompt_matrix("A")
                r = A.rank()
                print(f"\nrank(A) = {r}\n")

            elif choice == "10":
                A = prompt_matrix("A")
                try:
                    n = int(input("Enter exponent n (non-negative integer): ").strip())
                except ValueError:
                    print("Invalid exponent.\n")
                    continue
                result = A.power(n)
                print(f"\nA^{n} =\n{result}\n")

            elif choice == "11":
                A = prompt_matrix("A")
                result = A.rref()
                print(f"\nRREF(A) =\n{result}\n")

            elif choice == "12":
                try:
                    n = int(input("Enter size n for I_n: ").strip())
                except ValueError:
                    print("Invalid size.\n")
                    continue
                result = identity(n)
                print(f"\nI_{n} =\n{result}\n")

            elif choice == "13":
                A = prompt_matrix("A")
                evals, evecs = A.eigenvectors()
                print("\nEigenvalues and Eigenvectors:")
                for i, (val, vec) in enumerate(zip(evals, evecs)):
                    print(f"  λ{i+1} = {val:.6g}")
                    print(f"  v{i+1} =\n{vec}\n")
                print(f"\nI_{n} =\n{result}\n")

            else:
                print("Unknown option. Please enter a number from the menu.\n")

        except MatrixError as e:
            print(f"\nError: {e}\n")
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")


if __name__ == "__main__":
    run_calculator()
