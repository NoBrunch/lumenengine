"""Small dependency-free 3D rotation helpers."""

from __future__ import annotations

import math

from lumen_engine.models import EulerXYZ, Vec3

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _multiply(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def rotation_matrix_xyz(rotation: EulerXYZ) -> Matrix3:
    """Return intrinsic XYZ rotation as Rx * Ry * Rz for column vectors."""

    x = math.radians(rotation.x_deg)
    y = math.radians(rotation.y_deg)
    z = math.radians(rotation.z_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    rx: Matrix3 = ((1, 0, 0), (0, cx, -sx), (0, sx, cx))
    ry: Matrix3 = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    rz: Matrix3 = ((cz, -sz, 0), (sz, cz, 0), (0, 0, 1))
    return _multiply(_multiply(rx, ry), rz)


def apply_matrix(matrix: Matrix3, vector: Vec3) -> Vec3:
    return Vec3(
        matrix[0][0] * vector.x
        + matrix[0][1] * vector.y
        + matrix[0][2] * vector.z,
        matrix[1][0] * vector.x
        + matrix[1][1] * vector.y
        + matrix[1][2] * vector.z,
        matrix[2][0] * vector.x
        + matrix[2][1] * vector.y
        + matrix[2][2] * vector.z,
    )


def apply_transpose(matrix: Matrix3, vector: Vec3) -> Vec3:
    """Apply the inverse of an orthonormal rotation matrix."""

    return Vec3(
        matrix[0][0] * vector.x
        + matrix[1][0] * vector.y
        + matrix[2][0] * vector.z,
        matrix[0][1] * vector.x
        + matrix[1][1] * vector.y
        + matrix[2][1] * vector.z,
        matrix[0][2] * vector.x
        + matrix[1][2] * vector.y
        + matrix[2][2] * vector.z,
    )


def angular_distance_deg(a: float, b: float) -> float:
    """Mechanical travel distance; angles are intentionally not wrapped."""

    return abs(a - b)

