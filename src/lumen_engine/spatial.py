"""Calibrated point targeting for moving-head fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import math

from lumen_engine.geometry import (
    angular_distance_deg,
    apply_matrix,
    apply_transpose,
    rotation_matrix_xyz,
)
from lumen_engine.models import FixtureCalibration, FixturePatch, Vec3


@dataclass(frozen=True, slots=True)
class TargetingSolution:
    fixture_id: str
    target: Vec3
    pan_deg: float
    tilt_deg: float
    distance_m: float
    movement_cost_deg: float
    aim_error_deg: float
    branch: str


class UnreachableTargetError(ValueError):
    pass


class SpatialTargetingEngine:
    """Resolve world-space targets into calibrated mechanical pan and tilt."""

    def solve(
        self,
        fixture: FixturePatch,
        target: Vec3,
        previous_pan_deg: float | None = None,
        previous_tilt_deg: float | None = None,
    ) -> TargetingSolution:
        world_direction = target - fixture.position_m
        distance = world_direction.norm()
        if distance <= 1e-6:
            raise UnreachableTargetError(
                f"Target coincides with fixture {fixture.fixture_id}"
            )

        rotation = rotation_matrix_xyz(fixture.housing_rotation)
        local_direction = apply_transpose(rotation, world_direction.normalized())
        azimuth = math.degrees(math.atan2(local_direction.y, local_direction.x))
        elevation = math.degrees(
            math.atan2(
                local_direction.z,
                math.hypot(local_direction.x, local_direction.y),
            )
        )

        calibration = fixture.calibration
        candidates: list[tuple[float, float, str]] = []
        for turns in range(-3, 4):
            candidates.append(
                self._mechanical_candidate(
                    calibration,
                    azimuth + turns * 360.0,
                    elevation,
                    f"direct/{turns:+d}",
                )
            )
            candidates.append(
                self._mechanical_candidate(
                    calibration,
                    azimuth + 180.0 + turns * 360.0,
                    180.0 - elevation,
                    f"flipped/{turns:+d}",
                )
            )

        valid = [
            candidate
            for candidate in candidates
            if self._within_limits(calibration, candidate[0], candidate[1])
        ]
        if not valid:
            raise UnreachableTargetError(
                f"{fixture.name} cannot reach target {target.as_tuple()} within "
                f"pan [{calibration.pan_min_deg}, {calibration.pan_max_deg}] and "
                f"tilt [{calibration.tilt_min_deg}, {calibration.tilt_max_deg}]"
            )

        def score(candidate: tuple[float, float, str]) -> float:
            pan, tilt, _ = candidate
            if previous_pan_deg is None or previous_tilt_deg is None:
                pan_center = (
                    calibration.pan_min_deg + calibration.pan_max_deg
                ) / 2.0
                tilt_center = (
                    calibration.tilt_min_deg + calibration.tilt_max_deg
                ) / 2.0
                return 0.2 * abs(pan - pan_center) + abs(tilt - tilt_center)
            return angular_distance_deg(pan, previous_pan_deg) + angular_distance_deg(
                tilt, previous_tilt_deg
            )

        pan, tilt, branch = min(valid, key=score)
        movement_cost = (
            0.0
            if previous_pan_deg is None or previous_tilt_deg is None
            else angular_distance_deg(pan, previous_pan_deg)
            + angular_distance_deg(tilt, previous_tilt_deg)
        )
        predicted = self.direction_for_angles(fixture, pan, tilt)
        desired = world_direction.normalized()
        dot = max(-1.0, min(1.0, predicted.dot(desired)))
        error = math.degrees(math.acos(dot))

        return TargetingSolution(
            fixture_id=fixture.fixture_id,
            target=target,
            pan_deg=pan,
            tilt_deg=tilt,
            distance_m=distance,
            movement_cost_deg=movement_cost,
            aim_error_deg=error,
            branch=branch,
        )

    def direction_for_angles(
        self, fixture: FixturePatch, pan_deg: float, tilt_deg: float
    ) -> Vec3:
        calibration = fixture.calibration
        canonical_pan = (
            pan_deg - calibration.pan_offset_deg
        ) / calibration.pan_direction
        canonical_tilt = (
            tilt_deg - calibration.tilt_offset_deg
        ) / calibration.tilt_direction
        pan_radians = math.radians(canonical_pan)
        tilt_radians = math.radians(canonical_tilt)
        local = Vec3(
            math.cos(tilt_radians) * math.cos(pan_radians),
            math.cos(tilt_radians) * math.sin(pan_radians),
            math.sin(tilt_radians),
        )
        return apply_matrix(rotation_matrix_xyz(fixture.housing_rotation), local)

    @staticmethod
    def _mechanical_candidate(
        calibration: FixtureCalibration,
        canonical_pan: float,
        canonical_tilt: float,
        branch: str,
    ) -> tuple[float, float, str]:
        return (
            calibration.pan_offset_deg
            + calibration.pan_direction * canonical_pan,
            calibration.tilt_offset_deg
            + calibration.tilt_direction * canonical_tilt,
            branch,
        )

    @staticmethod
    def _within_limits(
        calibration: FixtureCalibration, pan: float, tilt: float
    ) -> bool:
        epsilon = 1e-9
        return (
            calibration.pan_min_deg - epsilon
            <= pan
            <= calibration.pan_max_deg + epsilon
            and calibration.tilt_min_deg - epsilon
            <= tilt
            <= calibration.tilt_max_deg + epsilon
        )

