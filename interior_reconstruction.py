"""
Interior object candidate extraction for Cloud2BIM.

This is a geometry-first recovery stage, not a trained semantic classifier.
It preserves uncertain clusters as "unclassified_object" rather than inventing
chair/table labels. Coordinates and dimensions are in the input scan's units
(expected metres by the upstream Cloud2BIM pipeline).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import open3d as o3d


def _remove_dominant_planes(pcd, distance, iterations=12, min_inliers=300):
    """Remove only building-scale planes, retaining smaller planar furniture surfaces."""
    points = np.asarray(pcd.points)
    searchable = pcd
    searchable_indices = np.arange(len(points))
    keep = np.ones(len(points), dtype=bool)
    for _ in range(iterations):
        if len(searchable.points) < min_inliers * 2:
            break
        model, inliers = searchable.segment_plane(
            distance_threshold=distance, ransac_n=3, num_iterations=800
        )
        if len(inliers) < min_inliers:
            break
        plane_points = np.asarray(searchable.points)[inliers]
        normal = np.asarray(model[:3], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-12)
        reference = np.array([0.0, 0.0, 1.0])
        axis_x = np.cross(normal, reference)
        if np.linalg.norm(axis_x) < 1e-6:
            axis_x = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        axis_x /= max(np.linalg.norm(axis_x), 1e-12)
        axis_y = np.cross(normal, axis_x)
        projected = plane_points @ np.column_stack((axis_x, axis_y))
        spans = np.ptp(projected, axis=0)
        if float(np.prod(spans)) >= 1.5:
            keep[searchable_indices[inliers]] = False
        keep_search = np.ones(len(searchable_indices), dtype=bool)
        keep_search[inliers] = False
        searchable_indices = searchable_indices[keep_search]
        searchable = searchable.select_by_index(inliers, invert=True)
    remaining_indices = np.flatnonzero(keep)
    return pcd.select_by_index(remaining_indices.tolist()), remaining_indices


def _voxelize_with_trace(points, voxel_size, max_points):
    """Create density-bounded voxel representatives and retain original point indices."""
    origin = points.min(axis=0)
    current_size = float(voxel_size)
    while True:
        keys = np.floor((points - origin) / current_size).astype(np.int64)
        _, first_indices, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_index=True, return_counts=True
        )
        if len(first_indices) <= max_points:
            representatives = np.zeros((len(first_indices), 3), dtype=np.float64)
            for axis in range(3):
                representatives[:, axis] = np.bincount(
                    inverse, weights=points[:, axis], minlength=len(first_indices)
                ) / counts
            return representatives, first_indices, inverse, current_size
        current_size *= 1.25


def analyze_point_cloud_density(points_xyz, rgb=None):
    """Estimate local density and preserve interior geometry by avoiding one-size-fits-all voxel settings."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return {
            "point_count": 0,
            "bounds": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "voxel_sizes_m": {"global": 0.0, "walls": 0.0, "openings": 0.0, "furniture": 0.0},
            "occupancy": {"interior_ratio": 0.0, "floor_band": [0.0, 0.0], "ceiling_band": [0.0, 0.0]},
        }
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return {
            "point_count": 0,
            "bounds": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "voxel_sizes_m": {"global": 0.0, "walls": 0.0, "openings": 0.0, "furniture": 0.0},
            "occupancy": {"interior_ratio": 0.0, "floor_band": [0.0, 0.0], "ceiling_band": [0.0, 0.0]},
        }

    x_min, y_min, z_min = finite.min(axis=0)
    x_max, y_max, z_max = finite.max(axis=0)
    z_range = max(z_max - z_min, 1e-6)
    z_hist, z_edges = np.histogram(finite[:, 2], bins=max(10, min(30, int(z_range / 0.05))), range=(z_min, z_max))
    floor_band = [float(z_edges[np.argmax(z_hist)]), float(z_edges[min(np.argmax(z_hist) + 1, len(z_edges)-1)])]
    upper_band = [float(z_edges[max(0, len(z_hist) - 1 - np.argmax(z_hist[::-1]))]), float(z_edges[-1])]

    xy_extent = max((x_max - x_min), (y_max - y_min), 1e-6)
    base_density = len(finite) / max(xy_extent ** 2, 1e-6)
    global_voxel = max(0.02, min(0.12, 0.25 * (1.0 / np.sqrt(max(base_density, 1e-6)))))

    return {
        "point_count": int(len(finite)),
        "bounds": {"x": [float(x_min), float(x_max)], "y": [float(y_min), float(y_max)], "z": [float(z_min), float(z_max)]},
        "points_per_m2": float(base_density),
        "voxel_sizes_m": {
            "global": float(global_voxel),
            "walls": float(max(0.025, global_voxel * 0.75)),
            "openings": float(max(0.015, global_voxel * 0.35)),
            "furniture": float(max(0.01, global_voxel * 0.25)),
        },
        "occupancy": {
            "interior_ratio": float(np.clip(len(finite) / max(len(points), 1), 0.0, 1.0)),
            "floor_band": [float(floor_band[0]), float(floor_band[1])],
            "ceiling_band": [float(upper_band[0]), float(upper_band[1])],
        },
        "rgb_preserved": bool(rgb is not None and np.asarray(rgb).size > 0),
    }


def build_room_partition_summary(points_xyz):
    """Return simple room-region candidates keeping interior volumes distinct without a fragile global voxel assumption."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return {"room_count": 0, "regions": [], "interior_regions": []}

    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return {"room_count": 0, "regions": [], "interior_regions": []}

    if len(finite) <= 2:
        centroid = finite.mean(axis=0)
        return {"room_count": 1, "regions": [{"label": 0, "point_count": int(len(finite)), "centroid": [float(v) for v in centroid]}], "interior_regions": [0]}

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(finite))
    voxel_size = max(0.08, min(0.35, float(np.std(finite[:, 2]) * 0.5 + 0.05)))
    downsampled = pcd.voxel_down_sample(voxel_size)
    if len(downsampled.points) == 0:
        return {"room_count": 1, "regions": [{"label": 0, "point_count": int(len(finite)), "centroid": [float(v) for v in finite.mean(axis=0)]}], "interior_regions": [0]}

    labels = np.asarray(downsampled.cluster_dbscan(eps=max(0.18, voxel_size * 2.0), min_points=max(6, int(len(downsampled.points) * 0.005)), print_progress=False))
    regions = []
    for label in sorted(set(labels.tolist()) - {-1}):
        idx = np.where(labels == label)[0]
        if len(idx) < 1:
            continue
        centroid = np.asarray(downsampled.points)[idx].mean(axis=0)
        regions.append({"label": int(label), "point_count": int(len(idx)), "centroid": [float(v) for v in centroid]})

    return {"room_count": len(regions), "regions": regions, "interior_regions": list(range(len(regions))) }


def extract_interior_object_candidates(points_xyz, output_dir, voxel_size=0.035,
                                        cluster_eps=0.12, min_points=18,
                                        max_points=250000, points_rgb=None):
    """
    Find non-planar interior object candidates and write a reviewable JSON report.
    Does not claim object-level semantic accuracy or create false BIM families.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_points = np.asarray(points_xyz, dtype=np.float64)
    finite_mask = np.isfinite(raw_points).all(axis=1)
    points = raw_points[finite_mask]
    original_indices = np.flatnonzero(finite_mask)
    if points_rgb is None:
        colors = None
    else:
        colors = np.asarray(points_rgb)
        if colors.ndim != 2 or colors.shape[0] != len(raw_points):
            raise ValueError("RGB values must have one row per input point.")
        colors = colors[finite_mask]
    report = {
        "stage": "interior_object_candidate_detection",
        "units": "metres",
        "method": "adaptive traced voxelization + building-plane removal + density-aware DBSCAN",
        "semantic_status": "geometric candidates only; not verified object classes",
        "input_points": int(len(points)),
        "parameters": {
            "requested_voxel_size_m": float(voxel_size),
            "cluster_eps_m": float(cluster_eps),
            "min_cluster_points": int(min_points),
            "max_analysis_points": int(max_points),
        },
        "rgb_preserved": bool(colors is not None and colors.ndim == 2 and colors.shape[1] >= 3),
        "candidates": [],
        "limitations": [
            "A geometric cluster is not proof of a chair, table, cabinet, or other object.",
            "Objects touching walls/floors may merge with architectural surfaces.",
            "Occluded or absent objects cannot be reconstructed from missing points.",
            "IFC candidate boxes are review envelopes; they are not fitted furniture families.",
            "The bundled segmentation research model has no deployed inference checkpoint and is trained on synthetic buildings."
        ],
    }
    if len(points) < max(min_points, 3):
        (output_dir / "interior_candidates.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report

    representatives, source_indices, _, actual_voxel_size = _voxelize_with_trace(
        points, voxel_size, max_points
    )
    report["analysis_voxel_size_m"] = float(actual_voxel_size)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(representatives))
    residual, residual_source_rows = _remove_dominant_planes(
        pcd, distance=max(0.025, actual_voxel_size * 1.25)
    )
    if len(residual.points) == 0:
        (output_dir / "interior_candidates.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report

    eps = max(float(cluster_eps), float(actual_voxel_size) * 2.5)
    labels = np.asarray(residual.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=False))
    xyz = np.asarray(residual.points)
    for label in sorted(set(labels.tolist()) - {-1}):
        cluster = xyz[labels == label]
        if len(cluster) < min_points:
            continue
        lo, hi = cluster.min(axis=0), cluster.max(axis=0)
        dims = hi - lo
        height, width, depth = float(dims[2]), float(max(dims[0], dims[1])), float(min(dims[0], dims[1]))
        volume = float(np.prod(dims))
        # Weak geometric priors only: candidates are deliberately not named chairs/tables.
        plausible_furnishing = (
            0.12 <= height <= 2.8 and 0.12 <= width <= 4.5
            and 0.08 <= depth <= 3.5 and volume >= 0.003
        )
        report["candidates"].append({
            "candidate_id": f"OBJ-{int(label)+1:05d}",
            "suggested_ifc_class": "IfcBuildingElementProxy",
            "semantic_label": "furnishing_candidate" if plausible_furnishing else "unclassified_object",
            "semantic_confidence": None,
            "geometry_confidence": "medium" if len(cluster) >= 80 else "low",
            "reconstruction_status": "observed_points_with_inferred_bbox_envelope",
            "point_count": int(len(cluster)),
            "bbox_min": [float(v) for v in lo],
            "bbox_max": [float(v) for v in hi],
            "dimensions_xyz": [float(v) for v in dims],
            "centroid": [float(v) for v in cluster.mean(axis=0)],
            "bbox_volume_m3": volume,
            "source_point_indices": [
                int(original_indices[source_indices[residual_source_rows[index]]])
                for index in np.flatnonzero(labels == label)
            ],
            "requires_review": True,
        })
    report["candidate_count"] = len(report["candidates"])
    report["furnishing_candidate_count"] = sum(
        c["semantic_label"] == "furnishing_candidate" for c in report["candidates"])
    (output_dir / "interior_candidates.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report
