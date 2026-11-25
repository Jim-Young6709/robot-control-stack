"""Depth preprocessing for the DRP depth / pointmap checkpoints, on the camera frames RCS delivers.

The checkpoints learned from renders that contain only the table, its objects and the arm (a ray that hits nothing is
stored as 0) with a 0.01-3 m clipping range. A real camera also sees the floor, the walls and people. `DepthMask`
reproduces the renders without extrinsics: a range gate (always on) and, optionally, a mask relative to the table
plane, which is fitted by RANSAC in each camera's own frame.

Everything is metres on the colour pixel grid, 0 = no return.
"""

from dataclasses import dataclass, field

import numpy as np

NEAR_M, FAR_M = 0.01, 3.0
#: footprint lookup grid: cell size and how far past the table outline it reaches [m]
_CELL_M, _REACH_M = 0.01, 1.0


def unproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """HxW z-depth -> HxWx3 points in the OpenCV camera frame (x right, y down, z forward)."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth_m.astype(np.float32)
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=-1)


@dataclass
class TablePlane:
    """normal . p + offset = height above the table [m] (positive on the camera's side), plus the table's visible
    outline in plane coordinates (`basis` rows) as a lookup grid of the distance outside it."""

    normal: np.ndarray
    offset: float
    inlier_frac: float
    far_edge_m: float
    basis: np.ndarray
    grid_origin: np.ndarray
    outside_grid: np.ndarray

    def outside_distance(self, pts: np.ndarray) -> np.ndarray:
        idx = np.floor((pts @ self.basis.T - self.grid_origin) / _CELL_M).astype(np.int64)
        h, w = self.outside_grid.shape
        ok = (idx[:, 0] >= 0) & (idx[:, 0] < h) & (idx[:, 1] >= 0) & (idx[:, 1] < w)
        out = np.full(len(pts), np.inf, dtype=np.float32)
        out[ok] = self.outside_grid[idx[ok, 0], idx[ok, 1]]
        return out

    def describe(self) -> str:
        """With the renders' nominal values for comparison (eye 0.9-1.4 m above the table, ~27 deg down)."""
        pitch = float(np.degrees(np.arccos(np.clip(-self.normal[1], -1.0, 1.0))))
        return (f"table {self.inlier_frac * 100:.0f}% of px | camera {self.offset:.2f} m above it (sim 0.9-1.4), "
                f"pitched {pitch:.0f} deg down (sim ~27) | far edge ~{self.far_edge_m:.2f} m")


def fit_table_plane(depth_m: np.ndarray, K: np.ndarray, iters: int = 400, tol_m: float = 0.012,
                    seed: int = 0) -> "TablePlane | None":
    """Largest plane in view that faces up in the image (the table: the cameras look down at it), or None."""
    from scipy import ndimage
    from scipy.spatial import ConvexHull

    z = depth_m.reshape(-1)
    pts = unproject(depth_m, K).reshape(-1, 3)[(z > 0.3) & (z < FAR_M)]
    if len(pts) < 500:
        return None
    rng = np.random.default_rng(seed)
    sample = pts[rng.choice(len(pts), size=min(6000, len(pts)), replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        p0, p1, p2 = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n = -n if n @ p0 > 0 else n  # towards the camera at the origin
        if -n[1] < 0.3:  # walls and box sides do not face up in the image
            continue
        d = -float(n @ p0)
        count = int(np.count_nonzero(np.abs(sample @ n + d) < tol_m))
        if count > best_n:
            best, best_n = (n, d), count
    if best is None:
        return None
    inl = sample[np.abs(sample @ best[0] + best[1]) < tol_m]
    centroid = inl.mean(axis=0)
    n = np.linalg.svd(inl - centroid)[2][-1]
    n = -n if n @ centroid > 0 else n
    d = -float(n @ centroid)
    on_plane = pts[np.abs(pts @ n + d) < tol_m]
    if -n[1] < 0.3 or len(on_plane) < 0.15 * len(pts):
        return None

    # outline: drop thin lines of inliers (walls, people crossing table height), keep the large blobs, convex hull
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    basis = np.stack([e1, np.cross(n, e1)])
    uv = on_plane @ basis.T
    cell = np.floor((uv - uv.min(axis=0)) / 0.02).astype(int)
    occ = np.zeros(cell.max(axis=0) + 1, dtype=bool)
    occ[cell[:, 0], cell[:, 1]] = True
    occ = ndimage.binary_opening(occ, structure=np.ones((3, 3), bool))
    labels, count = ndimage.label(occ)
    if count:
        sizes = ndimage.sum(occ, labels, range(1, count + 1))
        keep = np.isin(labels[cell[:, 0], cell[:, 1]], np.flatnonzero(sizes >= 0.1 * sizes.max()) + 1)
        if np.count_nonzero(keep) >= 3:
            uv, on_plane = uv[keep], on_plane[keep]
    hull = uv[ConvexHull(uv).vertices]  # counter-clockwise
    origin = hull.min(axis=0) - _REACH_M
    size = np.ceil((hull.max(axis=0) + _REACH_M - origin) / _CELL_M).astype(int)
    a, b = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing="ij")
    centres = origin + (np.stack([a, b], axis=-1).reshape(-1, 2) + 0.5) * _CELL_M
    edges = np.roll(hull, -1, axis=0) - hull
    outward = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    dist = np.full(len(centres), -np.inf, dtype=np.float32)
    for vertex, normal in zip(hull, outward):
        np.maximum(dist, (centres - vertex) @ normal, out=dist)
    return TablePlane(
        normal=n.astype(np.float32), offset=d, inlier_frac=float(len(on_plane) / len(pts)),
        far_edge_m=float(np.percentile(np.linalg.norm(on_plane, axis=1), 98)), basis=basis.astype(np.float32),
        grid_origin=origin.astype(np.float32), outside_grid=dist.reshape(size[0], size[1]),
    )


@dataclass
class DepthMask:
    """Range gate (as the renderer: keep NEAR_M <= d < FAR_M) plus an optional table-relative mask per camera:
    drop what is below the table, more than `max_height_m` above it, or further outside its outline than
    `footprint_margin_m` (from the rear cameras the arm's tall links stand well outside the visible table)."""

    #: the renderer's clipping range (also applied by the policy server, which gets told when it is off)
    use_range_gate: bool = True
    use_plane: bool = False
    min_height_m: float = -0.05
    max_height_m: float = 1.6
    footprint_margin_m: float = 0.65
    planes: dict = field(default_factory=dict)

    def fit(self, slot: str, depth_m: np.ndarray, K: np.ndarray) -> "TablePlane | None":
        plane = fit_table_plane(depth_m, K)
        if plane is None:
            self.planes.pop(slot, None)
        else:
            self.planes[slot] = plane
        return plane

    def apply(self, slot: str, depth_m: np.ndarray, K: "np.ndarray | None") -> np.ndarray:
        if self.use_range_gate:
            d = np.where(np.isfinite(depth_m) & (depth_m >= NEAR_M) & (depth_m < FAR_M), depth_m, 0.0)
        else:
            d = np.where(np.isfinite(depth_m) & (depth_m > 0.0), depth_m, 0.0)
        d = d.astype(np.float32)
        plane = self.planes.get(slot)
        if self.use_plane and plane is not None and K is not None:
            valid = d > 0
            pts = unproject(d, K)[valid]
            h = pts @ plane.normal + plane.offset
            keep = (h >= self.min_height_m) & (h <= self.max_height_m)
            keep &= plane.outside_distance(pts) <= self.footprint_margin_m
            d[valid] = np.where(keep, d[valid], 0.0)
        return d


_RAMP = np.array([[255, 60, 30], [255, 200, 40], [80, 220, 120], [40, 150, 255], [70, 40, 200]], dtype=np.float32)


def colorize(depth_m: np.ndarray, far_m: float = FAR_M) -> np.ndarray:
    """HxW metres -> HxWx3 uint8, near warm to far cool, 0 (no return) black."""
    t = np.clip(depth_m / far_m, 0.0, 1.0) * (len(_RAMP) - 1)
    i = np.minimum(t.astype(int), len(_RAMP) - 2)
    f = (t - i)[..., None]
    rgb = _RAMP[i] * (1 - f) + _RAMP[i + 1] * f
    rgb[~(depth_m > 0)] = 0
    return rgb.astype(np.uint8)
