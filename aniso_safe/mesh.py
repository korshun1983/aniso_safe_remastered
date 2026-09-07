"""Mesh preparation ported from spectrum/routines/mesh/*.m."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay

from .structures import AttrDict


def _domain_boundary_nodes(CompStruct, ii_d):
    """Create boundary nodes for one domain, port of the loop in PrepareMeshBH.m."""
    Rx = float(CompStruct.Model.DomainRx[ii_d])
    Ry = float(CompStruct.Model.DomainRy[ii_d])
    theta_rot = float(CompStruct.Model.DomainTheta[ii_d])
    ecc = float(CompStruct.Model.DomainEcc[ii_d])
    ecc_angle = float(CompStruct.Model.DomainEccAngle[ii_d])
    xc = ecc * np.cos(ecc_angle)
    yc = ecc * np.sin(ecc_angle)

    nth = int(CompStruct.Model.DomainNth[ii_d])
    dtheta = np.pi / nth
    theta = np.arange(-np.pi, np.pi - dtheta, dtheta)

    boundary_rec = False
    if str(CompStruct.Mesh.get("ext_boundary_shape", "cir")).lower() == "rec":
        add_loc = str(CompStruct.Model.get("AddDomainLoc", "ext")).lower()
        n_domain = int(CompStruct.Data.N_domain)
        if add_loc == "ext" and ii_d >= n_domain - 1:
            boundary_rec = True
        if add_loc == "int" and ii_d == n_domain - 1:
            boundary_rec = True

    if not boundary_rec:
        xb = xc + Rx * np.cos(theta) * np.cos(theta_rot) - Ry * np.sin(theta) * np.sin(theta_rot)
        yb = yc + Rx * np.cos(theta) * np.sin(theta_rot) + Ry * np.sin(theta) * np.cos(theta_rot)
    else:
        xb = Rx * np.cos(theta).copy()
        yb = Ry * np.sin(theta).copy()
        for ia, ang in enumerate(theta):
            if -np.pi <= ang <= -(3.0 / 4.0) * np.pi:
                xb[ia] = -Rx
                yb[ia] = -Rx * np.tan(np.pi + ang)
            if (3.0 / 4.0) * np.pi <= ang <= np.pi:
                xb[ia] = -Rx
                yb[ia] = Rx * np.tan(np.pi - ang)
            if -(3.0 / 4.0) * np.pi <= ang <= -(1.0 / 4.0) * np.pi:
                xb[ia] = Ry * np.tan(np.pi / 2.0 + ang)
                yb[ia] = -Ry
            if -(1.0 / 4.0) * np.pi <= ang <= (1.0 / 4.0) * np.pi:
                xb[ia] = Rx
                yb[ia] = Rx * np.tan(ang)
            if (1.0 / 4.0) * np.pi <= ang <= (3.0 / 4.0) * np.pi:
                xb[ia] = Ry * np.tan(np.pi / 2.0 - ang)
                yb[ia] = Ry
    return np.column_stack([xb, yb])


def _interior_nodes(CompStruct, boundary_nodes):
    """Add deterministic interior nodes for Delaunay triangulation.

    Mesh2D refines by a size function. This port uses concentric fractional
    rings between neighboring boundaries to obtain a controllable initial
    triangular mesh without changing the boundary definition.
    """
    points = [np.empty((0, 2))]
    if boundary_nodes:
        points.extend(boundary_nodes)
    n_domain = int(CompStruct.Data.N_domain)
    for ii_d in range(n_domain):
        nth = int(CompStruct.Model.DomainNth[ii_d])
        dtheta = np.pi / nth
        theta = np.arange(-np.pi, np.pi - dtheta, dtheta)
        theta_rot = float(CompStruct.Model.DomainTheta[ii_d])
        ecc = float(CompStruct.Model.DomainEcc[ii_d])
        ecc_angle = float(CompStruct.Model.DomainEccAngle[ii_d])
        xc = ecc * np.cos(ecc_angle)
        yc = ecc * np.sin(ecc_angle)
        if ii_d == 0:
            fractions = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
        else:
            fractions = np.array([1.0 / 3.0, 2.0 / 3.0])
        rx = float(CompStruct.Model.DomainRx[ii_d])
        ry = float(CompStruct.Model.DomainRy[ii_d])
        for frac in fractions:
            if frac == 0.0 and ii_d == 0:
                points.append(np.array([[xc, yc]]))
                continue
            if ii_d > 0:
                rx0 = float(CompStruct.Model.DomainRx[ii_d - 1])
                ry0 = float(CompStruct.Model.DomainRy[ii_d - 1])
                rxi = rx0 + frac * (rx - rx0)
                ryi = ry0 + frac * (ry - ry0)
            else:
                rxi = frac * rx
                ryi = frac * ry
            x = xc + rxi * np.cos(theta) * np.cos(theta_rot) - ryi * np.sin(theta) * np.sin(theta_rot)
            y = yc + rxi * np.cos(theta) * np.sin(theta_rot) + ryi * np.sin(theta) * np.cos(theta_rot)
            points.append(np.column_stack([x, y]))
    return np.vstack(points)


def _face_numbers(points, boundary_nodes):
    """Assign a domain/face number to each triangle centroid."""
    paths = [MplPath(nodes) for nodes in boundary_nodes]
    faces = np.zeros(points.shape[0], dtype=int)
    n_boundaries = len(paths)
    for i, point in enumerate(points):
        inside = np.array([path.contains_point(point) for path in paths], dtype=bool)
        if not np.any(inside):
            faces[i] = n_boundaries
        else:
            # For nested boundaries the innermost containing boundary count is
            # converted to the MATLAB one-based domain index.
            faces[i] = n_boundaries - int(np.count_nonzero(inside)) + 1
    return np.clip(faces, 1, n_boundaries)


def PrepareMeshBH(CompStruct):
    """Prepare linear triangular mesh, port of PrepareMeshBH.m."""
    boundary_nodes = []
    edges = []
    domain_faces = []
    prev_edge_size = 0
    node_offset = 0
    for ii_d in range(int(CompStruct.Data.N_domain)):
        nodes = _domain_boundary_nodes(CompStruct, ii_d)
        n_nodes = nodes.shape[0]
        domain_edges = np.column_stack([np.arange(n_nodes), np.roll(np.arange(n_nodes), -1)]) + node_offset
        face_start = len(edges) - prev_edge_size
        edges.extend(domain_edges.tolist())
        face_end = len(edges)
        domain_faces.append(np.arange(face_start, face_end, dtype=int))
        prev_edge_size = n_nodes
        node_offset += n_nodes
        boundary_nodes.append(nodes)

    nodes = _interior_nodes(CompStruct, boundary_nodes)
    tri = Delaunay(nodes)
    simplices = tri.simplices.astype(int)
    centroids = nodes[simplices].mean(axis=1)
    faces = _face_numbers(centroids, boundary_nodes)

    order = np.lexsort((faces, np.arange(len(faces))))
    mesh_tri = np.column_stack([simplices[order], faces[order]]).T
    mesh_nodes = nodes.T
    boundary_edges = FindBEdges(mesh_nodes, mesh_tri, CompStruct)
    return mesh_nodes, boundary_edges, mesh_tri, CompStruct


def AddNodesCubic(MeshNodes, MeshTri):
    """Convert linear triangles to cubic triangles, port of AddNodesCubic.m."""
    n_tri = MeshTri.shape[1]
    mesh_tri_cubic = np.zeros((11, n_tri), dtype=int)
    mesh_nodes_cubic = np.asarray(MeshNodes, dtype=float).copy()
    mesh_tri_cubic[0:3, :] = MeshTri[0:3, :]
    mesh_tri_cubic[10, :] = MeshTri[3, :]

    mesh_props = AttrDict()
    mesh_props.delta = np.zeros(n_tri)
    mesh_props.a = np.zeros((3, n_tri))
    mesh_props.b = np.zeros((3, n_tri))
    mesh_props.c = np.zeros((3, n_tri))

    node_index = {}
    for idx in range(mesh_nodes_cubic.shape[1]):
        key = (round(float(mesh_nodes_cubic[0, idx]), 12), round(float(mesh_nodes_cubic[1, idx]), 12))
        node_index[key] = idx

    for ii in range(n_tri):
        corner = MeshTri[0:3, ii].astype(int)
        x = mesh_nodes_cubic[0, corner]
        y = mesh_nodes_cubic[1, corner]
        cubic_x = np.empty(10)
        cubic_y = np.empty(10)
        cubic_x[0:3] = x
        cubic_y[0:3] = y
        cubic_x[3] = (2 * x[0] + x[1]) / 3
        cubic_y[3] = (2 * y[0] + y[1]) / 3
        cubic_x[4] = (x[0] + 2 * x[1]) / 3
        cubic_y[4] = (y[0] + 2 * y[1]) / 3
        cubic_x[5] = (2 * x[1] + x[2]) / 3
        cubic_y[5] = (2 * y[1] + y[2]) / 3
        cubic_x[6] = (x[1] + 2 * x[2]) / 3
        cubic_y[6] = (y[1] + 2 * y[2]) / 3
        cubic_x[7] = (x[0] + 2 * x[2]) / 3
        cubic_y[7] = (y[0] + 2 * y[2]) / 3
        cubic_x[8] = (2 * x[0] + x[2]) / 3
        cubic_y[8] = (2 * y[0] + y[2]) / 3
        cubic_x[9] = (x[0] + x[1] + x[2]) / 3
        cubic_y[9] = (y[0] + y[1] + y[2]) / 3

        mesh_props.delta[ii] = 0.5 * np.linalg.det(
            np.array([[1.0, x[0], y[0]], [1.0, x[1], y[1]], [1.0, x[2], y[2]]])
        )
        mesh_props.a[:, ii] = [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
        mesh_props.b[:, ii] = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
        mesh_props.c[:, ii] = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]

        for jj in range(3, 10):
            key = (round(float(cubic_x[jj]), 12), round(float(cubic_y[jj]), 12))
            found = node_index.get(key)
            if found is None:
                mesh_nodes_cubic = np.column_stack([mesh_nodes_cubic, [cubic_x[jj], cubic_y[jj]]])
                found = mesh_nodes_cubic.shape[1] - 1
                node_index[key] = found
            mesh_tri_cubic[jj, ii] = found
    return mesh_nodes_cubic, mesh_tri_cubic, mesh_props


def FindBEdges(MeshNodes, MeshTri, CompStruct):
    """Identify domain boundary edges, port of FindBEdges.m."""
    edge_domains = defaultdict(list)
    for ii in range(MeshTri.shape[1]):
        nodes = MeshTri[0:3, ii].astype(int)
        domain = int(MeshTri[3, ii])
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edge = (min(a, b), max(a, b))
            edge_domains[edge].append(domain)

    rows = []
    for (a, b), domains in edge_domains.items():
        unique = sorted(set(domains))
        if len(domains) == 1 or len(unique) > 1:
            domain = unique[0] if len(unique) == 1 else unique[0]
            rows.append((a, b, domain))
    if not rows:
        return np.zeros((3, 0), dtype=int)
    bedges = np.asarray(rows, dtype=int).T
    order = np.argsort(bedges[2, :], kind="mergesort")
    return bedges[:, order]


def plot_mesh(MeshNodes, MeshTri, BoundaryEdges, out_png, title="SAFE mesh"):
    """Save a control plot of the linear/cubic mesh."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    tri = MeshTri[0:3, :].T.astype(int)
    domain_row = 10 if MeshTri.shape[0] >= 11 else 3
    domains = MeshTri[domain_row, :].astype(int)
    ax.triplot(MeshNodes[0, :], MeshNodes[1, :], tri, color="0.55", linewidth=0.5)
    for domain in np.unique(domains):
        mask = domains == domain
        elem_nodes = np.unique(tri[mask].ravel())
        ax.scatter(MeshNodes[0, elem_nodes], MeshNodes[1, elem_nodes], s=8, label=f"domain {domain}")
    if BoundaryEdges.size:
        b = np.unique(BoundaryEdges[0:2, :].astype(int).ravel())
        ax.scatter(MeshNodes[0, b], MeshNodes[1, b], s=10, c="k", label="boundary")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def PrepareMesh_sp_SAFE(CompStruct, out_png=None):
    """Prepare mesh and optional control output, port of PrepareMesh_sp_SAFE.m."""
    if "Mesh" not in CompStruct:
        CompStruct.Mesh = AttrDict()
    if "ext_boundary_shape" not in CompStruct.Mesh:
        CompStruct.Mesh.ext_boundary_shape = "cir"
    if str(CompStruct.Mesh.ext_boundary_shape).lower() not in ("cir", "rec"):
        raise ValueError("Mesh.ext_boundary_shape has invalid value")

    mesh_nodes, boundary_edges, mesh_tri, CompStruct = PrepareMeshBH(CompStruct)
    order = np.argsort(mesh_tri[3, :], kind="mergesort")
    mesh_tri = mesh_tri[:, order]
    mesh_nodes, mesh_tri, mesh_props = AddNodesCubic(mesh_nodes, mesh_tri)
    if out_png is not None and str(CompStruct.Mesh.get("output", "no")).lower() == "yes":
        plot_mesh(mesh_nodes, mesh_tri, boundary_edges, out_png)
    return mesh_nodes, boundary_edges, mesh_tri, mesh_props, CompStruct
