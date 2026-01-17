#%%
"""
================================================================================
Nonlinear BIAXIAL RC COLUMN ANALYSIS TOOL (v0.1)
================================================================================

DESCRIPTION:
This open-source Python script performs advanced nonlinear analysis of "rectangular 
Reinforced Concrete (RC) columns" under biaxial bending and axial loads. 

KEY FEATURES:
- Material Nonlinearity: Implements the Hognestad constitutive model for 
  concrete and Elastic-Perfectly Plastic (EPP) model for steel.
- Confinement Effect: Optional modeling of core confinement (Mander model) to capture 
  enhanced strength and ductility due to stirrups.
- Tension Stiffening: Includes a simplified tension model for uncracked concrete.
- This code use High-Order Numerical Integration instead of fiber method
- Advanced Visualization: Generates stress/strain heatmaps, N-Mx-My capacity curves, 
  and full 3D interaction surfaces.

DISCLAIMER (v0.1):
This software is currently in the ALPHA stage (v0.1). It is provided "as is" for 
educational and research purposes. While efforts have been made to ensure accuracy, 
the code may contain bugs and has not yet undergone rigorous third-party validation. 
Users are advised to verify results against standard commercial software or hand calculations

You can use the code to study the column and develop more advanced tools

LICENSE & ATTRIBUTION:
Open Source. Free to use and modify.
Developed by: KHALIL OTHMAN 
contact me: eng.khalil.othman@gmail.com
================================================================================
================================================================================
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm

from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")



# ============================================================================
# PART 1: MANDER CONFINEMENT MODEL
# ============================================================================

def calculate_mander_confinement(fc: float, Lx: float, Ly: float, 
                                  cover: float, db_v: float, s: float,
                                  legs_x: int, legs_y: int, fy_v: float,
                                  rein_x: List[float], rein_y: List[float],
                                  rein_ds: List[float]) -> Dict:
    """
    Calculate Mander confinement parameters.
    Reference: Mander, Priestley, Park (1988)
    """
    # Core dimensions (centerline of stirrups)
    core_offset = cover + db_v / 2
    bc_x = Lx - 2 * core_offset
    bc_y = Ly - 2 * core_offset
    
    # Core boundaries
    core_x_min = core_offset
    core_x_max = Lx - core_offset
    core_y_min = core_offset
    core_y_max = Ly - core_offset
    
    # Area of stirrup legs
    Asv = np.pi * (db_v / 2)**2
    
    # Total area of transverse reinforcement in each direction
    Asx = legs_x * Asv
    Asy = legs_y * Asv
    
    # Transverse reinforcement ratios
    rho_x = Asx / (s * bc_y)
    rho_y = Asy / (s * bc_x)
    
    # Effective lateral confining pressures
    fl_x = rho_x * fy_v
    fl_y = rho_y * fy_v
    
    # Clear spacing between stirrups
    s_clear = s - db_v
    
    # Confinement Effectiveness Coefficient (Ke)
    rein_ds_arr = np.array(rein_ds)
    As_long = np.sum(np.pi * (rein_ds_arr / 2)**2)
    Ac = bc_x * bc_y
    rho_cc = As_long / Ac
    
    n_bars = len(rein_x)
    if n_bars >= 4:
        n_x_face = max(2, int(np.sqrt(n_bars)))
        n_y_face = max(2, int(np.sqrt(n_bars)))
        avg_spacing_x = bc_x / (n_x_face - 1) if n_x_face > 1 else bc_x
        avg_spacing_y = bc_y / (n_y_face - 1) if n_y_face > 1 else bc_y
        db_avg = np.mean(rein_ds_arr)
        w_x = max(avg_spacing_x - db_avg, 0)
        w_y = max(avg_spacing_y - db_avg, 0)
        sum_wi_sq = 2 * (n_x_face - 1) * w_x**2 + 2 * (n_y_face - 1) * w_y**2
    else:
        sum_wi_sq = 2 * (bc_x**2 + bc_y**2) / 4
    
    Ae_horizontal = max(bc_x * bc_y - sum_wi_sq / 6, 0)
    Ae_vertical_factor = max((1 - s_clear / (2 * bc_x)), 0) * max((1 - s_clear / (2 * bc_y)), 0)
    Ae = Ae_horizontal * Ae_vertical_factor
    Acc = Ac * (1 - rho_cc)
    
    Ke = Ae / Acc if Acc > 0 else 0
    Ke = min(max(Ke, 0.0), 1.0)
    
    # Effective Lateral Confining Pressures
    fl_x_eff = Ke * fl_x
    fl_y_eff = Ke * fl_y
    
    if fl_x_eff > 0 and fl_y_eff > 0:
        fl_eff = np.sqrt(fl_x_eff * fl_y_eff)
    else:
        fl_eff = (fl_x_eff + fl_y_eff) / 2
    
    # Confined Concrete Strength (f'cc) - Mander equation
    fl_ratio = fl_eff / fc if fc > 0 else 0
    
    if fl_ratio > 0:
        fcc = fc * (-1.254 + 2.254 * np.sqrt(1 + 7.94 * fl_ratio) - 2 * fl_ratio)
    else:
        fcc = fc
    fcc = max(fcc, fc)
    
    # Confined strain at peak
    eps_c0 = 0.002
    eps_cc = eps_c0 * (1 + 5 * (fcc / fc - 1))
    
    # Ultimate strain
    rho_s = (rho_x + rho_y) / 2
    eps_su = 0.10
    eps_cu = 0.004 + 1.4 * rho_s * fy_v * eps_su / fcc
    eps_cu = max(eps_cu, 0.01)
    
    # Curve shape parameter
    Ec = 4700 * np.sqrt(fc)
    Esec = fcc / eps_cc
    r = Ec / (Ec - Esec) if (Ec - Esec) > 0 else 5.0
    
    return {
        'core_x_min': core_x_min, 'core_x_max': core_x_max,
        'core_y_min': core_y_min, 'core_y_max': core_y_max,
        'bc_x': bc_x, 'bc_y': bc_y,
        'Ke': Ke, 'fl_eff': fl_eff,
        'fcc': fcc, 'eps_cc': eps_cc, 'eps_cu': eps_cu,
        'Ec': Ec, 'r': r,
        'strength_ratio': fcc / fc,
    }


# ============================================================================
# PART 2: MATERIAL MODELS (Vectorized for performance)
# ============================================================================

def mander_confined_stress(eps: np.ndarray, fcc: float, eps_cc: float, 
                            Ec: float, r: float) -> np.ndarray:
    """Mander confined concrete - vectorized."""
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    sigma = np.zeros_like(eps)
    
    mask = eps > 0
    if np.any(mask):
        x = eps[mask] / eps_cc
        denom = np.maximum(r - 1 + np.power(x, r), 1e-10)
        sigma[mask] = np.minimum(fcc * x * r / denom, fcc)
    
    return sigma


def unconfined_spalling_stress(eps: np.ndarray, fc: float, eps_c0: float, 
                                eps_spall: float) -> np.ndarray:
    """Unconfined concrete with spalling - vectorized."""
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    sigma = np.zeros_like(eps)
    
    # Ascending
    mask_asc = (eps > 0) & (eps <= eps_c0)
    if np.any(mask_asc):
        ratio = eps[mask_asc] / eps_c0
        sigma[mask_asc] = fc * (2 * ratio - ratio**2)
    
    # Descending (before spalling)
    mask_desc = (eps > eps_c0) & (eps <= eps_spall)
    if np.any(mask_desc):
        sigma[mask_desc] = np.maximum(
            fc * (1 - (eps[mask_desc] - eps_c0) / (eps_spall - eps_c0)), 0.0)
    
    # Spalled: already zero
    return sigma


def concrete_stress_with_tension(eps: np.ndarray, fc: float, Ec: float, 
                                  eps_c0: float) -> np.ndarray:
    """Standard concrete with tension - vectorized."""
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    sigma = np.zeros_like(eps)
    
    ft = 0.33 * np.sqrt(fc)
    eps_cr = -ft / Ec
    
    # Compression ascending
    mask_asc = (eps > 0) & (eps <= eps_c0)
    if np.any(mask_asc):
        ratio = eps[mask_asc] / eps_c0
        sigma[mask_asc] = fc * (2 * ratio - ratio**2)
    
    # Compression descending
    mask_desc = eps > eps_c0
    if np.any(mask_desc):
        eps_c50 = max((3 + 0.29 * fc) / (145 * fc - 1000), eps_c0 * 2.0)
        z = 0.5 / (eps_c50 - eps_c0)
        sigma[mask_desc] = np.maximum(fc * (1 - z * (eps[mask_desc] - eps_c0)), 0.2 * fc)
    
    # Tension pre-cracking
    mask_pre = (eps < 0) & (eps >= eps_cr)
    if np.any(mask_pre):
        sigma[mask_pre] = Ec * eps[mask_pre]
    
    # Tension post-cracking
    mask_post = eps < eps_cr
    if np.any(mask_post):
        decay = np.exp(0.4 * (eps[mask_post] - eps_cr) / abs(eps_cr))
        sigma[mask_post] = -ft * decay
    
    return sigma


def steel_stress(eps: np.ndarray, fy: float, Es: float) -> np.ndarray:
    """Steel stress - vectorized."""
    return np.clip(np.asarray(eps) * Es, -fy, fy)


# ============================================================================
# PART 3: NUMERICAL INTEGRATION POINTS RULES
# ============================================================================
# Includes: 
# - Gauss-Legendre 9-point rule for Quads
# - High-order 45-point rule for Triangles
# Implements Lazy Loading (Caching) for performance.
# ============================================================================


_GAUSS_9PT = None
_NC_45PT = None

def _get_gauss_9pt():
    global _GAUSS_9PT
    if _GAUSS_9PT is None:
        s = np.sqrt(0.6)
        pts = np.array([-s, 0.0, s])
        wts = np.array([5/9, 8/9, 5/9])
        xi, eta = np.meshgrid(pts, pts)
        wi, wj = np.meshgrid(wts, wts)
        _GAUSS_9PT = (xi.ravel(), eta.ravel(), (wi * wj).ravel())
    return _GAUSS_9PT


def _get_nc_45pt():
    global _NC_45PT
    if _NC_45PT is None:
        n = 8
        pts = [(1,0,0), (0,1,0), (0,0,1)]
        for k in range(1, n):
            pts.append(((n-k)/n, k/n, 0))
        for k in range(1, n):
            pts.append((0, (n-k)/n, k/n))
        for k in range(1, n):
            pts.append((k/n, 0, (n-k)/n))
        for k3 in range(1, n-1):
            for k2 in range(1, n-k3):
                k1 = n - k2 - k3
                if k1 >= 1:
                    pts.append((k1/n, k2/n, k3/n))
        pts = np.array(pts, dtype=float)
        cj = {0: [1,2,3], 94208: [4,10,11,17,18,24], -119808: [5,9,12,16,19,23],
              290816: [6,8,13,15,20,22], -277248: [7,14,21], 180224: [25,30,45],
              212992: [26,29,35,44,43,31], 172032: [27,28,39,42,40,36],
              -370688: [32,34,41], 376832: [33,37,38]}
        cj_arr = np.zeros(45)
        for val, idx_list in cj.items():
            for idx in idx_list:
                cj_arr[idx-1] = val
        weights = 0.5 * cj_arr / np.sum(cj_arr)
        _NC_45PT = (pts[:, 0], pts[:, 1], weights)
    return _NC_45PT


# ============================================================================
# PART 4: GEOMETRY 
# ============================================================================

def clip_polygon_by_rect(polygon: np.ndarray, x_min: float, y_min: float,
                          x_max: float, y_max: float) -> np.ndarray:
    """Sutherland-Hodgman clipping."""
    if len(polygon) < 3:
        return np.array([])
    
    def clip_edge(poly, x1, y1, x2, y2):
        if len(poly) < 3:
            return []
        
        dx, dy = x2 - x1, y2 - y1
        output = []
        n = len(poly)
        
        for i in range(n):
            curr = poly[i]
            next_p = poly[(i + 1) % n]
            
            # Cross product to determine side
            curr_side = dx * (curr[1] - y1) - dy * (curr[0] - x1)
            next_side = dx * (next_p[1] - y1) - dy * (next_p[0] - x1)
            
            if curr_side >= 0:
                if next_side >= 0:
                    output.append(next_p)
                else:
                    # Intersection
                    denom = (curr[0] - next_p[0]) * dy - (curr[1] - next_p[1]) * dx
                    if abs(denom) > 1e-12:
                        t = ((curr[0] - x1) * dy - (curr[1] - y1) * dx) / denom
                        output.append([curr[0] + t * (next_p[0] - curr[0]),
                                      curr[1] + t * (next_p[1] - curr[1])])
            elif next_side >= 0:
                denom = (curr[0] - next_p[0]) * dy - (curr[1] - next_p[1]) * dx
                if abs(denom) > 1e-12:
                    t = ((curr[0] - x1) * dy - (curr[1] - y1) * dx) / denom
                    output.append([curr[0] + t * (next_p[0] - curr[0]),
                                  curr[1] + t * (next_p[1] - curr[1])])
                output.append(next_p)
        
        return output
    
    result = list(polygon)
    # Clip by 4 edges
    result = clip_edge(result, x_min, y_min, x_max, y_min)  # bottom
    result = clip_edge(result, x_max, y_min, x_max, y_max)  # right
    result = clip_edge(result, x_max, y_max, x_min, y_max)  # top
    result = clip_edge(result, x_min, y_max, x_min, y_min)  # left
    
    return np.array(result) if len(result) >= 3 else np.array([])


def get_compression_zone(Lx: float, Ly: float, eps_0: float, 
                          kappa_x: float, kappa_y: float, 
                          cx: float, cy: float) -> np.ndarray:
    """Get compression zone polygon."""
    corners = np.array([(0,0), (Lx,0), (Lx,Ly), (0,Ly)], dtype=float)
    strains = eps_0 + kappa_x * (corners[:, 1] - cy) + kappa_y * (corners[:, 0] - cx)
    
    if np.all(strains >= 0):
        return corners
    if np.all(strains <= 0):
        return np.array([])
    
    out = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i+1) % 4]
        e1, e2 = strains[i], strains[(i+1) % 4]
        
        if e1 >= 0:
            out.append(p1.copy())
        if (e1 > 0 and e2 < 0) or (e1 < 0 and e2 > 0):
            t = e1 / (e1 - e2)
            out.append(p1 + t * (p2 - p1))
    
    return np.array(out) if len(out) >= 3 else np.array([])


def get_tension_zone(Lx: float, Ly: float, eps_0: float,
                      kappa_x: float, kappa_y: float,
                      cx: float, cy: float) -> np.ndarray:
    """Get tension zone polygon."""
    corners = np.array([(0,0), (Lx,0), (Lx,Ly), (0,Ly)], dtype=float)
    strains = eps_0 + kappa_x * (corners[:, 1] - cy) + kappa_y * (corners[:, 0] - cx)
    
    if np.all(strains <= 0):
        return corners
    if np.all(strains >= 0):
        return np.array([])
    
    out = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i+1) % 4]
        e1, e2 = strains[i], strains[(i+1) % 4]
        
        if e1 <= 0:
            out.append(p1.copy())
        if (e1 > 0 and e2 < 0) or (e1 < 0 and e2 > 0):
            t = e1 / (e1 - e2)
            out.append(p1 + t * (p2 - p1))
    
    return np.array(out) if len(out) >= 3 else np.array([])


def decompose_polygon(vertices: np.ndarray) -> List[Dict]:
    """Decompose polygon into triangles/quads."""
    if len(vertices) < 3:
        return []
    
    n = len(vertices)
    # Check orientation
    area = sum(vertices[i,0]*vertices[(i+1)%n,1] - vertices[(i+1)%n,0]*vertices[i,1] 
               for i in range(n))
    if area < 0:
        vertices = vertices[::-1]
    
    if n == 3:
        return [{'type': 'tri', 'verts': vertices}]
    if n == 4:
        return [{'type': 'quad', 'verts': vertices}]
    
    # Fan triangulation for n > 4
    return [{'type': 'tri', 'verts': np.array([vertices[0], vertices[i], vertices[i+1]])}
            for i in range(1, n-1)]


# ============================================================================
# PART 5: SECTION INTEGRATION 
# ============================================================================

def integrate_zone(zone: np.ndarray, strain_func, stress_func, 
                   cx: float, cy: float, stress_params: Dict) -> Tuple[float, float, float]:
    """Integrate stresses over a zone """
    if len(zone) < 3:
        return 0.0, 0.0, 0.0
    
    shapes = decompose_polygon(zone)
    N, Mx, My = 0.0, 0.0, 0.0
    
    for shape in shapes:
        verts = shape['verts']
        
        if shape['type'] == 'quad':
            xi_pts, eta_pts, weights = _get_gauss_9pt()
            
            # Vectorized shape functions
            for xi, eta, w in zip(xi_pts, eta_pts, weights):
                N_sh = 0.25 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                                        (1+xi)*(1+eta), (1-xi)*(1+eta)])
                x = np.dot(N_sh, verts[:, 0])
                y = np.dot(N_sh, verts[:, 1])
                
                dN_dxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
                dN_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
                
                det_J = (np.dot(dN_dxi, verts[:,0]) * np.dot(dN_deta, verts[:,1]) -
                         np.dot(dN_dxi, verts[:,1]) * np.dot(dN_deta, verts[:,0]))
                
                if det_J <= 0:
                    continue
                
                eps = strain_func(x, y)
                sigma = stress_func(np.array([eps]), **stress_params)[0]
                dA = w * det_J
                
                N += sigma * dA
                Mx += sigma * (y - cy) * dA
                My += sigma * (x - cx) * dA
        
        else:  # triangle
            L1_pts, L2_pts, weights = _get_nc_45pt()
            L3_pts = 1.0 - L1_pts - L2_pts
            
            V0, V1, V2 = verts
            area = 0.5 * abs((V1[0]-V0[0])*(V2[1]-V0[1]) - (V2[0]-V0[0])*(V1[1]-V0[1]))
            
            if area < 1e-12:
                continue
            
            # Vectorized integration
            x_pts = L1_pts * V0[0] + L2_pts * V1[0] + L3_pts * V2[0]
            y_pts = L1_pts * V0[1] + L2_pts * V1[1] + L3_pts * V2[1]
            
            eps_pts = strain_func(x_pts, y_pts)
            sigma_pts = stress_func(eps_pts, **stress_params)
            dA = weights * (area / 0.5)
            
            N += np.sum(sigma_pts * dA)
            Mx += np.sum(sigma_pts * (y_pts - cy) * dA)
            My += np.sum(sigma_pts * (x_pts - cx) * dA)
    
    return N, Mx, My


def analyze_section(eps_0: float, kappa_x: float, kappa_y: float,
                    section_params: Dict,
                    confinement_params: Optional[Dict] = None,
                    include_tension: bool = True) -> Tuple[float, float, float, np.ndarray, bool, float]:
    """
    Analyze section forces .
    Only computes confinement if confinement_params is provided.
    """
    Lx, Ly = section_params['Lx'], section_params['Ly']
    fc, Ec = section_params['fc'], section_params['Ec']
    eps_c0 = section_params['eps_c0']
    cx, cy = Lx / 2, Ly / 2
    
    def strain_at(x, y):
        return eps_0 + kappa_x * (y - cy) + kappa_y * (x - cx)
    
    N_total, Mx_total, My_total = 0.0, 0.0, 0.0
    
    # Compression zone
    comp_zone = get_compression_zone(Lx, Ly, eps_0, kappa_x, kappa_y, cx, cy)
    
    if len(comp_zone) >= 3:
        if confinement_params is not None:
            # CONFINED: Split into core and cover
            core_x_min = confinement_params['core_x_min']
            core_x_max = confinement_params['core_x_max']
            core_y_min = confinement_params['core_y_min']
            core_y_max = confinement_params['core_y_max']
            
            # Core (confined)
            core_zone = clip_polygon_by_rect(comp_zone, core_x_min, core_y_min,
                                              core_x_max, core_y_max)
            if len(core_zone) >= 3:
                core_params = {
                    'fcc': confinement_params['fcc'],
                    'eps_cc': confinement_params['eps_cc'],
                    'Ec': confinement_params['Ec'],
                    'r': confinement_params['r']
                }
                Nc, Mxc, Myc = integrate_zone(core_zone, strain_at, 
                                               mander_confined_stress, cx, cy, core_params)
                N_total += Nc
                Mx_total += Mxc
                My_total += Myc
            
            # Cover (4 strips around core)
            eps_spall = max(0.004, 2 * eps_c0)
            cover_params = {'fc': fc, 'eps_c0': eps_c0, 'eps_spall': eps_spall}
            
            # Get cover regions by subtracting core from compression zone
            poly_bounds = (np.min(comp_zone[:, 0]), np.max(comp_zone[:, 0]),
                          np.min(comp_zone[:, 1]), np.max(comp_zone[:, 1]))
            
            # Bottom strip
            if core_y_min > poly_bounds[2]:
                bottom = clip_polygon_by_rect(comp_zone, poly_bounds[0], poly_bounds[2],
                                               poly_bounds[1], core_y_min)
                if len(bottom) >= 3:
                    Nb, Mxb, Myb = integrate_zone(bottom, strain_at,
                                                   unconfined_spalling_stress, cx, cy, cover_params)
                    N_total += Nb
                    Mx_total += Mxb
                    My_total += Myb
            
            # Top strip
            if core_y_max < poly_bounds[3]:
                top = clip_polygon_by_rect(comp_zone, poly_bounds[0], core_y_max,
                                            poly_bounds[1], poly_bounds[3])
                if len(top) >= 3:
                    Nt, Mxt, Myt = integrate_zone(top, strain_at,
                                                   unconfined_spalling_stress, cx, cy, cover_params)
                    N_total += Nt
                    Mx_total += Mxt
                    My_total += Myt
            
            # Left strip (between core y bounds)
            if core_x_min > poly_bounds[0]:
                left = clip_polygon_by_rect(comp_zone, poly_bounds[0], core_y_min,
                                             core_x_min, core_y_max)
                if len(left) >= 3:
                    Nl, Mxl, Myl = integrate_zone(left, strain_at,
                                                   unconfined_spalling_stress, cx, cy, cover_params)
                    N_total += Nl
                    Mx_total += Mxl
                    My_total += Myl
            
            # Right strip
            if core_x_max < poly_bounds[1]:
                right = clip_polygon_by_rect(comp_zone, core_x_max, core_y_min,
                                              poly_bounds[1], core_y_max)
                if len(right) >= 3:
                    Nr, Mxr, Myr = integrate_zone(right, strain_at,
                                                   unconfined_spalling_stress, cx, cy, cover_params)
                    N_total += Nr
                    Mx_total += Mxr
                    My_total += Myr
        
        else:
            # UNCONFINED: Single zone
            comp_params = {'fc': fc, 'Ec': Ec, 'eps_c0': eps_c0}
            Nc, Mxc, Myc = integrate_zone(comp_zone, strain_at,
                                           concrete_stress_with_tension, cx, cy, comp_params)
            N_total += Nc
            Mx_total += Mxc
            My_total += Myc
    
    # Tension zone
    if include_tension:
        tens_zone = get_tension_zone(Lx, Ly, eps_0, kappa_x, kappa_y, cx, cy)
        if len(tens_zone) >= 3:
            tens_params = {'fc': fc, 'Ec': Ec, 'eps_c0': eps_c0}
            Nt, Mxt, Myt = integrate_zone(tens_zone, strain_at,
                                           concrete_stress_with_tension, cx, cy, tens_params)
            N_total += Nt
            Mx_total += Mxt
            My_total += Myt
    
    # Steel
    rein_x = np.asarray(section_params['rein_x'])
    rein_y = np.asarray(section_params['rein_y'])
    rein_ds = np.asarray(section_params['rein_ds'])
    
    eps_s = eps_0 + kappa_x * (rein_y - cy) + kappa_y * (rein_x - cx)
    sigma_s = steel_stress(eps_s, section_params['fy'], section_params['Es'])
    As = np.pi * (rein_ds / 2)**2
    Fs = sigma_s * As
    
    N_total += np.sum(Fs)
    Mx_total += np.sum(Fs * (rein_y - cy))
    My_total += np.sum(Fs * (rein_x - cx))
    
    # Cracking check
    corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    min_strain = np.min(eps_0 + kappa_x * (corners[:, 1] - cy) + kappa_y * (corners[:, 0] - cx))
    ft = 0.33 * np.sqrt(fc)
    eps_cr = -ft / Ec
    is_cracked = min_strain < eps_cr
    
    return float(N_total), float(Mx_total), float(My_total), eps_s, is_cracked, min_strain


# ============================================================================
# PART 6: Numerical Optimization Solvers
# ============================================================================

def nelder_mead(func, x0, tol=1e-8, max_iter=300):
    """Nelder-Mead optimizer """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] = x0[i] * 1.1 if abs(x0[i]) > 1e-12 else 1e-5
    
    fvals = np.array([func(v) for v in simplex])
    
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        if fvals[0] < tol or np.max(np.abs(fvals[-1] - fvals[0])) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect
        xr = centroid + (centroid - simplex[-1])
        fr = func(xr)
        
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            # Expand
            xe = centroid + 2 * (xr - centroid)
            fe = func(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
        else:
            # Contract
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc_val = func(xc)
            if fc_val < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc_val
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
    
    best = np.argmin(fvals)
    return simplex[best], fvals[best]


def grid_search(obj_func, eps_max, Lx, Ly, e0x, e0y):
    """Coarse grid search for initial guess - FIXED for all cases."""
    best_x, best_f = np.array([0.0, 0.0]), float('inf')
    
    cx, cy = Lx / 2, Ly / 2
    L_diag = np.sqrt(Lx**2 + Ly**2)
    
    # Determine eccentricity type
    abs_ex, abs_ey = abs(e0x), abs(e0y)
    is_extreme = (abs_ex > 5 * Lx) or (abs_ey > 5 * Ly)
    is_uniaxial_x = abs_ex < 0.1 and abs_ey > 0.1  # Bending about X only
    is_uniaxial_y = abs_ey < 0.1 and abs_ex > 0.1  # Bending about Y only
    
    kappa_base = eps_max / min(Lx, Ly)
    
    # ==========================================================================
    # Case 1: Uniaxial bending about X (e0x ≈ 0)
    # ==========================================================================
    if is_uniaxial_x:
        sy = np.sign(e0y) if abs_ey > 0.1 else 1
        # Only search along kx axis (ky = 0)
        for mag in np.logspace(-3, 1, 30) * kappa_base:
            kx = mag * sy
            f = obj_func(np.array([kx, 0.0]))
            if f < best_f:
                best_f = f
                best_x = np.array([kx, 0.0])
        return best_x
    
    # ==========================================================================
    # Case 2: Uniaxial bending about Y (e0y ≈ 0)
    # ==========================================================================
    if is_uniaxial_y:
        sx = np.sign(e0x) if abs_ex > 0.1 else 1
        # Only search along ky axis (kx = 0)
        for mag in np.logspace(-3, 1, 30) * kappa_base:
            ky = mag * sx
            f = obj_func(np.array([0.0, ky]))
            if f < best_f:
                best_f = f
                best_x = np.array([0.0, ky])
        return best_x
    
    # ==========================================================================
    # Case 3: Extreme eccentricity (N → 0, pure bending)
    # ==========================================================================
    if is_extreme:
        # For extreme eccentricity, search high curvatures that give N ≈ 0
        # The neutral axis should pass near the centroid
        ratio = abs_ey / max(abs_ex, 0.1) if abs_ex > 0.1 else 1000
        
        for mag in np.logspace(0, 2, 20) * kappa_base:
            # Direction based on eccentricity ratio
            if abs_ex < 0.1:
                angles = [np.pi/2]  # Pure X bending
            elif abs_ey < 0.1:
                angles = [0]  # Pure Y bending
            else:
                angles = [np.arctan2(abs_ey, abs_ex)]
            
            for theta in angles:
                sy = np.sign(e0y) if abs_ey > 0.1 else 1
                sx = np.sign(e0x) if abs_ex > 0.1 else 1
                kx = mag * np.sin(theta) * sy
                ky = mag * np.cos(theta) * sx
                f = obj_func(np.array([kx, ky]))
                if f < best_f:
                    best_f = f
                    best_x = np.array([kx, ky])
        return best_x
    
    # ==========================================================================
    # Case 4: Biaxial bending (general case)
    # ==========================================================================
    sx = np.sign(e0x) if abs_ex > 0.1 else 1
    sy = np.sign(e0y) if abs_ey > 0.1 else 1
    
    # Polar search
    magnitudes = np.logspace(-2, 2, 15) * kappa_base
    angles = np.linspace(0.1, np.pi/2 - 0.1, 12)
    
    for theta in angles:
        for mag in magnitudes:
            kx = mag * np.sin(theta) * sy
            ky = mag * np.cos(theta) * sx
            f = obj_func(np.array([kx, ky]))
            if f < best_f:
                best_f = f
                best_x = np.array([kx, ky])
    
    # Also try axis-aligned cases
    for mag in magnitudes:
        # Pure X bending
        f = obj_func(np.array([mag * sy, 0.0]))
        if f < best_f:
            best_f = f
            best_x = np.array([mag * sy, 0.0])
        # Pure Y bending
        f = obj_func(np.array([0.0, mag * sx]))
        if f < best_f:
            best_f = f
            best_x = np.array([0.0, mag * sx])
    
    return best_x


# ============================================================================
# PART 7: N-ε CURVE GENERATOR
# ============================================================================

def generate_N_eps_curve(e0x: float, e0y: float, section_params: Dict,
                         confinement_params: Optional[Dict] = None,
                         eps_start: float = 0.0002, eps_end: float = 0.0035,
                         n_points: int = 25, verbose: bool = True,
                         include_tension: bool = True):
    """Generate N-ε curve - optimized."""
    Lx, Ly = section_params['Lx'], section_params['Ly']
    cx, cy = Lx / 2, Ly / 2
    
    N_scale = section_params['fc'] * Lx * Ly
    M_scale = N_scale * max(Lx, Ly) / 2
    
    if confinement_params is not None:
        eps_end = min(eps_end, confinement_params['eps_cu'])
    
    eps_list, N_list, Mx_list, My_list = [], [], [], []
    cracked_list = []
    
    strains = np.linspace(eps_start, eps_end, n_points)
    last_kx, last_ky, last_eps = 0.0, 0.0, eps_start
    
    # Detect extreme eccentricity
    L_diag = np.sqrt(Lx**2 + Ly**2)
    is_extreme_ex = abs(e0x) > 5 * Lx
    is_extreme_ey = abs(e0y) > 5 * Ly
    is_extreme = is_extreme_ex or is_extreme_ey
    
    # For uniaxial, one eccentricity should be ~0
    is_uniaxial_x = abs(e0x) < 0.1 and abs(e0y) > 0.1
    is_uniaxial_y = abs(e0y) < 0.1 and abs(e0x) > 0.1
    
    if verbose:
        conf_str = "(CONFINED)" if confinement_params else "(UNCONFINED)"
        print(f"\n{'='*80}")
        print(f"N-ε ANALYSIS {conf_str}")
        if is_extreme:
            print(f"  *** EXTREME ECCENTRICITY MODE (N → 0) ***")
        if is_uniaxial_x:
            print(f"  *** UNIAXIAL BENDING ABOUT X (ky=0, My=0) ***")
        if is_uniaxial_y:
            print(f"  *** UNIAXIAL BENDING ABOUT Y (kx=0, Mx=0) ***")
        print(f"{'='*80}")
        print(f"{'ε[‰]':>8} {'N[kN]':>12} {'Mx[kN·m]':>12} {'My[kN·m]':>12} {'Status':>10}")
        print(f"{'-'*80}")
    
    for eps_max in strains:
        
        # =================================================================
        # UNIAXIAL X: Only optimize kx, force ky=0
        # =================================================================
        if is_uniaxial_x:
            def objective_1d(kx_val):
                kx, ky = kx_val, 0.0
                corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
                contrib = kx * (corners[:, 1] - cy)
                eps_0 = eps_max - np.max(contrib)
                # For uniaxial, allow negative eps_0 (centroid in tension)
                # but limit to prevent numerical issues
                eps_0 = max(eps_0, -0.02)  # Allow up to 20‰ tension at centroid
                
                try:
                    N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                          confinement_params, include_tension)
                except:
                    return 1e20, 0, 0
                
                if not np.isfinite(N + Mx):
                    return 1e20, 0, 0
                
                # Equilibrium: Mx = N * e0y
                return ((Mx - N * e0y) / M_scale)**2, N, Mx
            
            # 1D search for kx with continuity from previous solution
            sy = np.sign(e0y)
            kappa_base = eps_max / Ly
            
            # Start from previous solution if available
            if len(eps_list) > 0 and abs(last_kx) > 1e-12:
                # Scale previous solution
                scale = eps_max / last_eps if last_eps > 1e-10 else 1.0
                start_kx = last_kx * scale
                
                # Search around previous solution first
                best_kx, best_f, best_N, best_Mx = start_kx, float('inf'), 0, 0
                for mult in np.linspace(0.5, 2.0, 30):
                    kx_test = start_kx * mult
                    if abs(kx_test) > 1e-12:
                        f, N_test, Mx_test = objective_1d(kx_test)
                        if f < best_f:
                            best_f = f
                            best_kx = kx_test
                            best_N, best_Mx = N_test, Mx_test
            else:
                best_kx, best_f, best_N, best_Mx = 0.0, float('inf'), 0, 0
            
            # Wide search
            for mag in np.logspace(-4, 2, 50) * kappa_base:
                kx_test = mag * sy
                f, N_test, Mx_test = objective_1d(kx_test)
                if f < best_f:
                    best_f = f
                    best_kx = kx_test
                    best_N, best_Mx = N_test, Mx_test
            
            # Fine refinement
            for _ in range(5):
                improved = False
                for delta in [0.2, 0.1, 0.05, 0.02, 0.01]:
                    for mult in [1-delta, 1+delta]:
                        kx_test = best_kx * mult
                        if abs(kx_test) > 1e-12:
                            f, N_test, Mx_test = objective_1d(kx_test)
                            if f < best_f:
                                best_f = f
                                best_kx = kx_test
                                best_N, best_Mx = N_test, Mx_test
                                improved = True
                if not improved:
                    break
            
            kx_opt, ky_opt = best_kx, 0.0
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = kx_opt * (corners[:, 1] - cy)
            eps_0 = max(eps_max - np.max(contrib), -0.02)
            
            try:
                N, Mx, My, _, is_cracked, _ = analyze_section(
                    eps_0, kx_opt, ky_opt, section_params, confinement_params, include_tension)
            except:
                continue
            
            eps_list.append(eps_max)
            N_list.append(N)
            Mx_list.append(Mx)
            My_list.append(My)
            cracked_list.append(is_cracked)
            last_kx, last_ky, last_eps = kx_opt, ky_opt, eps_max
            
            if verbose:
                status = "cracked" if is_cracked else "ok"
                # Check equilibrium quality
                eq_x = abs(Mx/(N*e0y) - 1) if abs(N*e0y) > 1 else 0
                eq_y = abs(My/(N*e0x) - 1) if abs(N*e0x) > 1 else 0
                eq_err = max(eq_x, eq_y)
                if eq_err > 0.05:  # More than 5% error
                    status += "*"  # Mark with asterisk
                print(f"{eps_max*1000:8.3f} {N/1000:12.2f} {Mx/1e6:12.3f} {My/1e6:12.3f} {status:>10}")
            continue
        
        # =================================================================
        # UNIAXIAL Y: Only optimize ky, force kx=0
        # =================================================================
        if is_uniaxial_y:
            def objective_1d(ky_val):
                kx, ky = 0.0, ky_val
                corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
                contrib = ky * (corners[:, 0] - cx)
                eps_0 = eps_max - np.max(contrib)
                # Allow negative eps_0 for extreme eccentricity
                eps_0 = max(eps_0, -0.02)
                
                try:
                    N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                          confinement_params, include_tension)
                except:
                    return 1e20, 0, 0
                
                if not np.isfinite(N + My):
                    return 1e20, 0, 0
                
                # Equilibrium: My = N * e0x
                return ((My - N * e0x) / M_scale)**2, N, My
            
            # 1D search for ky with continuity
            sx = np.sign(e0x)
            kappa_base = eps_max / Lx
            
            if len(eps_list) > 0 and abs(last_ky) > 1e-12:
                scale = eps_max / last_eps if last_eps > 1e-10 else 1.0
                start_ky = last_ky * scale
                
                best_ky, best_f = start_ky, float('inf')
                for mult in np.linspace(0.5, 2.0, 30):
                    ky_test = start_ky * mult
                    if abs(ky_test) > 1e-12:
                        f, _, _ = objective_1d(ky_test)
                        if f < best_f:
                            best_f = f
                            best_ky = ky_test
            else:
                best_ky, best_f = 0.0, float('inf')
            
            for mag in np.logspace(-4, 2, 50) * kappa_base:
                ky_test = mag * sx
                f, _, _ = objective_1d(ky_test)
                if f < best_f:
                    best_f = f
                    best_ky = ky_test
            
            # Fine refinement
            for _ in range(5):
                improved = False
                for delta in [0.2, 0.1, 0.05, 0.02, 0.01]:
                    for mult in [1-delta, 1+delta]:
                        ky_test = best_ky * mult
                        if abs(ky_test) > 1e-12:
                            f, _, _ = objective_1d(ky_test)
                            if f < best_f:
                                best_f = f
                                best_ky = ky_test
                                improved = True
                if not improved:
                    break
            
            kx_opt, ky_opt = 0.0, best_ky
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = ky_opt * (corners[:, 0] - cx)
            eps_0 = max(eps_max - np.max(contrib), -0.02)
            
            try:
                N, Mx, My, _, is_cracked, _ = analyze_section(
                    eps_0, kx_opt, ky_opt, section_params, confinement_params, include_tension)
            except:
                continue
            
            eps_list.append(eps_max)
            N_list.append(N)
            Mx_list.append(Mx)
            My_list.append(My)
            cracked_list.append(is_cracked)
            last_kx, last_ky, last_eps = kx_opt, ky_opt, eps_max
            
            if verbose:
                status = "cracked" if is_cracked else "ok"
                eq_x = abs(Mx/(N*e0y) - 1) if abs(N*e0y) > 1 else 0
                eq_y = abs(My/(N*e0x) - 1) if abs(N*e0x) > 1 else 0
                eq_err = max(eq_x, eq_y)
                if eq_err > 0.05:
                    status += "*"
                print(f"{eps_max*1000:8.3f} {N/1000:12.2f} {Mx/1e6:12.3f} {My/1e6:12.3f} {status:>10}")
            continue
        
        # =================================================================
        # BIAXIAL: 2D optimization
        # =================================================================
        def objective(x):
            kx, ky = x
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = kx * (corners[:, 1] - cy) + ky * (corners[:, 0] - cx)
            eps_0 = eps_max - np.max(contrib)
            
            # For extreme eccentricity, allow negative eps_0
            if is_extreme:
                if eps_0 < -eps_max:
                    return 1e20
            else:
                if eps_0 < -eps_max * 0.5:
                    return 1e20
            eps_0 = max(eps_0, -eps_max * 0.9) if is_extreme else max(eps_0, 1e-10)
            
            try:
                N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                      confinement_params, include_tension)
            except:
                return 1e20
            
            if not np.isfinite(N + Mx + My):
                return 1e20
            
            # EXTREME ECCENTRICITY: Drive N → 0, maintain moment ratio
            if is_extreme:
                N_penalty = (N / N_scale) ** 2 * 1000
                
                if abs(e0x) > 0.1 and abs(e0y) > 0.1:
                    target_ratio = e0y / e0x
                    if abs(My) > 1:
                        actual_ratio = Mx / My
                        ratio_error = ((actual_ratio - target_ratio) / (1 + abs(target_ratio))) ** 2
                    else:
                        ratio_error = 0 if abs(Mx) < 1 else 1
                else:
                    ratio_error = 0
                
                return N_penalty + ratio_error * 10
            
            # NORMAL BIAXIAL: Standard equilibrium
            return ((Mx - N * e0y) / M_scale)**2 + ((My - N * e0x) / M_scale)**2
        
        # Initial guess
        if len(eps_list) == 0:
            x0 = grid_search(objective, eps_max, Lx, Ly, e0x, e0y)
        else:
            scale = eps_max / last_eps if last_eps > 1e-10 else 1.0
            x0 = np.array([last_kx * scale, last_ky * scale])
            if objective(x0) > 0.01:
                x0 = grid_search(objective, eps_max, Lx, Ly, e0x, e0y)
        
        x_opt, err = nelder_mead(objective, x0, tol=1e-10, max_iter=300)
        
        if err > 1e-6:
            # Retry with different scale
            x_opt2, err2 = nelder_mead(objective, x_opt * 0.5, tol=1e-10, max_iter=200)
            if err2 < err:
                x_opt, err = x_opt2, err2
        
        if err > 1e8:
            continue
        
        kx_opt, ky_opt = x_opt
        corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
        contrib = kx_opt * (corners[:, 1] - cy) + ky_opt * (corners[:, 0] - cx)
        eps_0 = max(eps_max - np.max(contrib), 1e-10)
        
        try:
            N, Mx, My, _, is_cracked, _ = analyze_section(
                eps_0, kx_opt, ky_opt, section_params, confinement_params, include_tension)
        except:
            continue
        
        eps_list.append(eps_max)
        N_list.append(N)
        Mx_list.append(Mx)
        My_list.append(My)
        cracked_list.append(is_cracked)
        
        last_kx, last_ky, last_eps = kx_opt, ky_opt, eps_max
        
        if verbose:
            status = "cracked" if is_cracked else "ok"
            # Check equilibrium quality
            eq_x = abs(Mx/(N*e0y) - 1) if abs(N*e0y) > 1 else 0
            eq_y = abs(My/(N*e0x) - 1) if abs(N*e0x) > 1 else 0
            eq_err = max(eq_x, eq_y)
            if eq_err > 0.05:  # More than 5% error
                status += "*"  # Mark with asterisk
            print(f"{eps_max*1000:8.3f} {N/1000:12.2f} {Mx/1e6:12.3f} {My/1e6:12.3f} {status:>10}")
    
    if verbose:
        print(f"{'-'*80}")
        print(f"Complete: {len(eps_list)}/{n_points} points")
        # Check equilibrium quality
        if eps_list:
            eq_errors = []
            for N, Mx, My in zip(N_list, Mx_list, My_list):
                eq_x = abs(Mx/(N*e0y) - 1) if abs(N*e0y) > 1 else 0
                eq_y = abs(My/(N*e0x) - 1) if abs(N*e0x) > 1 else 0
                eq_errors.append(max(eq_x, eq_y))
            max_eq_err = max(eq_errors)
            if max_eq_err > 0.05:
                print(f"⚠ WARNING: Max equilibrium error = {max_eq_err*100:.1f}%")
                print(f"  (Eccentricity may not be fully achievable for this section)")
        print(f"{'='*80}\n")
    
    return eps_list, N_list, Mx_list, My_list, {'cracked_points': cracked_list}


# ============================================================================
# PART 8: PLOTTING 
# ============================================================================
def plot_N_M_curves(eps_list, N_list, Mx_list, My_list, section_params,
                    confinement_params=None, info=None, filename=None, show=True):
    """
    Plot N-ε and M-ε curves with a DISTINCTIVE CIRCULAR MARKER for the cracking point.
    """
    if not eps_list:
        print("No data to plot!")
        return None
    
    eps_permille = np.array(eps_list) * 1000
    N_kN = np.array(N_list) / 1000
    Mx_kNm = np.array(Mx_list) / 1e6
    My_kNm = np.array(My_list) / 1e6
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    
    # --- LOCATE CRACKING POINT ---
    crack_idx = -1
    if info and 'cracked_points' in info:
        cracked_list = info['cracked_points']
        for i, is_cracked in enumerate(cracked_list):
            if is_cracked:
                crack_idx = i
                break
    
    # =========================================================================
    # PLOT 1: Axial Force vs Strain (N-ε)
    # =========================================================================
    axes[0].plot(eps_permille, N_kN, 'b-', lw=2.5, label='N')
    
    # Draw Cracking Point Marker (MODIFIED STYLE)
    if crack_idx != -1:
        # Style: Gold Circle with Black Edge
        axes[0].plot(eps_permille[crack_idx], N_kN[crack_idx], 
                     marker='o',              # Shape: Circle
                     linestyle='None',        # No connecting line
                     markerfacecolor='gold',  # Fill color: Gold (Distinctive)
                     markeredgecolor='black', # Border color: Black
                     markeredgewidth=1.5,     # Border thickness
                     markersize=10,           # Size
                     label='Cracking Point', 
                     zorder=10)
        
        # Annotation text
        axes[0].annotate('Cracking', 
                         xy=(eps_permille[crack_idx], N_kN[crack_idx]), 
                         xytext=(eps_permille[crack_idx] + 0.5, N_kN[crack_idx]),
                         arrowprops=dict(facecolor='black', arrowstyle='->'), 
                         fontsize=9, fontweight='bold')

    axes[0].set_xlabel('ε [‰]', fontsize=12)
    axes[0].set_ylabel('N [kN]', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].set_title('Axial Force vs Strain', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # PLOT 2: Moments vs Strain (Mx, My-ε)
    # =========================================================================
    axes[1].plot(eps_permille, Mx_kNm, 'r-', lw=2.5, label='Mx')
    axes[1].plot(eps_permille, My_kNm, 'g-', lw=2.5, label='My')
    
    # Draw Cracking Point Markers for Moments (SAME STYLE)
    if crack_idx != -1:
        # Plot for Mx
        axes[1].plot(eps_permille[crack_idx], Mx_kNm[crack_idx], 
                     marker='o', linestyle='None',
                     markerfacecolor='gold', markeredgecolor='black',
                     markeredgewidth=1.5, markersize=10, zorder=10)
        
        # Plot for My
        axes[1].plot(eps_permille[crack_idx], My_kNm[crack_idx], 
                     marker='o', linestyle='None',
                     markerfacecolor='gold', markeredgecolor='black',
                     markeredgewidth=1.5, markersize=10, zorder=10)

    axes[1].set_xlabel('ε [‰]', fontsize=12)
    axes[1].set_ylabel('M [kN·m]', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    axes[1].legend(frameon=False, fontsize=11)
    axes[1].set_title('Bending Moments vs Strain', fontsize=12, fontweight='bold')
    
    # --- Title Construction ---
    conf_str = "CONFINED" if confinement_params else "UNCONFINED"
    title = f"{section_params['Lx']:.0f}×{section_params['Ly']:.0f} mm | fc={section_params['fc']:.0f} MPa"
    if confinement_params:
        title += f" | fcc={confinement_params['fcc']:.1f} MPa"
    title += f" | [{conf_str}]"
    fig.suptitle(title, fontsize=11, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
    
    if show:
        plt.show()
    
    return fig


def plot_section_heatmap(eps_0: float, kappa_x: float, kappa_y: float,
                          section_params: Dict,
                          confinement_params: Optional[Dict] = None,
                          n_grid: int = 60,
                          filename: Optional[str] = None,
                          show: bool = True):
    """
    Plot cross-section stress/strain heatmaps.
    FIXED: Colormap properly centered at zero using TwoSlopeNorm.
    """
    Lx, Ly = section_params['Lx'], section_params['Ly']
    fc, Ec = section_params['fc'], section_params['Ec']
    eps_c0 = section_params['eps_c0']
    cx, cy = Lx / 2, Ly / 2
    
    # Create grid
    x = np.linspace(0, Lx, n_grid)
    y = np.linspace(0, Ly, n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Strain field
    strain_field = eps_0 + kappa_x * (Y - cy) + kappa_y * (X - cx)
    
    # Stress field
    stress_field = np.zeros_like(strain_field)
    
    if confinement_params is not None:
        core_x_min = confinement_params['core_x_min']
        core_x_max = confinement_params['core_x_max']
        core_y_min = confinement_params['core_y_min']
        core_y_max = confinement_params['core_y_max']
        fcc = confinement_params['fcc']
        eps_cc = confinement_params['eps_cc']
        Ec_conf = confinement_params['Ec']
        r = confinement_params['r']
        eps_spall = max(0.004, 2 * eps_c0)
        
        # Vectorized stress calculation
        for i in range(n_grid):
            for j in range(n_grid):
                xi, yi = X[i, j], Y[i, j]
                eps = strain_field[i, j]
                in_core = (core_x_min <= xi <= core_x_max and 
                          core_y_min <= yi <= core_y_max)
                
                if eps > 0:  # Compression
                    if in_core:
                        stress_field[i, j] = mander_confined_stress(
                            np.array([eps]), fcc, eps_cc, Ec_conf, r)[0]
                    else:
                        stress_field[i, j] = unconfined_spalling_stress(
                            np.array([eps]), fc, eps_c0, eps_spall)[0]
                else:  # Tension
                    stress_field[i, j] = concrete_stress_with_tension(
                        np.array([eps]), fc, Ec, eps_c0)[0]
    else:
        # Unconfined - vectorized
        stress_field = concrete_stress_with_tension(strain_field.ravel(), fc, Ec, eps_c0).reshape(strain_field.shape)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # =========================================================================
    # STRESS HEATMAP - FIXED: Zero-centered with SYMMETRIC scaling
    # =========================================================================
    ax1 = axes[0]
    
    stress_min = np.min(stress_field)
    stress_max = np.max(stress_field)
    
    # SYMMETRIC scaling: same absolute value for + and -
    stress_abs_max = max(abs(stress_min), abs(stress_max), 0.1)
    
    # Handle case where all values are same sign
    if stress_min >= 0:
        # All compression or zero
        norm1 = plt.Normalize(vmin=0, vmax=stress_abs_max)
        cmap1 = plt.cm.Reds
    elif stress_max <= 0:
        # All tension
        norm1 = plt.Normalize(vmin=-stress_abs_max, vmax=0)
        cmap1 = plt.cm.Blues_r
    else:
        # Mixed: SYMMETRIC limits around zero
        norm1 = TwoSlopeNorm(vmin=-stress_abs_max, vcenter=0, vmax=stress_abs_max)
        cmap1 = plt.cm.RdBu_r  # Red=positive(compression), Blue=negative(tension)
    
    im1 = ax1.pcolormesh(X, Y, stress_field, cmap=cmap1, norm=norm1, shading='auto')
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label('Stress σ [MPa]\n(+compression, -tension)', fontsize=10)
    
    # Neutral axis
    if stress_min < 0 < stress_max:
        cs1 = ax1.contour(X, Y, strain_field, levels=[0], colors='black', 
                          linewidths=2.5, linestyles='--')
        ax1.clabel(cs1, fmt='N.A.', fontsize=9)
    
    # Core boundary
    if confinement_params is not None:
        rect = plt.Rectangle(
            (confinement_params['core_x_min'], confinement_params['core_y_min']),
            confinement_params['bc_x'], confinement_params['bc_y'],
            fill=False, edgecolor='green', linewidth=2.5, linestyle='-', label='Core')
        ax1.add_patch(rect)
        ax1.legend(loc='upper right', fontsize=9)
    
    # Section outline and reinforcement
    ax1.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], 'k-', lw=2)
    for xi, yi, di in zip(section_params['rein_x'], section_params['rein_y'], 
                          section_params['rein_ds']):
        ax1.add_patch(plt.Circle((xi, yi), di/2, color='#333', fill=True, zorder=10))
    
    ax1.set_xlim(-Lx*0.05, Lx*1.05)
    ax1.set_ylim(-Ly*0.05, Ly*1.05)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [mm]', fontsize=11)
    ax1.set_ylabel('Y [mm]', fontsize=11)
    ax1.set_title('Stress Distribution', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # STRAIN HEATMAP - FIXED: Zero-centered with SYMMETRIC scaling
    # =========================================================================
    ax2 = axes[1]
    
    strain_permille = strain_field * 1000
    strain_min = np.min(strain_permille)
    strain_max = np.max(strain_permille)
    
    # SYMMETRIC scaling: same absolute value for + and -
    strain_abs_max = max(abs(strain_min), abs(strain_max), 0.1)
    
    if strain_min >= 0:
        norm2 = plt.Normalize(vmin=0, vmax=strain_abs_max)
        cmap2 = plt.cm.Oranges
    elif strain_max <= 0:
        norm2 = plt.Normalize(vmin=-strain_abs_max, vmax=0)
        cmap2 = plt.cm.Blues_r
    else:
        # Mixed: SYMMETRIC limits around zero
        norm2 = TwoSlopeNorm(vmin=-strain_abs_max, vcenter=0, vmax=strain_abs_max)
        cmap2 = plt.cm.RdBu_r
    
    im2 = ax2.pcolormesh(X, Y, strain_permille, cmap=cmap2, norm=norm2, shading='auto')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label('Strain ε [‰]\n(+compression, -tension)', fontsize=10)
    
    # Contours
    n_levels = 8
    levels = np.linspace(strain_min, strain_max, n_levels)
    levels = levels[levels != 0]  # Remove zero to avoid double line
    if len(levels) > 0:
        cs2 = ax2.contour(X, Y, strain_permille, levels=levels, colors='gray', 
                          linewidths=0.8, alpha=0.7)
        ax2.clabel(cs2, fmt='%.1f', fontsize=8)
    
    # Neutral axis
    if strain_min < 0 < strain_max:
        cs_na = ax2.contour(X, Y, strain_field, levels=[0], colors='black',
                            linewidths=2.5, linestyles='--')
    
    # Core boundary
    if confinement_params is not None:
        rect2 = plt.Rectangle(
            (confinement_params['core_x_min'], confinement_params['core_y_min']),
            confinement_params['bc_x'], confinement_params['bc_y'],
            fill=False, edgecolor='green', linewidth=2.5)
        ax2.add_patch(rect2)
    
    # Section outline and reinforcement (colored by strain)
    ax2.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], 'k-', lw=2)
    for xi, yi, di in zip(section_params['rein_x'], section_params['rein_y'],
                          section_params['rein_ds']):
        eps_bar = eps_0 + kappa_x * (yi - cy) + kappa_y * (xi - cx)
        color = '#dc2626' if eps_bar > 0 else '#2563eb'
        ax2.add_patch(plt.Circle((xi, yi), di/2, color=color, fill=True, zorder=10))
    
    ax2.set_xlim(-Lx*0.05, Lx*1.05)
    ax2.set_ylim(-Ly*0.05, Ly*1.05)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X [mm]', fontsize=11)
    ax2.set_ylabel('Y [mm]', fontsize=11)
    ax2.set_title('Strain Distribution', fontsize=12, fontweight='bold')
    
    # Title
    conf_str = "CONFINED" if confinement_params else "UNCONFINED"
    eps_max = np.max(strain_field)
    title = f"Section Analysis [{conf_str}] | ε_max = {eps_max*1000:.2f}‰"
    fig.suptitle(title, fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
    
    if show:
        plt.show()
    
    return fig


def plot_section_at_strain(eps_max: float, e0x: float, e0y: float,
                            section_params: Dict,
                            confinement_params: Optional[Dict] = None,
                            include_tension: bool = True,
                            filename: Optional[str] = None,
                            show: bool = True):
    """
    Solve equilibrium at given max strain and plot section.
    Uses 1D optimization for uniaxial cases, 2D for biaxial.
    """
    Lx, Ly = section_params['Lx'], section_params['Ly']
    cx, cy = Lx / 2, Ly / 2
    
    N_scale = section_params['fc'] * Lx * Ly
    M_scale = N_scale * max(Lx, Ly) / 2
    
    # Detect case type
    abs_ex, abs_ey = abs(e0x), abs(e0y)
    is_uniaxial_x = abs_ex < 0.1 and abs_ey > 0.1
    is_uniaxial_y = abs_ey < 0.1 and abs_ex > 0.1
    
    # ==========================================================================
    # UNIAXIAL X: Use 1D optimization (ky = 0)
    # ==========================================================================
    if is_uniaxial_x:
        def objective_1d(kx_val):
            kx, ky = kx_val, 0.0
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = kx * (corners[:, 1] - cy)
            eps_0 = eps_max - np.max(contrib)
            # Allow negative eps_0 for extreme eccentricity
            eps_0 = max(eps_0, -0.02)
            
            try:
                N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                      confinement_params, include_tension)
            except:
                return 1e20
            
            if not np.isfinite(N + Mx):
                return 1e20
            
            return ((Mx - N * e0y) / M_scale)**2
        
        # 1D search
        sy = np.sign(e0y)
        kappa_base = eps_max / Ly
        best_kx, best_f = 0.0, float('inf')
        
        for mag in np.logspace(-4, 2, 60) * kappa_base:
            kx_test = mag * sy
            f = objective_1d(kx_test)
            if f < best_f:
                best_f = f
                best_kx = kx_test
        
        # Refine
        for _ in range(3):
            for delta in [0.3, 0.1, 0.03]:
                for mult in [1-delta, 1+delta]:
                    kx_test = best_kx * mult
                    if kx_test != 0:
                        f = objective_1d(kx_test)
                        if f < best_f:
                            best_f = f
                            best_kx = kx_test
        
        kx, ky = best_kx, 0.0
        
    # ==========================================================================
    # UNIAXIAL Y: Use 1D optimization (kx = 0)
    # ==========================================================================
    elif is_uniaxial_y:
        def objective_1d(ky_val):
            kx, ky = 0.0, ky_val
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = ky * (corners[:, 0] - cx)
            eps_0 = eps_max - np.max(contrib)
            # Allow negative eps_0 for extreme eccentricity
            eps_0 = max(eps_0, -0.02)
            
            try:
                N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                      confinement_params, include_tension)
            except:
                return 1e20
            
            if not np.isfinite(N + My):
                return 1e20
            
            return ((My - N * e0x) / M_scale)**2
        
        # 1D search
        sx = np.sign(e0x)
        kappa_base = eps_max / Lx
        best_ky, best_f = 0.0, float('inf')
        
        for mag in np.logspace(-4, 2, 60) * kappa_base:
            ky_test = mag * sx
            f = objective_1d(ky_test)
            if f < best_f:
                best_f = f
                best_ky = ky_test
        
        # Refine
        for _ in range(3):
            for delta in [0.3, 0.1, 0.03]:
                for mult in [1-delta, 1+delta]:
                    ky_test = best_ky * mult
                    if ky_test != 0:
                        f = objective_1d(ky_test)
                        if f < best_f:
                            best_f = f
                            best_ky = ky_test
        
        kx, ky = 0.0, best_ky
        
    # ==========================================================================
    # BIAXIAL: Use 2D optimization
    # ==========================================================================
    else:
        def objective(x):
            kx, ky = x
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = kx * (corners[:, 1] - cy) + ky * (corners[:, 0] - cx)
            eps_0 = eps_max - np.max(contrib)
            
            if eps_0 < -eps_max * 0.5:
                return 1e20
            eps_0 = max(eps_0, 1e-10)
            
            try:
                N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                      confinement_params, include_tension)
            except:
                return 1e20
            
            return ((Mx - N * e0y) / M_scale)**2 + ((My - N * e0x) / M_scale)**2
        
        x0 = grid_search(objective, eps_max, Lx, Ly, e0x, e0y)
        x_opt, _ = nelder_mead(objective, x0, tol=1e-10, max_iter=400)
        kx, ky = x_opt
    
    # Calculate eps_0 from final curvatures
    corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
    contrib = kx * (corners[:, 1] - cy) + ky * (corners[:, 0] - cx)
    eps_0 = eps_max - np.max(contrib)
    # Allow negative eps_0 for extreme eccentricity
    if is_uniaxial_x or is_uniaxial_y:
        eps_0 = max(eps_0, -0.02)
    else:
        eps_0 = max(eps_0, 1e-10)
    
    N, Mx, My, eps_s, is_cracked, _ = analyze_section(
        eps_0, kx, ky, section_params, confinement_params, include_tension)
    
    # Plot
    fig = plot_section_heatmap(eps_0, kx, ky, section_params, confinement_params,
                                filename=filename, show=show)
    
    print(f"\nSection State at ε_max = {eps_max*1000:.2f}‰:")
    print(f"  N  = {N/1000:.2f} kN")
    print(f"  Mx = {Mx/1e6:.3f} kN·m")
    print(f"  My = {My/1e6:.3f} kN·m")
    print(f"  Cracked: {'Yes' if is_cracked else 'No'}")
    
    # Equilibrium check
    if abs_ey > 0.1:
        eq_x = Mx / (N * e0y) if abs(N * e0y) > 1 else 0
        print(f"  Mx/(N·ey) = {eq_x:.4f} (target: 1.0)")
    if abs_ex > 0.1:
        eq_y = My / (N * e0x) if abs(N * e0x) > 1 else 0
        print(f"  My/(N·ex) = {eq_y:.4f} (target: 1.0)")
    
    return fig, {'N': N, 'Mx': Mx, 'My': My, 'eps_0': eps_0, 'kappa_x': kx, 'kappa_y': ky}






# ============================================================================
# PART 9: Mx-My INTERACTION CURVE GENERATOR 
# ============================================================================

def generate_Mx_My_interaction(N_target: float, section_params: Dict,
                                confinement_params: Optional[Dict] = None,
                                eps_max: float = 0.0035,
                                n_points: int = 72,
                                include_tension: bool = True,
                                verbose: bool = True) -> Tuple[List, List]:
    """
    Generate Mx-My interaction curve at constant N using Neutral Axis Rotation.
    Guarantees a convex hull and avoids numerical errors.
    """
    Lx, Ly = section_params['Lx'], section_params['Ly']
    cx, cy = Lx / 2, Ly / 2
    
    # Scale for convergence check
    N_scale = section_params['fc'] * Lx * Ly
    
    Mx_list, My_list = [], []
    
    # Scan Neutral Axis (NA) angle from 0 to 360 degrees
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Mx-My INTERACTION (Neutral Axis Rotation Method)")
        print(f"Target Axial Load N = {N_target/1000:.1f} kN")
        print(f"{'='*80}")
        print(f"{'NA Angle[°]':>12} {'Mx[kN·m]':>12} {'My[kN·m]':>12} {'Error N[%]':>12}")
        print(f"{'-'*80}")

    for theta in angles:
        # Define curvature direction
        dir_x = np.sin(theta)
        dir_y = np.cos(theta)
        
        # Function to find curvature magnitude 'mag' that satisfies N_target
        def solve_for_N(mag):
            kx = mag * dir_x
            ky = mag * dir_y
            
            # Calculate eps_0 to maintain eps_max at extreme compression fiber
            corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
            contrib = kx * (corners[:, 1] - cy) + ky * (corners[:, 0] - cx)
            eps_0 = eps_max - np.max(contrib)
            
            try:
                N, _, _, _, _, _ = analyze_section(eps_0, kx, ky, section_params,
                                                    confinement_params, include_tension)
                return N - N_target
            except:
                return 1e20

        # Search for the correct curvature using Bisection
        # Range estimation: max curvature approx 10 * yield curvature
        low, high = 0.0, (eps_max / min(Lx, Ly)) * 20
        best_mag = 0.0
        
        # Check bounds
        f_low = solve_for_N(low + 1e-9)
        f_high = solve_for_N(high)
        
        if f_low * f_high > 0:
            # If target N is not reachable in this range, pick closest
            best_mag = low if abs(f_low) < abs(f_high) else high
        else:
            # Bisection Loop
            for _ in range(30):
                mid = (low + high) / 2
                f_mid = solve_for_N(mid)
                
                if abs(f_mid) < N_scale * 1e-4: # Tolerance
                    best_mag = mid
                    break
                
                if f_low * f_mid < 0:
                    high = mid
                    f_high = f_mid
                else:
                    low = mid
                    f_low = f_mid
            best_mag = (low + high) / 2
        
        # Calculate final moments
        kx = best_mag * dir_x
        ky = best_mag * dir_y
        corners = np.array([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)])
        eps_0 = eps_max - np.max(kx * (corners[:, 1] - cy) + ky * (corners[:, 0] - cx))
        
        N, Mx, My, _, _, _ = analyze_section(eps_0, kx, ky, section_params, 
                                             confinement_params, include_tension)
        
        Mx_list.append(Mx)
        My_list.append(My)
        
        if verbose:
            err_p = (N - N_target) / N_target * 100 if abs(N_target)>1 else 0
            # Only print every 4th point to save space
            if int(np.degrees(theta)) % 20 == 0: 
                 print(f"{np.degrees(theta):12.0f} {Mx/1e6:12.3f} {My/1e6:12.3f} {err_p:12.2f}")

    if verbose:
        print(f"{'='*80}\n")
    
    return Mx_list, My_list

def plot_Mx_My_interaction(Np: float, Mxp: float, Myp: float,
                            section_params: Dict,
                            confinement_params: Optional[Dict] = None,
                            eps_max: float = 0.0035,
                            n_points: int = 72,
                            filename: Optional[str] = None,
                            show: bool = True):
    """Plot the Interaction Diagram."""
    
    # Generate the curve
    Mx_list, My_list = generate_Mx_My_interaction(
        Np, section_params, confinement_params, eps_max, n_points, verbose=True)
    
    Mx_kNm = np.array(Mx_list) / 1e6
    My_kNm = np.array(My_list) / 1e6
    Mxp_kNm = Mxp / 1e6
    Myp_kNm = Myp / 1e6
    
    # Close the loop for plotting
    Mx_kNm = np.concatenate([Mx_kNm, [Mx_kNm[0]]])
    My_kNm = np.concatenate([My_kNm, [My_kNm[0]]])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Capacity Curve
    ax.plot(My_kNm, Mx_kNm, 'b-', lw=2, label='Capacity')
    ax.fill(My_kNm, Mx_kNm, 'b', alpha=0.1)
    
    # Applied Load Point
    from matplotlib.path import Path
    curve_path = Path(np.column_stack([My_kNm, Mx_kNm]))
    is_safe = curve_path.contains_point([Myp_kNm, Mxp_kNm])
    
    color = 'green' if is_safe else 'red'
    status = "SAFE" if is_safe else "UNSAFE"
    
    ax.plot(Myp_kNm, Mxp_kNm, 'o', color=color, markersize=10, label=f'Applied ({status})')
    
    # Add axes lines
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    
    ax.set_xlabel('My [kN·m]')
    ax.set_ylabel('Mx [kN·m]')
    ax.set_title(f'Biaxial Interaction at N = {Np/1000:.0f} kN')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()





# ============================================================================
# PART 10: 3D INTERACTION CURVE GENERATOR 
# ============================================================================
def generate_3d_interaction_surface(section_params: Dict,
                                    confinement_params: Optional[Dict] = None,
                                    n_slices: int = 15,  
                                    n_points: int = 60,  
                                    eps_max: float = 0.0035):
    """
    Generate data for 3D interaction surface by stacking 2D contours.
    """
    Lx, Ly = section_params['Lx'], section_params['Ly']
    
    # 1. Calculate limits (Pure Compression & Pure Tension)
    # Pure Compression (uniform strain eps_c0)
    eps_c0 = section_params['eps_c0']
    N_max, _, _, _, _, _ = analyze_section(eps_c0, 0, 0, section_params, confinement_params)
    
    # Pure Tension (uniform strain -0.01 or similar large value)
    N_min, _, _, _, _, _ = analyze_section(-0.02, 0, 0, section_params, confinement_params)
    
    print(f"Generating 3D Surface:")
    print(f"  P_max (Compression) = {N_max/1000:.0f} kN")
    print(f"  P_min (Tension)     = {N_min/1000:.0f} kN")
    
    # 2. Define N levels to slice
    # We concentrate points near the "balanced" region (middle) for better shape
    # Using a sine distribution or simple linear
    N_levels = np.linspace(N_min * 0.99, N_max * 0.99, n_slices)
    
    Nx_data, Ny_data, Nz_data = [], [], []
    
    # 3. Generate loops for each level
    for i, N_target in enumerate(reversed(N_levels)): # Top to bottom
        print(f"  Slice {i+1}/{n_slices}: N = {N_target/1000:.0f} kN...")
        
        # Use our robust generator
        Mx_row, My_row = generate_Mx_My_interaction(
            N_target, section_params, confinement_params, 
            eps_max=eps_max, n_points=n_points, verbose=False
        )
        
        # Close the loop
        Mx_row.append(Mx_row[0])
        My_row.append(My_row[0])
        
        Nx_data.append(np.array(Mx_row) / 1e6) # Mx [kNm]
        Ny_data.append(np.array(My_row) / 1e6) # My [kNm]
        Nz_data.append(np.full_like(Mx_row, N_target / 1000)) # P [kN]
        
    return np.array(Nx_data), np.array(Ny_data), np.array(Nz_data)

def plot_3d_interaction(section_params: Dict,
                        confinement_params: Optional[Dict] = None,
                        applied_N: float = 0.0,    # Input in [N]
                        applied_Mx: float = 0.0,   # Input in [N.mm]
                        applied_My: float = 0.0):  # Input in [N.mm]
    """
    Plots the 3D Interaction Surface (P-Mx-My) and checks the applied load.
    Includes aspect ratio correction to prevent slender sections from looking wide.
    """
    
    print("Generating 3D Surface data... (This may take a moment)")
    
    # 1. Generate the 3D Surface Data (Visualization only)
    #    X=Mx, Y=My, Z=P (Axial)
    X, Y, Z = generate_3d_interaction_surface(section_params, confinement_params, 
                                              n_slices=20, n_points=60)
    
    # 2. PERFORM SAFETY CHECK
    #    To be accurate, we generate the exact 2D capacity curve at the Applied N level
    print(f"Checking safety at N = {applied_N/1000:.1f} kN...")
    
    # Check if N is within absolute vertical bounds first
    z_min, z_max = np.min(Z), np.max(Z)
    N_k = applied_N / 1000       # Convert to kN
    Mx_k = applied_Mx / 1e6      # Convert to kNm
    My_k = applied_My / 1e6      # Convert to kNm
    
    is_safe = False
    
    # Check vertical bounds (Axial Capacity)
    if N_k < z_min or N_k > z_max:
        is_safe = False # Fails in pure tension or pure compression
    else:
        # Generate the specific 2D slice for this axial load N
        Mx_cap, My_cap = generate_Mx_My_interaction(applied_N, section_params, 
                                                    confinement_params, verbose=False)
        
        # Create a geometric path from the capacity curve
        # Note: generate_Mx_My_interaction returns N.mm, convert to kNm
        poly_points = np.column_stack([np.array(My_cap)/1e6, np.array(Mx_cap)/1e6])
        interaction_path = Path(poly_points)
        
        # Check if the applied point (My, Mx) is inside the capacity polygon
        point_check = [My_k, Mx_k]
        is_safe = interaction_path.contains_point(point_check)

    # 3. Setup Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 4. Plot Surface
    #    alpha=0.2 makes it transparent to see the point inside
    surf = ax.plot_surface(Y, X, Z, cmap=cm.viridis, alpha=0.3, 
                           edgecolor='gray', linewidth=0.2, rstride=1, cstride=1)
    
    # 5. Plot the Applied Point
    #    Green if Safe, Red if Unsafe
    pt_color = 'green' if is_safe else 'red'
    status_str = "SAFE" if is_safe else "UNSAFE"
    
    ax.scatter(My_k, Mx_k, N_k, color=pt_color, s=200, 
               label=f'Applied Load ({status_str})', zorder=100, edgecolors='black')
    
    # 6. FIX ASPECT RATIO (Crucial for slender sections)
    #    We must ensure that 1 kNm on X-axis has the same visual length 
    #    as 1 kNm on Y-axis.
    
    # Calculate the data range for Moments (Mx and My)
    # X data is Mx, Y data is My
    range_Mx = np.ptp(X) # Peak-to-peak (max - min)
    range_My = np.ptp(Y)
    
    # Find the maximum range to define a bounding box
    max_moment_range = max(range_Mx, range_My)
    
    # Calculate centers
    mid_Mx = np.mean([np.min(X), np.max(X)])
    mid_My = np.mean([np.min(Y), np.max(Y)])
    
    # Set limits to be equal around the center
    # This forces the plot to be "square" in the plan view, revealing slenderness
    ax.set_ylim(mid_Mx - max_moment_range/2, mid_Mx + max_moment_range/2) # Mx is mapped to plot Y usually
    ax.set_xlim(mid_My - max_moment_range/2, mid_My + max_moment_range/2) # My is mapped to plot X usually
    
    # 7. Labels and Titles
    ax.set_xlabel('My [kN·m]', fontsize=11, labelpad=10)
    ax.set_ylabel('Mx [kN·m]', fontsize=11, labelpad=10)
    ax.set_zlabel('Axial Force P [kN]', fontsize=11, labelpad=10)
    
    # Remove pane fills for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    # Title
    conf_str = "Confined" if confinement_params else "Unconfined"
    ax.set_title(f"3D Interaction Diagram [{conf_str}]\n"
                 f"Load: P={N_k:.0f}kN, Mx={Mx_k:.1f}, My={My_k:.1f}\n"
                 f"Status: {status_str}", 
                 fontsize=14, fontweight='bold', color=pt_color)
    
    ax.legend()
    
    # Set initial camera view
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show()
#%%
# ============================================================================
# MAIN PRPGRAM (INPUT AND ANALYSIS)
# ============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # USER INPUT
    # =========================================================================
    
    # Section geometry [mm]
    Lx, Ly = 400, 600
    
    # Materials
    fc, fy = 30.0, 420.0
    Ec = 4700 * np.sqrt(fc)
    Es = 200000.0
    eps_c0 = 0.0035
    
    # Longitudinal reinforcement
    cover_clear = 25.0
    db_stirrup = 10.0
    db_long = 30.0
    cover = cover_clear + db_stirrup + db_long / 2
    
    rein_x = [cover, Lx-cover, cover, Lx-cover]
    rein_y = [cover, cover, Ly-cover, Ly-cover]
    rein_ds = [db_long] * 4
    
    # Eccentricities [mm]
    e0x, e0y = 300,100
    
    # =========================================================================
    # USER CHOICE: INCLUDE CONFINEMENT?
    # =========================================================================
    
    INCLUDE_CONFINEMENT = True   # <-- SET TO False TO DISABLE CONFINEMENT
    
    # Shear reinforcement (only used if INCLUDE_CONFINEMENT = True)
    db_v = 10.0                  # Stirrup diameter [mm]
    s = 100.0                    # Stirrup spacing [mm]
    legs_x, legs_y = 2, 2        # Number of legs
    fy_v = 420.0                 # Stirrup yield strength [MPa]
    
    # Analysis strain for section heatmap (default 0.0035)
    EPS_PLOT = 0.0035           # <-- Strain for section plot
    
    # =========================================================================
    # Create parameters
    # =========================================================================
    
    section_params = {
        'Lx': Lx, 'Ly': Ly,
        'rein_x': rein_x, 'rein_y': rein_y, 'rein_ds': rein_ds,
        'fc': fc, 'Ec': Ec, 'fy': fy, 'Es': Es, 'eps_c0': eps_c0
    }
    
    # Confinement (only if enabled)
    if INCLUDE_CONFINEMENT:
        confinement_params = calculate_mander_confinement(
            fc=fc, Lx=Lx, Ly=Ly, cover=cover_clear, db_v=db_v, s=s,
            legs_x=legs_x, legs_y=legs_y, fy_v=fy_v,
            rein_x=rein_x, rein_y=rein_y, rein_ds=rein_ds
        )
    else:
        confinement_params = None
    
    # =========================================================================
    # Header
    # =========================================================================
    
    print("\n" + "="*80)
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║      BIAXIAL RC COLUMN ANALYSIS - Optimized v5.0                             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("="*80)
    print(f"  Section:       {Lx} × {Ly} mm")
    print(f"  Concrete:      fc = {fc} MPa")
    print(f"  Reinforcement: 4 × Ø{db_long} mm")
    print(f"  Eccentricity:  ex = {e0x} mm, ey = {e0y} mm")
    print(f"  Confinement:   {'ENABLED' if INCLUDE_CONFINEMENT else 'DISABLED'}")
    
    if INCLUDE_CONFINEMENT and confinement_params:
        print("-"*80)
        print(f"  MANDER MODEL:")
        print(f"    Core:    {confinement_params['bc_x']:.1f} × {confinement_params['bc_y']:.1f} mm")
        print(f"    Ke:      {confinement_params['Ke']:.4f}")
        print(f"    fcc:     {confinement_params['fcc']:.2f} MPa (×{confinement_params['strength_ratio']:.3f})")
        print(f"    εcc:     {confinement_params['eps_cc']*1000:.3f}‰")
        print(f"    εcu:     {confinement_params['eps_cu']*1000:.2f}‰")
    
    print("="*80)
    
    # =========================================================================
    # Generate N-ε Curve
    # =========================================================================
    
    eps_end = confinement_params['eps_cu'] if confinement_params else 0.0035
    
    eps_list, N_list, Mx_list, My_list, info = generate_N_eps_curve(
        e0x, e0y, section_params, confinement_params,
        eps_start=0.00001, eps_end=eps_end, n_points=25, verbose=True
    )
    # Plot N-M curves
    if eps_list:
        # Pass 'info' dictionary to visualize cracking point
        plot_N_M_curves(eps_list, N_list, Mx_list, My_list, section_params,
                        confinement_params, info=info, filename="N_M_curves.png", show=False)
    # =========================================================================
    # Section Heatmap at specified strain
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"SECTION HEATMAP at ε = {EPS_PLOT*1000:.1f}‰")
    print("="*80)
    
    plot_section_at_strain(EPS_PLOT, e0x, e0y, section_params, confinement_params,
                           filename="section_heatmap.png", show=True)
    
    print("\n✓ Analysis complete!")

# %%
# =========================================================================
# Mx-My Interaction Diagram Analysis
# =========================================================================
    
    print(f"\n{'='*80}")
    print(f"GENERATING Mx-My INTERACTION DIAGRAM")
    print("="*80)
    
    applied_N =  4000 * 1e3   #  kN
    applied_Mx = 150.0 * 1e6  #  kNm
    applied_My = 100.0 * 1e6  #  kNm
    
    plot_Mx_My_interaction(
        Np=applied_N, 
        Mxp=applied_Mx, 
        Myp=applied_My,
        section_params=section_params,
        confinement_params=confinement_params,
        eps_max=0.0035, # ACI limit
        n_points=72,
        filename="Mx_My_interaction.png",
        show=True
    )
#%%
plot_3d_interaction(section_params, confinement_params, 
                                 applied_N=applied_N, 
                                 applied_Mx=applied_Mx, 
                                 applied_My=applied_My)
# %%
