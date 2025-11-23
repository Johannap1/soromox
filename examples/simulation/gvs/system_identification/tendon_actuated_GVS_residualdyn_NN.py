import jax
import pandas as pd
import jax.numpy as jnp
from soromox.systems.gvs.attributes import LinkAttributes, JointAttributes, BasisAttributes
#from soromox.systems.gvs.core import GVS
from soromox.systems.gvs.tendon_actuated_gvs import TendonActuatedGVS
from functools import partial
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import optimistix as optx
import equinox as eqx
import optax
from jax import Array, lax
import numpy as onp
from dataclasses import replace
from jax.tree_util import tree_leaves
from pathlib import Path
jax.config.update("jax_enable_x64", True)

# jax.config.update("jax_platform_name", "gpu")  # or "cpu"

print("JAX default backend:", jax.default_backend())
print("JAX devices:", jax.devices())

### MOCAP DATA TRANSFORMATION FUNCTIONS ###

def _transform_points(pb: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray,
                      Rot: jnp.ndarray, R_alg1: jnp.ndarray, R_alg2: jnp.ndarray, offset: jnp.ndarray):
    # equivalente del blocco richiesto
    pb_gl = Rot @ pb
    p1_gl = Rot @ p1
    p2_gl = Rot @ p2
    p3_gl = Rot @ p3
    p4_gl = Rot @ p4

    pb_lcl = -offset
    p1_lcl = p1_gl - (pb_gl + offset)
    p2_lcl = p2_gl - (pb_gl + offset)
    p3_lcl = p3_gl - (pb_gl + offset)
    p4_lcl = p4_gl - (pb_gl + offset)

    R_total = R_alg1 @ R_alg2
    pb_tr = R_total @ pb_lcl
    p1_tr = R_total @ p1_lcl
    p2_tr = R_total @ p2_lcl
    p3_tr = R_total @ p3_lcl
    p4_tr = R_total @ p4_lcl

    return {"pb_tr": pb_tr, "p1_tr": p1_tr, "p2_tr": p2_tr, "p3_tr": p3_tr, "p4_tr": p4_tr}

### EXCEL DATA READING ###
# Rot = jnp.array([[1.0, 0.0, 0.0],
#                  [0.0, 0.0, -1.0],
#                  [0.0, 1.0, 0.0]], dtype=jnp.float64)
# R_alg1 = jnp.array([[-1.0, 0.0, 0.0],
                    # [0.0, -1.0, 0.0],
                    # [0.0, 0.0, 1.0]], dtype=jnp.float64)
# R_alg2 = jnp.array([[1.0, 0.0, 0.0],
                    # [0.0, -1.0, 0.0],
                    # [0.0, 0.0, -1.0]], dtype=jnp.float64)
Rot = jnp.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]], dtype=jnp.float64)
R_alg1 = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]], dtype=jnp.float64)
R_alg2 = jnp.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=jnp.float64)
offset = jnp.array([-0.03, -0.013, 0.0], dtype=jnp.float64)

# df = pd.read_csv("data/ss_NL2.csv", header=None)
# vals = df.iloc[:, 2:].to_numpy(dtype=float)     # (N, M-2)
# filtered = jnp.asarray(vals.mean(axis=0))       # (M-2,)
# # mapping 1-based MATLAB -> 0-based Python (end esclusivo)
# pb = filtered[0:3]
# p1 = filtered[6:9]
# p2 = filtered[3:6]
# p3 = filtered[12:15]
# p4 = filtered[9:12]

# res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
# pb_NL = res["pb_tr"]
# p1_NL = res["p1_tr"]
# p2_NL = res["p2_tr"]
# p3_NL = res["p3_tr"]
# p4_NL = res["p4_tr"]

# print("pb_NL:", pb_tr)
# print("p1_NL:", p1_tr)
# print("p2_NL:", p2_tr)
# print("p3_NL:", p3_tr)
# print("p4_NL:", p4_tr)


df = pd.read_csv("data/m1_u01.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[12:15]
p1 = filtered[0:3]
p2 = filtered[9:12]
p3 = filtered[3:6]
p4 = filtered[6:9]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u01 = res["pb_tr"]
p1_m1u01 = res["p1_tr"]
p2_m1u01 = res["p2_tr"]
p3_m1u01 = res["p3_tr"]
p4_m1u01 = res["p4_tr"]

# print("p4_m1u01:", p4_m1u01)
# print("p3_m1u01:", p3_m1u01)
# print("p2_m1u01:", p2_m1u01)
# print("p1_m1u01:", p1_m1u01)
# print("pb_m1u01:", pb_m1u01)

df = pd.read_csv("data/m1_u015.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u015 = res["pb_tr"]
p1_m1u015 = res["p1_tr"]
p2_m1u015 = res["p2_tr"]
p3_m1u015 = res["p3_tr"]
p4_m1u015 = res["p4_tr"]

df = pd.read_csv("data/m1_u02.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u02 = res["pb_tr"]
p1_m1u02 = res["p1_tr"]
p2_m1u02 = res["p2_tr"]
p3_m1u02 = res["p3_tr"]
p4_m1u02 = res["p4_tr"]

df = pd.read_csv("data/m1_u025.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[12:15]
p1 = filtered[9:12]
p2 = filtered[0:3]
p3 = filtered[6:9]
p4 = filtered[3:6]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1u025 = res["pb_tr"]
p1_m1u025 = res["p1_tr"]
p2_m1u025 = res["p2_tr"]
p3_m1u025 = res["p3_tr"]
p4_m1u025 = res["p4_tr"]

df = pd.read_csv("data/m2_u01.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[6:9]
p1 = filtered[0:3]
p2 = filtered[12:15]
p3 = filtered[9:12]
p4 = filtered[3:6]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u01 = res["pb_tr"]
p1_m2u01 = res["p1_tr"]
p2_m2u01 = res["p2_tr"]
p3_m2u01 = res["p3_tr"]
p4_m2u01 = res["p4_tr"]

df = pd.read_csv("data/m2_u015.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u015 = res["pb_tr"]
p1_m2u015 = res["p1_tr"]
p2_m2u015 = res["p2_tr"]
p3_m2u015 = res["p3_tr"]
p4_m2u015 = res["p4_tr"]

df = pd.read_csv("data/m2_u02.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u02 = res["pb_tr"]
p1_m2u02 = res["p1_tr"]
p2_m2u02 = res["p2_tr"]
p3_m2u02 = res["p3_tr"]
p4_m2u02 = res["p4_tr"]

df = pd.read_csv("data/m2_u025.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)     
filtered = jnp.asarray(vals.mean(axis=0))    
pb = filtered[6:9]
p1 = filtered[3:6]
p2 = filtered[9:12]
p3 = filtered[0:3]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2u025 = res["pb_tr"]
p1_m2u025 = res["p1_tr"]
p2_m2u025 = res["p2_tr"]
p3_m2u025 = res["p3_tr"]
p4_m2u025 = res["p4_tr"]

# --- m1_005 ---
df = pd.read_csv("data/m1_u005.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[3:6]
p2 = filtered[6:9]
p3 = filtered[12:15]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1_005, p1_m1_005, p2_m1_005, p3_m1_005, p4_m1_005 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m1_012 ---
df = pd.read_csv("data/m1_u012.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1_012, p1_m1_012, p2_m1_012, p3_m1_012, p4_m1_012 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m1_018 ---
df = pd.read_csv("data/m1_u018.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[6:9]
p1 = filtered[3:6]
p2 = filtered[9:12]
p3 = filtered[0:3]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m1_018, p1_m1_018, p2_m1_018, p3_m1_018, p4_m1_018 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m2_005 ---
df = pd.read_csv("data/m2_u005.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[6:9]
p1 = filtered[3:6]
p2 = filtered[9:12]
p3 = filtered[0:3]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2_005, p1_m2_005, p2_m2_005, p3_m2_005, p4_m2_005 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m2_012 ---
df = pd.read_csv("data/m2_u012.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[6:9]
p1 = filtered[3:6]
p2 = filtered[9:12]
p3 = filtered[12:15]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2_012, p1_m2_012, p2_m2_012, p3_m2_012, p4_m2_012 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m2_018 ---
df = pd.read_csv("data/m2_u018.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[3:6]
p1 = filtered[0:3]
p2 = filtered[6:9]
p3 = filtered[9:12]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m2_018, p1_m2_018, p2_m2_018, p3_m2_018, p4_m2_018 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_001 ---
df = pd.read_csv("data/m12_u001.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[6:9]
p1 = filtered[12:15]
p2 = filtered[9:12]
p3 = filtered[0:3]
p4 = filtered[3:6]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_01, p1_m12_01, p2_m12_01, p3_m12_01, p4_m12_01 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_002 ---
df = pd.read_csv("data/m12_u002.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_02, p1_m12_02, p2_m12_02, p3_m12_02, p4_m12_02 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_001002 ---
df = pd.read_csv("data/m12_u001002.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[3:6]
p3 = filtered[0:3]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_0102, p1_m12_0102, p2_m12_0102, p3_m12_0102, p4_m12_0102 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_0015002 ---
df = pd.read_csv("data/m12_u0015002.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[12:15]
p1 = filtered[3:6]
p2 = filtered[9:12]
p3 = filtered[6:9]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_01502, p1_m12_01502, p2_m12_01502, p3_m12_01502, p4_m12_01502 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_002001 ---
df = pd.read_csv("data/m12_u002001.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[12:15]
p1 = filtered[6:9]
p2 = filtered[9:12]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_0201, p1_m12_0201, p2_m12_0201, p3_m12_0201, p4_m12_0201 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_0020015 ---
df = pd.read_csv("data/m12_u0020015.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[3:6]
p2 = filtered[6:9]
p3 = filtered[0:3]
p4 = filtered[12:15]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_02015, p1_m12_02015, p2_m12_02015, p3_m12_02015, p4_m12_02015 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_00080015 ---
df = pd.read_csv("data/m12_u00080015.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[12:15]
p1 = filtered[6:9]
p2 = filtered[9:12]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_008015, p1_m12_008015, p2_m12_008015, p3_m12_008015, p4_m12_008015 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_00150008 ---
df = pd.read_csv("data/m12_u00150008.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_015008, p1_m12_015008, p2_m12_015008, p3_m12_015008, p4_m12_015008 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_00100005 ---
df = pd.read_csv("data/m12_u00100005.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_01005, p1_m12_01005, p2_m12_01005, p3_m12_01005, p4_m12_01005 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]

# --- m12_00050010 ---
df = pd.read_csv("data/m12_u00050010.csv", header=None)
vals = df.iloc[:, 2:].to_numpy(dtype=float)
filtered = jnp.asarray(vals.mean(axis=0))
pb = filtered[9:12]
p1 = filtered[6:9]
p2 = filtered[12:15]
p3 = filtered[3:6]
p4 = filtered[0:3]
res = _transform_points(pb, p1, p2, p3, p4, Rot, R_alg1, R_alg2, offset)
pb_m12_00501, p1_m12_00501, p2_m12_00501, p3_m12_00501, p4_m12_00501 = res["pb_tr"], res["p1_tr"], res["p2_tr"], res["p3_tr"], res["p4_tr"]




def compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_batch):
    """
    Restituisce gli errori (M, 12) = differenze per ogni marker e ogni esperimento
    """
    u_T = u_batch.T
    meas_T = measured_markers_batch.T
    M = u_T.shape[0]
    if tau_batch.ndim == 1:
        tau_mat = jnp.tile(tau_batch.reshape(dof, 1), (1, M))      # (dof, M)
    elif tau_batch.shape == (dof, M):
        tau_mat = tau_batch                                        # (dof, M)
    elif tau_batch.shape == (M, dof):
        tau_mat = tau_batch.T                                      # (dof, M)
    else:
        raise ValueError(f"Unexpected tau_batch shape {tau_batch.shape}; expected (dof,M), (M,dof) o (dof,) with dof={dof}, M={M}")

    diffs = []
    for m in range(M):
        res = solve_equilibrium_with_tau(robot, u_T[m], tau_mat[:, m], q0)
        q_star = res.value
        pred = markers_from_q(robot, q_star)
        diffs.append(pred - meas_T[m])
    diffs = jnp.stack(diffs)  # (M, 12)
    return diffs


###drawing


def draw_robot_curve(
    robot: TendonActuatedGVS,
    q: Array,
    num_points: int = 50,
):
    batched_forward_kinematics = jax.vmap(robot.forward_kinematics, in_axes=(None, 0))
    L_max = jnp.sum(robot.V_L)

    s_ps = jnp.linspace(0, L_max, num_points)
    g_ps = batched_forward_kinematics(q, s_ps)[:, :3, 3]

    # q_gathered = robot._min_size_gathered(q)
    # V_g = robot._forward_kinematics_gauss(q_gathered)

    # g_ps = jnp.concatenate(V_g[:, :, :-1, -1:], axis=0)

    curve = onp.array(g_ps, dtype=onp.float64)
    return curve  # (N, 3)


### SOFT ROBOT UTILITIES FUNCTIONS ###

# ---- Marker prediction for a fixed q  ----
def markers_from_q(robot: TendonActuatedGVS, q: jnp.ndarray) -> jnp.ndarray:
    """
    Output [p1; p2; p3; p4] (12,), where:
      p1 @ s=0.1291 with offset [0,0, 0.025]
      p2 @ s=0.2195 with offset [0,0,-0.021]
      p3 @ s=0.2800 with offset [0,0, 0.020]
      p4 @ s=sum(L) with offset [0.008,0,0]
    """
    def tip_at_s(s_val, offset):
        G = robot.forward_kinematics(q, s_val)     # 4x4
        p = G[:3, 3]; R = G[:3, :3]
        return p + R @ offset

    p1 = tip_at_s(0.1291, jnp.array([0.0, 0.0,  0.025]))
    p2 = tip_at_s(0.2195, jnp.array([0.0, 0.0, -0.021]))
    p3 = tip_at_s(0.2800, jnp.array([0.0, 0.0,  0.020]))
    p4 = tip_at_s(jnp.sum(robot.V_L), jnp.array([0.008, 0.0, 0.0]))
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)

#alternatively, u can u see data from a continuous movement and use dynamics(more useful for the whole simulation I think)
def solve_equilibrium_with_tau(robot: TendonActuatedGVS,
                               u: jnp.ndarray,          # (na,)
                               tau_ext: jnp.ndarray,    # (dof,)
                               q0: jnp.ndarray):        # (dof,)
    """
    Solve static equilibrium with an additive generalized external force tau_ext:
      K(q) q + G(q) - B(q) u - tau_ext = 0
    Returns: res (optimistix result with .value = q*)
    """
    def statics_eq(q, args):
        u_m, tau_ext = args
        K = robot.stiffness_matrix()       # (dof,dof)
        B = robot.actuation_matrix(q)      # (dof, na)
        G = robot.gravitational_force(q)   # (dof,)
        return K @ q + G - B @ u_m - tau_ext

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    statics_eq_jit = jax.jit(statics_eq)   # robot in closure (statico)
    return optx.root_find(statics_eq_jit, solver, q0, (u, tau_ext), max_steps=200)


def make_tau_loss_for_sample(robot: TendonActuatedGVS,
                             u_i: jnp.ndarray,           # (na,)
                             p_meas_i: jnp.ndarray,      # (12,)
                             q0: jnp.ndarray,
                             lambda_reg: float):
 
    # def loss_tau(raw_tau):
    def loss_tau(tau):
        
        tau_min = jnp.array([-1,-1,-1,-1,-1,-1,-1,-0.01,-0.01])
        tau_max = jnp.array([1, 1, 1, 1, 1, 1, 1, 0.01, 0.01])

        # tau = 0.5 * (tau_max + tau_min) + 0.5 * (tau_max - tau_min) * jnp.tanh(raw_tau)  # (dof,)

        res = solve_equilibrium_with_tau(robot, u_i, tau, q0)
        q_star = res.value
        pred = markers_from_q(robot, q_star)           # (12,)
        err = pred - p_meas_i
        
        # q_min = jnp.array([-10,-10,-10,-10,-10,-10,-10,-5,-5])
        # q_max = jnp.array([ 10, 10, 10, 10, 10, 10, 10, 5, 5])


        # penalty = jnp.sum(jnp.square(jnp.maximum(0.0, q_star - q_max))) \
        #         + jnp.sum(jnp.square(jnp.maximum(0.0, q_min - q_star)))
     
        penalty = jnp.sum(jnp.square(jnp.maximum(0.0, tau - tau_max))) \
                + jnp.sum(jnp.square(jnp.maximum(0.0, tau_min - tau)))


        return 0.5 * (err @ err) \
               + 0.5 * lambda_reg * (tau @ tau) \
               + 1e1 * penalty 


    return loss_tau


def solve_tau_star_for_batch(robot: TendonActuatedGVS,
                             u_batch: jnp.ndarray,                 # (na, M)
                             measured_markers_batch: jnp.ndarray,  # (12, M)
                             q0: jnp.ndarray,
                             lambda_reg: float = 1e-6,
                             steps: int = 50,
                             lr: float = 1e-2):

    M = u_batch.shape[1]
    dof = q0.shape[0]
    tau_stars = []
    tau0 = jnp.zeros((dof,), dtype=jnp.float64)  # cold-start
   
    # tau_min = jnp.array([-1,-1,-1,-1,-1,-1,-1,-0.01,-0.01])
    # tau_max = jnp.array([1, 1, 1, 1, 1, 1, 1, 0.01, 0.01])
    # tau_init = jnp.zeros((dof,), dtype=jnp.float64)
        
    # mu_tau = 0.5 * (tau_max + tau_min)
    # r_tau  = 0.5 * (tau_max - tau_min)
    # x_tau  = (tau_init - mu_tau) / r_tau
    # def _atanh_clipped(x, eps=1e-6):
    #     return jnp.arctanh(jnp.clip(x, -1 + eps, 1 - eps))
    # raw_tau0 = _atanh_clipped(x_tau)
   
    for m in range(M):
        u_i = u_batch[:, m]
        p_meas_i = measured_markers_batch[:, m]
        loss_tau = make_tau_loss_for_sample(robot, u_i, p_meas_i, q0, lambda_reg)
        val_and_grad = eqx.filter_value_and_grad(loss_tau)

        opt = optax.adam(lr)
        opt_state = opt.init(tau0)
        # opt_state = opt.init(raw_tau0)

        best_tau = tau0
        # best_tau_raw = raw_tau0
        best_val = jnp.inf

        tau = tau0
        # tau_raw = raw_tau0

        for step in range(steps):
            val, g = val_and_grad(tau)
            # val, g = val_and_grad(tau_raw)
            
            print("Current step:", step)
            print(f"[Sample {m:02d}] Current tau loss={float(val):.6e}")

            # tau = 0.5 * (tau_max + tau_min) + 0.5 * (tau_max - tau_min) * jnp.tanh(tau_raw)  # (dof,)
 
            print(f"[Sample {m:02d}] Current tau={tau}")
            
            if val < best_val:
                best_val = val
                best_tau = tau
                # best_tau_raw = tau_raw

            updates, opt_state = opt.update(g, opt_state, tau)
            tau = optax.apply_updates(tau, updates)
            # updates, opt_state = opt.update(g, opt_state, tau_raw)
            # tau_raw = optax.apply_updates(tau_raw, updates)
            
        # best_tau = 0.5 * (tau_max + tau_min) + 0.5 * (tau_max - tau_min) * jnp.tanh(best_tau_raw)  # (dof,)
        tau_stars.append(best_tau)

        print(f"[Sample {m:02d}] Identified tau* with loss={float(best_val):.6e}")

    tau_star_batch = jnp.stack(tau_stars, axis=1)  # (dof, M)
    return tau_star_batch


# ========================= NN to fit tau*(u) =========================

class TauNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 2, *, key: Array):
        self.mlp = eqx.nn.MLP(in_dim, out_dim, width, depth, key=key)

    def __call__(self, x):
        return self.mlp(x)

def train_tau_nn(u_batch: jnp.ndarray,                 # (na, M)
                 tau_star_batch: jnp.ndarray,          # (dof, M)
                 gamma_wd: float = 1e-4,
                 steps: int = 2000,
                 lr: float = 1e-3,
                 width: int = 64,
                 depth: int = 2,
                 key: Array | None = None):
    """
    Allena NN: x = [t; u] -> tau_ext (dof,)
    Loss: mean 0.5||NN(x_i)-tau*_i||^2 + 0.5*gamma*||theta||^2
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    M = u_batch.shape[1]
    na = u_batch.shape[0]
    dof = tau_star_batch.shape[0]

    # Build features X: (M, na)
    X = u_batch.T                             # (M, na)
    Y = tau_star_batch.T                      # (M, dof)

    model = TauNN(in_dim=na, out_dim=dof, width=width, depth=depth, key=key)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_model(m: TauNN):
        pred = jax.vmap(lambda x: m(x))(X)               # (M, dof)
        diff = pred - Y
        data_term = 0.5 * jnp.mean(jnp.sum(diff * diff, axis=1))
        # weight decay su tutti i tensori trainabili
        # l2 = sum(jnp.sum(p**2) for p in jax.tree_leaves(eqx.filter(m, eqx.is_inexact_array)))
        # l2 = sum(jnp.sum(p**2) for p in tree_leaves(eqx.filter(m, eqx.is_inexact_array)))
        l2 = sum(jnp.sum(p**2) for p in jax.tree.leaves(eqx.filter(m, eqx.is_inexact_array)))
        return data_term + 0.5 * gamma_wd * l2
        
    loss_and_grad = eqx.filter_value_and_grad(loss_model)

    for step in range(steps):
        loss_val, grads = loss_and_grad(model)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        # if step % 200 == 0:
        # print(f"[NN step {step:04d}] loss={float(loss_val):.6e}")

    return model




### BODY DEFINITION OF THE SOFT ROBOT ###

#2 link version -> DA ADATTARE CON IL VALORE DELLO STATICS ID!!!!!!
# Link 1
link1 = LinkAttributes(
    section="Circular",
    E=3.2e5,
    # E=3.04e5,
    nu=0.45,
    rho=1270.0,
    # rho=1310.0,
    eta=1e4,
    L=0.0250+0.2550+0.0250,
    r_i=0.01541,
    r_f=0.00642,
)

# Link 2
link2 = LinkAttributes(
    section="Circular",
    E=3.2e5,
    # E=3.04e5,
    nu=0.45,
    rho=1270.0,
    # rho=1310.0,
    eta=1e4,
    L=0.0550,
    r_i=0.00642,
    r_f=0.00480,
)
joint1 = JointAttributes(jointtype="Fixed")
joint2 = JointAttributes(jointtype="Fixed")
basis1 = BasisAttributes(basistype="Monomial", Bdof=[1, 1, 1, 1, 0, 0], Bodr=[0, 1, 1, 1, 0, 0])
basis2 = BasisAttributes(basistype="Monomial", Bdof=[0, 1, 1, 0, 0, 0], Bodr=[0, 0, 0, 0, 0, 0])

n_gauss_list = [8, 8]
gravity_vector = [0.0, 0.0, -9.81]

tendon_routing_params = {
     "ry": jnp.array([0.0114*jnp.cos(jnp.pi/180*30), 0.0114*jnp.cos(jnp.pi/180*150)]),
     "my": jnp.array([-0.0295*jnp.cos(jnp.pi/180*30),-0.0295*jnp.cos(jnp.pi/180*150)]),
     "rz": jnp.array([0.0114*jnp.sin(jnp.pi/180*30), 0.0114*jnp.sin(jnp.pi/180*150)]),
     "mz": jnp.array([-0.0295*jnp.sin(jnp.pi/180*30), -0.0295*jnp.sin(jnp.pi/180*150)]),
     "idx_seg_att": jnp.array([0, 0]),
}

robot = TendonActuatedGVS(
    links_list=[link1, link2],
    joints_list=[joint1, joint2],
    basis_list=[basis1, basis2],
    n_gauss_list=n_gauss_list,
    gravity_vector=gravity_vector,
    tendon_routing_params=tendon_routing_params,
)

# Initialize parameters
n_links = int(robot.num_segments)


# Prepare batch data (measured coordinates and inputs)
def _pack_markers(p1, p2, p3, p4):
    return jnp.concatenate([p1, p2, p3, p4], axis=0)  # (12,)

# measured_list = [
#     _pack_markers(p1_m1u01,  p2_m1u01,  p3_m1u01,  p4_m1u01),
#     _pack_markers(p1_m1u015, p2_m1u015, p3_m1u015, p4_m1u015),
#     _pack_markers(p1_m1u02,  p2_m1u02,  p3_m1u02,  p4_m1u02),
#     _pack_markers(p1_m1u025, p2_m1u025, p3_m1u025, p4_m1u025),
#     _pack_markers(p1_m2u01,  p2_m2u01,  p3_m2u01,  p4_m2u01),
#     _pack_markers(p1_m2u015, p2_m2u015, p3_m2u015, p4_m2u015),
#     _pack_markers(p1_m2u02,  p2_m2u02,  p3_m2u02,  p4_m2u02),
#     _pack_markers(p1_m2u025, p2_m2u025, p3_m2u025, p4_m2u025),
# ]

measured_list = [
    _pack_markers(p1_m1u01,  p2_m1u01,  p3_m1u01,  p4_m1u01),
    _pack_markers(p1_m1u015, p2_m1u015, p3_m1u015, p4_m1u015),
    _pack_markers(p1_m1u02,  p2_m1u02,  p3_m1u02,  p4_m1u02),
    _pack_markers(p1_m1u025, p2_m1u025, p3_m1u025, p4_m1u025),
    _pack_markers(p1_m2u01,  p2_m2u01,  p3_m2u01,  p4_m2u01),
    _pack_markers(p1_m2u015, p2_m2u015, p3_m2u015, p4_m2u015),
    _pack_markers(p1_m2u02,  p2_m2u02,  p3_m2u02,  p4_m2u02),
    _pack_markers(p1_m2u025, p2_m2u025, p3_m2u025, p4_m2u025),

    _pack_markers(p1_m1_005, p2_m1_005, p3_m1_005, p4_m1_005),
    _pack_markers(p1_m1_012, p2_m1_012, p3_m1_012, p4_m1_012),
    _pack_markers(p1_m1_018, p2_m1_018, p3_m1_018, p4_m1_018),
    _pack_markers(p1_m2_005, p2_m2_005, p3_m2_005, p4_m2_005),
    _pack_markers(p1_m2_012, p2_m2_012, p3_m2_012, p4_m2_012),
    _pack_markers(p1_m2_018, p2_m2_018, p3_m2_018, p4_m2_018),
    _pack_markers(p1_m12_01,    p2_m12_01,    p3_m12_01,    p4_m12_01),
    _pack_markers(p1_m12_02,    p2_m12_02,    p3_m12_02,    p4_m12_02),
    _pack_markers(p1_m12_0102,  p2_m12_0102,  p3_m12_0102,  p4_m12_0102),
    _pack_markers(p1_m12_01502, p2_m12_01502, p3_m12_01502, p4_m12_01502),
    _pack_markers(p1_m12_0201,  p2_m12_0201,  p3_m12_0201,  p4_m12_0201),
    _pack_markers(p1_m12_02015, p2_m12_02015, p3_m12_02015, p4_m12_02015),
    _pack_markers(p1_m12_008015, p2_m12_008015, p3_m12_008015, p4_m12_008015),
    _pack_markers(p1_m12_015008, p2_m12_015008, p3_m12_015008, p4_m12_015008),
    _pack_markers(p1_m12_01005,  p2_m12_01005,  p3_m12_01005,  p4_m12_01005),
    _pack_markers(p1_m12_00501,  p2_m12_00501,  p3_m12_00501,  p4_m12_00501),
]
measured_markers_batch = jnp.stack(measured_list, axis=1) 
radius = 0.0325

def _neg(val):
    return -float(val) / float(radius)

# Define mapping from sample name to u
sample_to_u = {
    "m1u01":      jnp.array([_neg(0.1), 0.0]),
    "m1u015":     jnp.array([_neg(0.15), 0.0]),
    "m1u02":      jnp.array([_neg(0.2), 0.0]),
    "m1u025":     jnp.array([_neg(0.25), 0.0]),
    "m2u01":      jnp.array([0.0, _neg(0.1)]),
    "m2u015":     jnp.array([0.0, _neg(0.15)]),
    "m2u02":      jnp.array([0.0, _neg(0.2)]),
    "m2u025":     jnp.array([0.0, _neg(0.25)]),

    "m1_005":      jnp.array([_neg(0.05), 0.0]),
    "m1_012":      jnp.array([_neg(0.12), 0.0]),
    "m1_018":      jnp.array([_neg(0.18), 0.0]),
    "m2_005":      jnp.array([0.0, _neg(0.05)]),
    "m2_012":      jnp.array([0.0, _neg(0.12)]),
    "m2_018":      jnp.array([0.0, _neg(0.18)]),

    "m12_001":         jnp.array([_neg(0.1), _neg(0.1)]),
    "m12_002":         jnp.array([_neg(0.2), _neg(0.2)]),
    "m12_001002":      jnp.array([_neg(0.1), _neg(0.2)]),
    "m12_0015002":     jnp.array([_neg(0.15), _neg(0.2)]),
    "m12_002001":      jnp.array([_neg(0.2), _neg(0.1)]),
    "m12_0020015":     jnp.array([_neg(0.2), _neg(0.15)]),
    "m12_00080015":    jnp.array([_neg(0.08), _neg(0.15)]),
    "m12_00150008":    jnp.array([_neg(0.15), _neg(0.08)]),
    "m12_00100005":    jnp.array([_neg(0.1), _neg(0.05)]),
    "m12_00050010":    jnp.array([_neg(0.05), _neg(0.1)]),
}


u_list = [
    sample_to_u["m1u01"],
    sample_to_u["m1u015"],
    sample_to_u["m1u02"],
    sample_to_u["m1u025"],
    sample_to_u["m2u01"],
    sample_to_u["m2u015"],
    sample_to_u["m2u02"],
    sample_to_u["m2u025"],
    sample_to_u["m1_005"],
    sample_to_u["m1_012"],
    sample_to_u["m1_018"],
    sample_to_u["m2_005"],
    sample_to_u["m2_012"],
    sample_to_u["m2_018"],
    sample_to_u["m12_001"],
    sample_to_u["m12_002"],
    sample_to_u["m12_001002"],
    sample_to_u["m12_0015002"],
    sample_to_u["m12_002001"],
    sample_to_u["m12_0020015"],
    sample_to_u["m12_00080015"],
    sample_to_u["m12_00150008"],
    sample_to_u["m12_00100005"],
    sample_to_u["m12_00050010"],
]
u_batch = jnp.stack(u_list, axis=1) 
print("u_batch:", u_batch)
print("measured_markers_batch:", measured_markers_batch)

M = measured_markers_batch.shape[1]
na = u_batch.shape[0]
dof = int(sum(robot.V_dof.reshape(-1)))
assert u_batch.shape == (na, M)
assert measured_markers_batch.shape == (12, M)

q0 = jnp.zeros((dof,), dtype=jnp.float64)


# 1) Identify tau*_i for each sample
lambda_reg = 1e-8   # regularization on tau
# print("Starting identification of tau* for each sample ===")


tau_star_batch = solve_tau_star_for_batch(
    robot, u_batch, measured_markers_batch, q0,
    lambda_reg=lambda_reg, steps=300, lr=1e-2
)  # (dof, M)


# Save
onp.save("tau_star_batch.npy", onp.asarray(tau_star_batch))
onp.savetxt("tau_star_batch.txt", onp.asarray(tau_star_batch))

# Load (later)
# tau_star_batch_m1std = jnp.array(onp.load("tau_star_batch_m1std.npy"))
# tau_star_batch_allrest = jnp.array(onp.load("tau_star_batch_allrest.npy"))
# tau_star_batch = jnp.concatenate([tau_star_batch_m1std, tau_star_batch_allrest], axis=1)

print("Identified tau* shape:", tau_star_batch.shape)
print("Identified tau* samples:", tau_star_batch)

# # Save in a text file
# onp.savetxt("tau_star_batch.txt", onp.asarray(tau_star_batch))

# # Load (shape must be known afterward)
# data = onp.loadtxt("tau_star_batch.txt")
# tau_star_loaded = jnp.array(data)
# tau_star_loaded = tau_star_loaded.reshape(tau_star_batch.shape) # if necessary


tau_zero_batch = jnp.zeros((dof, M), dtype=jnp.float64)
errors_before = compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_zero_batch)

def marker_rmse(errors):
    # errors shape: (M, 12)
    errors_reshaped = errors.reshape(errors.shape[0], 4, 3)  # (M, 4, 3)
    rms_per_marker = jnp.sqrt(jnp.mean(jnp.sum(errors_reshaped**2, axis=2), axis=0))
    return rms_per_marker  # (4,)

rmse_before = marker_rmse(errors_before)
print("RMSE per marker before tau* identification:", rmse_before)

errors_tau_star = compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_star_batch)
rmse_tau_star = marker_rmse(errors_tau_star)
print("RMSE per marker with tau* identification:", rmse_tau_star)


####all
# RMSE per marker before tau* identification: [0.01230611 0.02107147 0.02689435 0.05295949]
# RMSE per marker with tau* identification: [0.0118526  0.01157757 0.01469062 0.02352333]

# tau_pred = []


# # 2) Train NN: x=[u] -> tau*
# model = train_tau_nn(
#     u_batch, tau_star_batch,
#     gamma_wd=1e-4, steps=3000, lr=5e-4, width=64, depth=2, key=jax.random.PRNGKey(0)
# )

# #train on a subset of data
# train_fraction = 0.7  # for example, 70% train
# M_train = int(M * train_fraction)

# # shuffle indices (optional)
# key = jax.random.PRNGKey(0)
# perm = jax.random.permutation(key, M)

# # indices
# train_idx = perm[:M_train]
# test_idx = perm[M_train:]

# # split data
# u_train = u_batch[:, train_idx]
# tau_star_train = tau_star_batch[:, train_idx]

# u_test = u_batch[:, test_idx]
# tau_star_test = tau_star_batch[:, test_idx]
# model = train_tau_nn(
#     u_test, tau_star_test,
#     gamma_wd=1e-4, steps=3000, lr=1e-3, width=64, depth=2, key=jax.random.PRNGKey(0)
# )
# def evaluate_model(model, u_batch, tau_star_batch):
#     X = u_batch.T
#     Y = tau_star_batch.T
#     pred = jax.vmap(lambda x: model(x))(X)
#     mse = jnp.mean((pred - Y) ** 2)
#     return mse, pred

# train_mse, _ = evaluate_model(model, u_train, tau_star_train)
# test_mse, _  = evaluate_model(model, u_test, tau_star_test)
# full_mse, _  = evaluate_model(model, u_batch, tau_star_batch)

# print(f"Train MSE: {train_mse:.6e}")
# print(f"Test  MSE: {test_mse:.6e}")
# print(f"Full  MSE: {full_mse:.6e}")

# # onp.save("tau_nn_model.npy", model)
# eqx.tree_serialise_leaves("tau_nn_model_full.eqx", model)

# # Load model later
# # model = TauNN(in_dim=na, out_dim=dof, width=64, depth=2, key=jax.random.PRNGKey(0))
# # model = eqx.tree_deserialise_leaves("tau_nn_model.eqx", model)



# metrics = onp.array([train_mse, test_mse, full_mse])
# labels = ["train", "test", "full"]

# plt.figure(figsize=(4.5, 3.2))
# plt.bar(labels, metrics, color=["tab:blue", "tab:orange", "tab:green"], alpha=0.8)
# plt.ylabel("MSE")
# plt.title("Tau-NN MSE")
# plt.yscale("log")
# plt.grid(True, axis="y", linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.savefig("tau_nn_mse_bar.svg", bbox_inches="tight")
# plt.savefig("tau_nn_mse_bar.pdf", bbox_inches="tight")
# plt.savefig("tau_nn_mse_bar.jpg", dpi=300, bbox_inches="tight")
# plt.show()


# # # # 3) Esempio di uso NN su un campione
# # # x0 = u_batch[:, :1].T.reshape(-1) # (na,)
# # # tau_pred0 = tau_nn(x0)   # (dof,)
# # # print("NN tau_pred on first sample:", tau_pred0)

# # # # Valutazione su tutti i campioni: errore tra p_est(u, tau_nn(u)) e p_meas
# # # print("\n=== Evaluation over all samples ===")
# # # q_guess = q0
# # # total_err2 = 0.0

# for m in range(M):
#     u_i = u_batch[:, m]
#     tau_pred_i = model(u_i)  # (dof,)
#     tau_pred.append(tau_pred_i)

# tau_pred_batch = jnp.stack(tau_pred, axis=1)  # (dof, M)

# errors_tau_pred = compute_marker_errors(robot, u_batch, q0, measured_markers_batch, tau_pred_batch)
# rmse_tau_pred = marker_rmse(errors_tau_pred)
# print("RMSE per marker with tau* NN prediction:", rmse_tau_pred)


# # # # === Plots: tau_zero vs tau_star vs tau_pred ===
# marker_names = [f"marker {i+1}" for i in range(4)]
# x = onp.arange(len(marker_names))
# width = 0.25

# # Per-marker RMSE bars
# rmse_zero = onp.asarray(rmse_before)
# rmse_star = onp.asarray(rmse_tau_star)
# rmse_pred = onp.asarray(rmse_tau_pred)

# plt.figure(figsize=(9, 5))
# plt.bar(x - width, rmse_zero, width, label="tau_zero", alpha=0.8)
# plt.bar(x,         rmse_star, width, label="tau_star", alpha=0.8)
# plt.bar(x + width, rmse_pred, width, label="tau_NN", alpha=0.8)
# plt.xticks(x, marker_names)
# plt.ylabel("RMSE [m]")
# plt.title("Marker RMSE per marker (tau_zero vs tau_star vs tau_NN)")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.savefig("rmse_per_marker_three.svg", format="svg", bbox_inches="tight")
# plt.savefig("rmse_per_marker_three.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("rmse_per_marker_three.jpg", format="jpg", dpi=300, bbox_inches="tight")
# plt.show()

# # Total marker error per configuration (12D norm)
# errnorm_zero = onp.asarray(jnp.sqrt(jnp.sum(errors_before**2, axis=1)))
# errnorm_star = onp.asarray(jnp.sqrt(jnp.sum(errors_tau_star**2, axis=1)))
# errnorm_pred = onp.asarray(jnp.sqrt(jnp.sum(errors_tau_pred**2, axis=1)))

# plt.figure(figsize=(10, 5))
# plt.plot(errnorm_zero, "o-", label="tau_zero")
# plt.plot(errnorm_star, "s-", label="tau_star")
# plt.plot(errnorm_pred, "^-", label="tau_NN")
# plt.xlabel("Sample index")
# plt.ylabel("Total position error norm [m]")
# plt.title("Total marker error per configuration")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.savefig("error_per_configuration_three.svg", format="svg", bbox_inches="tight")
# plt.savefig("error_per_configuration_three.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("error_per_configuration_three.jpg", format="jpg", dpi=300, bbox_inches="tight")
# plt.show()

# # Per-marker error per configuration (4 subplot)
# errs_zero = onp.asarray(errors_before).reshape(-1, 4, 3)      # (M, 4, 3)
# errs_star = onp.asarray(errors_tau_star).reshape(-1, 4, 3)    # (M, 4, 3)
# errs_pred = onp.asarray(errors_tau_pred).reshape(-1, 4, 3)    # (M, 4, 3)
# errnorms_zero = onp.linalg.norm(errs_zero, axis=2)            # (M, 4)
# errnorms_star = onp.linalg.norm(errs_star, axis=2)            # (M, 4)
# errnorms_pred = onp.linalg.norm(errs_pred, axis=2)            # (M, 4)

# fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
# for i, ax in enumerate(axs.ravel()):
#     ax.plot(errnorms_zero[:, i], "o-", label="tau_zero")
#     ax.plot(errnorms_star[:, i], "s-", label="tau_star")
#     ax.plot(errnorms_pred[:, i], "^-", label="tau_NN")
#     ax.set_title(f"{marker_names[i]} error norm")
#     ax.set_ylabel("||error|| [m]")
#     ax.grid(True, linestyle="--", alpha=0.4)
# axs[-1, 0].set_xlabel("Sample index")
# axs[-1, 1].set_xlabel("Sample index")
# axs[0, 0].legend()
# fig.suptitle("Per-marker error per configuration")
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.savefig("per_marker_error_per_configuration_three.svg", format="svg", bbox_inches="tight")
# fig.savefig("per_marker_error_per_configuration_three.pdf", format="pdf", bbox_inches="tight")
# fig.savefig("per_marker_error_per_configuration_three.jpg", format="jpg", dpi=300, bbox_inches="tight")
# plt.show()


# q_star_list = jnp.array(onp.load("q_star_list.npy"))
# print("q_star_list:", q_star_list)
# print("q_star_list size:", q_star_list.shape)



# # ===== Plot: forma del robot per ciascuna tau* =====
q_star_list = []
curves_list = []
for m in range(M):
    u_i = u_batch[:, m]
    tau_i = tau_star_batch[:, m]
    res = solve_equilibrium_with_tau(robot, u_i, tau_i, q0)
    q_star = res.value
    q_star_list.append(onp.asarray(q_star))
    curves_list.append(draw_robot_curve(robot, q_star, num_points=80))

    # curves_list.append(draw_robot_curve(robot, q_star_list[m,:], num_points=80))

# Limiti comuni per i plot
all_pts = onp.concatenate(curves_list, axis=0)  # (sum N_i, 3)
xmin, ymin, zmin = all_pts.min(axis=0)
xmax, ymax, zmax = all_pts.max(axis=0)
pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
lims = ( (xmin - pad, xmax + pad), (ymin - pad, ymax + pad), (zmin - pad, zmax + pad) )

# Plot 1: tutte le curve in un unico 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
for m, curve in enumerate(curves_list):
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=2)
ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
ax.set_title("Forme statiche con tau*")
ax.grid(True)
plt.tight_layout()
plt.savefig("shapes_tau_star_all.svg", bbox_inches="tight")
plt.savefig("shapes_tau_star_all.pdf", bbox_inches="tight")
plt.savefig("shapes_tau_star_all.jpg", dpi=300, bbox_inches="tight")
plt.show()

# # Plot 2: griglia adattiva con overlay marker misurati
ncols = 4
nrows = int(onp.ceil(M / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), subplot_kw={"projection": "3d"})
axs = onp.ravel(axs) if isinstance(axs, onp.ndarray) else [axs]
for m in range(M):
    axm = axs[m]
    curve = curves_list[m]
    axm.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=3, color="tab:blue")
    pm = onp.asarray(measured_markers_batch[:, m]).reshape(4, 3)
    axm.scatter(pm[:, 0], pm[:, 1], pm[:, 2], c=["r","g","m","k"], s=25, marker="o")
    axm.set_title(f"sample {m}")
    axm.set_xlim(*lims[0]); axm.set_ylim(*lims[1]); axm.set_zlim(*lims[2])
    try:
        axm.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    axm.set_xlabel("X"); axm.set_ylabel("Y"); axm.set_zlabel("Z")
    axm.grid(True)
# nascondi assi extra
for k in range(M, len(axs)):
    axs[k].set_visible(False)
plt.tight_layout()
plt.savefig("shapes_tau_star_grid.svg", bbox_inches="tight")
plt.savefig("shapes_tau_star_grid.pdf", bbox_inches="tight")
plt.savefig("shapes_tau_star_grid.jpg", dpi=300, bbox_inches="tight")
plt.show()

# # Save
onp.save("q_star_list.npy", onp.asarray(q_star_list))
onp.savetxt("q_star_list.txt", onp.asarray(q_star_list))


# q_star_list = []
# curves_list = []
# for m in range(M):
#     u_i = u_batch[:, m]
#     tau_i = tau_zero_batch[:, m]
#     res = solve_equilibrium_with_tau(robot, u_i, tau_i, q0)
#     q_star = res.value
#     q_star_list.append(onp.asarray(q_star))
#     curves_list.append(draw_robot_curve(robot, q_star, num_points=80))

# # # Limiti comuni per i plot
# all_pts = onp.concatenate(curves_list, axis=0)  # (sum N_i, 3)
# xmin, ymin, zmin = all_pts.min(axis=0)
# xmax, ymax, zmax = all_pts.max(axis=0)
# pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)
# lims = ( (xmin - pad, xmax + pad), (ymin - pad, ymax + pad), (zmin - pad, zmax + pad) )

# # Plot 1: tutte le curve in un unico 3D
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# for m, curve in enumerate(curves_list):
#     ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=2)
# ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
# try:
#     ax.set_box_aspect((1, 1, 1))
# except Exception:
#     pass
# ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
# ax.set_title("Forme statiche con tau*")
# ax.grid(True)
# plt.tight_layout()
# plt.savefig("shapes_tau_star_all.svg", bbox_inches="tight")
# plt.savefig("shapes_tau_star_all.pdf", bbox_inches="tight")
# plt.savefig("shapes_tau_star_all.jpg", dpi=300, bbox_inches="tight")
# plt.show()

# # Plot 2: griglia adattiva con overlay marker misurati
# ncols = 4
# nrows = int(onp.ceil(M / ncols))
# fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), subplot_kw={"projection": "3d"})
# axs = onp.ravel(axs) if isinstance(axs, onp.ndarray) else [axs]
# for m in range(M):
#     axm = axs[m]
#     curve = curves_list[m]
#     axm.plot(curve[:, 0], curve[:, 1], curve[:, 2], lw=3, color="tab:blue")
#     pm = onp.asarray(measured_markers_batch[:, m]).reshape(4, 3)
#     axm.scatter(pm[:, 0], pm[:, 1], pm[:, 2], c=["r","g","m","k"], s=25, marker="o")
#     axm.set_title(f"sample {m}")
#     axm.set_xlim(*lims[0]); axm.set_ylim(*lims[1]); axm.set_zlim(*lims[2])
#     try:
#         axm.set_box_aspect((1, 1, 1))
#     except Exception:
#         pass
#     axm.set_xlabel("X"); axm.set_ylabel("Y"); axm.set_zlabel("Z")
#     axm.grid(True)
# # nascondi assi extra
# for k in range(M, len(axs)):
#     axs[k].set_visible(False)
# plt.tight_layout()
# plt.savefig("shapes_tau_star_grid.svg", bbox_inches="tight")
# plt.savefig("shapes_tau_star_grid.pdf", bbox_inches="tight")
# plt.savefig("shapes_tau_star_grid.jpg", dpi=300, bbox_inches="tight")
# plt.show()


