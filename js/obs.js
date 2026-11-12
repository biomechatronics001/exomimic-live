// The torque-QP human's 656-number observation, built from MjData as the CPU env builds it (the entries and their
// indices come from assets/human/obs_layout.json, written by export/export_clips.py):
//   root_gravity 3 | q_all_pos 82 | dq_free_joint_heading 6 | dq_all_vel 82 | last_action 31 | achieved_torque 31 |
//   foot_height_scan 36 | touch x4 | GoalTrajMimic 381
// and the terminal check of MeanRelativeSiteDeviationWithRootTerminalStateHandler.
// `st` is the env carry the observation reads: { clip, t (reference frame), lastAction, achieved }.

const yawOf = (w, x, y, z) => Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));

// rotation matrix (row-major 3x3) -> rotation vector, as scipy Rotation.from_matrix(...).as_rotvec()
function matToRotvec(m) {
  const tr = m[0] + m[4] + m[8];
  const dec = [m[0], m[4], m[8], tr];
  let c = 0;
  for (let i = 1; i < 4; i++) if (dec[i] > dec[c]) c = i;
  const q = [0, 0, 0, 0];                                     // x, y, z, w
  if (c !== 3) {
    const i = c, j = (i + 1) % 3, k = (j + 1) % 3;
    q[i] = 1 - tr + 2 * m[i * 3 + i];
    q[j] = m[j * 3 + i] + m[i * 3 + j];
    q[k] = m[k * 3 + i] + m[i * 3 + k];
    q[3] = m[k * 3 + j] - m[j * 3 + k];
  } else {
    q[0] = m[7] - m[5]; q[1] = m[2] - m[6]; q[2] = m[3] - m[1]; q[3] = 1 + tr;
  }
  let n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (q[3] < 0) n = -n;
  const x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  const s = Math.hypot(x, y, z);
  const angle = 2 * Math.atan2(s, w);
  const scale = angle <= 1e-3 ? 2 + angle * angle / 12 + 7 * angle ** 4 / 2880 : angle / Math.sin(angle / 2);
  return [scale * x, scale * y, scale * z];
}

export class HumanObs {
  constructor(layout, terrain) {
    this.layout = layout;
    this.dim = layout.obs_dim;
    this.terrain = terrain;
    this.e = Object.fromEntries(layout.entries.map((e) => [e.name, e]));
    this.goal = layout.entries.find((e) => e.type === "GoalTrajMimic");
    this.touch = layout.entries.filter((e) => e.type === "TouchSensor");
    this.nRel = this.goal.site_ids.length - 1;
    this.obs = new Float64Array(this.dim);
    this.simRpos = new Float64Array(3 * this.nRel);
    const t = layout.terminal || {};
    this.termSite = t.params && t.params.mean_site_deviation_threshold;
    this.termThreshold = t.threshold != null ? t.threshold : this.termSite;
    this.rootThreshold = t.params ? t.params.root_deviation_threshold : 0.5;
    this.rootSite = t.root_site_id;
  }

  // world-frame positions of the mimic sites relative to the main (pelvis) site, from the simulator
  simSiteRpos(data) {
    const ids = this.goal.site_ids, sx = data.site_xpos, m0 = 3 * ids[0];
    for (let k = 1; k <= this.nRel; k++) {
      const s = 3 * ids[k];
      for (let a = 0; a < 3; a++) this.simRpos[3 * (k - 1) + a] = sx[s + a] - sx[m0 + a];
    }
    return this.simRpos;
  }

  build(data, st) {
    const o = this.obs, qpos = data.qpos, qvel = data.qvel;
    let p = 0;
    const put = (v) => { o[p++] = v; };
    for (const e of this.layout.entries) {
      if (p !== e.start) throw new Error(`obs layout: ${e.name} starts at ${e.start}, writing at ${p}`);
      switch (e.type) {
        case "ProjectedGravityVector": {
          const [iw, ix, iy, iz] = e.data_ind;
          let w = qpos[iw], x = qpos[ix], y = qpos[iy], z = qpos[iz];
          const n = Math.hypot(w, x, y, z); w /= n; x /= n; y /= n; z /= n;
          // R^T (0, 0, -1) = -(third row of R)
          put(-2 * (x * z - w * y)); put(-2 * (y * z + w * x)); put(-(1 - 2 * (x * x + y * y)));
          break;
        }
        case "JointPosArray": for (const i of e.data_ind) put(qpos[i]); break;
        case "JointVelArray": for (const i of e.data_ind) put(qvel[i]); break;
        case "FreeJointVelHeading": {
          const [iw, ix, iy, iz] = e.quat_ind;
          const yaw = yawOf(qpos[iw], qpos[ix], qpos[iy], qpos[iz]);
          const c = Math.cos(yaw), s = Math.sin(yaw), v = e.qvel_ind.map((i) => qvel[i]);
          put(c * v[0] + s * v[1]); put(-s * v[0] + c * v[1]); put(v[2]);
          put(v[3]); put(v[4]); put(v[5]);
          break;
        }
        case "LastAction": {
          const cl = e.clip;
          for (let i = 0; i < e.dim; i++) {
            const a = st.lastAction[i];
            put(cl != null ? Math.min(Math.max(a, -cl), cl) : a);
          }
          break;
        }
        case "ControlFuncField": for (let i = 0; i < e.dim; i++) put(st.achieved[i]); break;
        case "FootHeightScan": {
          const R = data.xmat, r0 = 9 * e.root_body;
          let fx = e.sign * R[r0 + e.axis], fy = e.sign * R[r0 + 3 + e.axis];
          const fn = Math.max(Math.hypot(fx, fy), 1e-6); fx /= fn; fy /= fn;
          const lx = -fy, ly = fx;
          for (const b of e.feet) {
            const X = data.xpos[3 * b], Y = data.xpos[3 * b + 1], Z = data.xpos[3 * b + 2];
            for (let g = 0; g < e.off_f.length; g++) {
              const px = X + e.off_f[g] * fx + e.off_l[g] * lx, py = Y + e.off_f[g] * fy + e.off_l[g] * ly;
              put(this.terrain.height(px, py) - Z);
            }
          }
          break;
        }
        case "TouchSensor": put(data.sensordata[e.adr]); break;
        case "GoalTrajMimic": p = this.buildGoal(data, st, p); break;
        default: throw new Error(`observation type ${e.type} not ported`);
      }
    }
    if (p !== this.dim) throw new Error(`obs: wrote ${p} of ${this.dim}`);
    return o;
  }

  buildGoal(data, st, p) {
    const g = this.goal, o = this.obs, qpos = data.qpos, qvel = data.qvel, clip = st.clip, n = this.nRel;
    const [iw, ix, iy, iz] = g.root_quat_ind;
    const yaw = yawOf(qpos[iw], qpos[ix], qpos[iy], qpos[iz]);
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const head = (x, y, z) => { o[p++] = c * x + s * y; o[p++] = -s * x + c * y; o[p++] = z; };
    // simulated sites relative to the pelvis (heading frame), then their relative rotations
    const rp = this.simSiteRpos(data);
    for (let k = 0; k < n; k++) head(rp[3 * k], rp[3 * k + 1], rp[3 * k + 2]);
    const ids = g.site_ids, X = data.site_xmat, m0 = 9 * ids[0], rel = new Float64Array(9);
    for (let k = 1; k <= n; k++) {
      const m1 = 9 * ids[k];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        rel[3 * a + b] = X[m0 + a] * X[m1 + b] + X[m0 + 3 + a] * X[m1 + 3 + b] + X[m0 + 6 + a] * X[m1 + 6 + b];
      }
      const rv = matToRotvec(rel);
      o[p++] = rv[0]; o[p++] = rv[1]; o[p++] = rv[2];
    }
    if (g.site_rvel) throw new Error("site_rvel observations not ported");
    // concise lookahead: [step0 site_rpos, (root pos delta, root vel delta, site_rpos) x (n_lookahead - 1)]
    const T = clip.length, t = st.t, nq = clip.nq, nv = clip.nv;
    const Q = clip.qpos, V = clip.qvel, S = clip.site_rpos;
    const rq = g.root_qpos_ind, rv = g.root_qvel_ind;
    const sites = (f) => { for (let k = 0; k < n; k++) head(S[(f * n + k) * 3], S[(f * n + k) * 3 + 1], S[(f * n + k) * 3 + 2]); };
    sites(t);
    for (let k = 1; k < g.n_lookahead; k++) {
      const f = Math.max(0, Math.min(t + k * g.stride, T - 1));
      head(Q[f * nq + rq[0]] - Q[t * nq + rq[0]], Q[f * nq + rq[1]] - Q[t * nq + rq[1]], Q[f * nq + rq[2]] - Q[t * nq + rq[2]]);
      head(V[f * nv + rv[0]] - V[t * nv + rv[0]], V[f * nv + rv[1]] - V[t * nv + rv[1]], V[f * nv + rv[2]] - V[t * nv + rv[2]]);
      for (let a = 3; a < 6; a++) o[p++] = V[f * nv + rv[a]] - V[t * nv + rv[a]];
      sites(f);
    }
    if (g.root_dev) {
      if (g.rezero_root_xy) throw new Error("rezero_root_xy clips not ported");
      head(Q[t * nq + rq[0]] - qpos[rq[0]], Q[t * nq + rq[1]] - qpos[rq[1]], Q[t * nq + rq[2]] - qpos[rq[2]]);
      const dyaw = yawOf(Q[t * nq + iw], Q[t * nq + ix], Q[t * nq + iy], Q[t * nq + iz]) - yaw;
      o[p++] = Math.sin(dyaw); o[p++] = Math.cos(dyaw);
      head(V[t * nv + rv[0]] - qvel[rv[0]], V[t * nv + rv[1]] - qvel[rv[1]], V[t * nv + rv[2]] - qvel[rv[2]]);
    }
    if (g.phase) o[p++] = t / Math.max(T, 1);
    return p;
  }

  // MeanRelativeSiteDeviationWithRootTerminalStateHandler: mean |sim - ref| of the pelvis-relative sites, and the
  // pelvis site's distance from the reference's. Returns { fallen, siteDev, rootDev }.
  terminal(data, st) {
    const clip = st.clip, t = st.t, n = this.nRel, S = clip.site_rpos, rp = this.simSiteRpos(data);
    let dev = 0;
    for (let k = 0; k < n; k++) {
      const a = 3 * k, b = (t * n + k) * 3;
      dev += Math.hypot(rp[a] - S[b], rp[a + 1] - S[b + 1], rp[a + 2] - S[b + 2]);
    }
    dev /= n;
    const sx = data.site_xpos, r = 3 * this.rootSite, R = clip.root_xpos;
    const root = Math.hypot(sx[r] - R[3 * t], sx[r + 1] - R[3 * t + 1], sx[r + 2] - R[3 * t + 2]);
    return { fallen: dev > this.termThreshold || root > this.rootThreshold, siteDev: dev, rootDev: root };
  }
}
