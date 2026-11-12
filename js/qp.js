// Torque control of the muscle model, ported from musclemimic/core/control_functions/torque_qp.py and
// musclemimic/core/qp_recruitment.py (TorqueQPControl._solve, QPRecruitment.gain_bias / projected_arms /
// solve_soft). The policy commands 31 joint torques; every control step a QP recruits the 354 muscles:
//   tau_cmd = clip(action, -1, 1) * scale
//   lo / hi = activation reachable by the end of the step with excitation 0 / 1, a = lo + D x, D = hi - lo
//   min 1/2 |a|^2 + cocon.a + 1/2 sum_j w_j (A a + P M^T b - tau_cmd)_j^2,  0 <= x <= 1   (interior point)
//   excitation u reaching a in one step (bisection), which is the ctrl of each muscle.
// Same algorithm, same constants; float64 here, float32 in the jitted Python.

const TINY = 1e-12;

export class TorqueQP {
  constructor(blob) {
    const m = blob.meta;
    this.nIter = m.n_iter; this.nBisect = m.n_bisect; this.mu = m.cocon_mu; this.dt = m.dt; this.nSub = m.n_sub;
    this.nCtrl = m.n_ctrl; this.eps = m.eps; this.minval = m.minval; this.names = m.dof_names;
    this.mus = Int32Array.from(blob.muscles); this.ind = Int32Array.from(blob.ind_dofs);
    this.sag = Int32Array.from(blob.sag_dofs);
    this.scale = Float64Array.from(blob.scale); this.w = Float64Array.from(blob.w);
    this.dyn = Float64Array.from(blob.dyn); this.actAdr = Int32Array.from(blob.act_adr);
    this.musPos = Int32Array.from(blob.mus_pos); this.actionIdx = Int32Array.from(blob.action_indices);
    const nm = this.nm = this.mus.length, ni = this.ni = this.ind.length;
    this.prm = {};
    for (const k of ["range0", "lr0", "L0", "F0", "lmin", "lmax", "vmax", "fpmax", "fvmax"]) this.prm[k] = blob["prm_" + k];
    // dof -> column of the independent coordinate it is (or is folded into), and how
    this.nv = m.nv;
    this.indCol = new Int32Array(m.nv).fill(-1);
    for (let c = 0; c < ni; c++) this.indCol[this.ind[c]] = c;
    this.dep = [];
    for (let k = 0; k < blob.dep_dof.length; k++) {
      this.dep.push({ dof: blob.dep_dof[k], col: blob.dep_col[k], qadr: blob.dep_qadr[k],
        c: Array.from(blob.dep_coef.subarray(5 * k, 5 * k + 5)) });
    }
    this.depOf = new Map(this.dep.map((d) => [d.dof, d]));
    this.isSag = new Uint8Array(m.nv);
    for (const j of this.sag) this.isSag[j] = 1;
    // work buffers
    this.G = new Float64Array(nm); this.b = new Float64Array(nm);
    this.Mp = new Float64Array(nm * ni); this.sagAbs = new Float64Array(nm);
    this.A = new Float64Array(ni * nm); this.r = new Float64Array(ni); this.c = new Float64Array(nm);
    this.a0 = new Float64Array(nm); this.lo = new Float64Array(nm); this.D = new Float64Array(nm);
    this.ones = new Float64Array(nm).fill(1); this.zeros = new Float64Array(nm);
    this.tmp = new Float64Array(nm); this.u = new Float64Array(nm);
    this.ctrl = new Float64Array(this.nCtrl);
    this.tauCmd = new Float64Array(ni); this.unmet = new Float64Array(ni); this.achieved = new Float64Array(ni);
    this.ip = new IPM(ni, nm, this.nIter, this.eps);
    // exoskeleton motors after the torques in the action (none for the plain human): [-1, 1] -> ctrlrange,
    // 2nd-order output low-pass, torque on the independent coordinates; "ideal" adaptation takes it off the muscles
    const ne = this.nExo = m.n_exo || 0;
    this.target = new Float64Array(ni);
    this.exoTau = new Float64Array(ni);
    if (ne) {
      this.exoIdeal = m.exo_ideal; this.exoNames = m.exo_names;
      this.exoPos = Int32Array.from(blob.exo_pos);
      this.exoMean = Float64Array.from(blob.exo_mean); this.exoDelta = Float64Array.from(blob.exo_delta);
      this.exoLo = Float64Array.from(blob.exo_lo); this.exoHi = Float64Array.from(blob.exo_hi);
      this.exoMap = Float64Array.from(blob.exo_map);                     // (ni, ne)
      this.filt = Float64Array.from(blob.exo_filt);                      // (5, ne): b0 b1 b2 a1 a2
      this.fx1 = new Float64Array(ne); this.fx2 = new Float64Array(ne);
      this.fy1 = new Float64Array(ne); this.fy2 = new Float64Array(ne);
      this.exoRaw = new Float64Array(ne); this.exoOut = new Float64Array(ne);
    }
  }

  // the control function's state at a reset (TorqueQPControl.init_state)
  reset() {
    if (this.nExo) { this.fx1.fill(0); this.fx2.fill(0); this.fy1.fill(0); this.fy2.fill(0); this.exoOut.fill(0); }
    this.achieved.fill(0); this.unmet.fill(0); this.tauCmd.fill(0); this.exoTau.fill(0);
  }

  // the exo motors' command this step (Butterworth state advanced), and its torque on the coordinates
  exoStep(action) {
    const ne = this.nExo, ni = this.ni, f = this.filt;
    for (let k = 0; k < ne; k++) {
      const a = Math.min(Math.max(action[ni + k], -1), 1);
      const x = Math.min(Math.max(a * this.exoDelta[k] + this.exoMean[k], this.exoLo[k]), this.exoHi[k]);
      const y = Math.min(Math.max(f[k] * x + f[ne + k] * this.fx1[k] + f[2 * ne + k] * this.fx2[k]
        - f[3 * ne + k] * this.fy1[k] - f[4 * ne + k] * this.fy2[k], this.exoLo[k]), this.exoHi[k]);
      this.exoRaw[k] = x; this.exoOut[k] = y;
      this.fx2[k] = this.fx1[k]; this.fx1[k] = x; this.fy2[k] = this.fy1[k]; this.fy1[k] = y;
    }
    for (let c = 0; c < ni; c++) {
      let s = 0;
      for (let k = 0; k < ne; k++) s += this.exoMap[c * ne + k] * this.exoOut[k];
      this.exoTau[c] = s;
    }
  }

  // MuJoCo's muscle model is affine in activation: F = G(l, v) a + b(l)  (mju_muscleGain / mju_muscleBias)
  gainBias(length, velocity) {
    const p = this.prm, MV = this.minval;
    for (let k = 0; k < this.nm; k++) {
      const i = this.mus[k];
      const L = p.range0[k] + (length[i] - p.lr0[k]) / Math.max(MV, p.L0[k]);
      const V = velocity[i] / Math.max(MV, p.L0[k] * p.vmax[k]);
      const lmin = p.lmin[k], lmax = p.lmax[k];
      const a = 0.5 * (lmin + 1), b = 0.5 * (1 + lmax);
      let FL = 0;
      if (L >= lmin && L <= a) { const x = (L - lmin) / Math.max(MV, a - lmin); FL = 0.5 * x * x; }
      else if (L > a && L <= 1) { const x = (1 - L) / Math.max(MV, 1 - a); FL = 1 - 0.5 * x * x; }
      else if (L > 1 && L <= b) { const x = (L - 1) / Math.max(MV, b - 1); FL = 1 - 0.5 * x * x; }
      else if (L > b && L <= lmax) { const x = (lmax - L) / Math.max(MV, lmax - b); FL = 0.5 * x * x; }
      const fvmax = p.fvmax[k], y = fvmax - 1;
      let FV;
      if (V <= -1) FV = 0;
      else if (V <= 0) FV = (V + 1) * (V + 1);
      else if (V <= y) FV = fvmax - (y - V) * (y - V) / Math.max(MV, y);
      else FV = fvmax;
      let FP;
      if (L <= 1) FP = 0;
      else if (L <= b) { const x = (L - 1) / Math.max(MV, b - 1); FP = 0.5 * p.fpmax[k] * x * x; }
      else { const x = (L - b) / Math.max(MV, b - 1); FP = p.fpmax[k] * (0.5 + x); }
      this.G[k] = -p.F0[k] * FL * FV;
      this.b[k] = -p.F0[k] * FP;
    }
  }

  // Mp = M P^T (muscle moment arms on the independent coordinates; dependent dofs folded into their driver with
  // the coupling's derivative) from the sparse actuator_moment rows, and sum_j in sagittal |M_ij|
  projectedArms(data) {
    const Mp = this.Mp, ni = this.ni;
    Mp.fill(0); this.sagAbs.fill(0);
    const val = data.actuator_moment, nnz = data.moment_rownnz, adr = data.moment_rowadr, col = data.moment_colind;
    const qpos = data.qpos;
    for (let k = 0; k < this.nm; k++) {
      const i = this.mus[k], a0 = adr[i], n = nnz[i];
      for (let t = 0; t < n; t++) {
        const j = col[a0 + t], v = val[a0 + t];
        if (this.isSag[j]) this.sagAbs[k] += Math.abs(v);
        const c = this.indCol[j];
        if (c >= 0) Mp[k * ni + c] += v;
        const d = this.depOf.get(j);
        if (d) {
          const x = qpos[d.qadr], cf = d.c;
          Mp[k * ni + d.col] += v * (cf[1] + 2 * cf[2] * x + 3 * cf[3] * x * x + 4 * cf[4] * x * x * x);
        }
      }
    }
  }

  // MuJoCo muscle activation dynamics (tausmooth 0, Euler) over nSub physics steps, elementwise
  actStep(act, u, out) {
    const dyn = this.dyn, dt = this.dt;
    for (let k = 0; k < this.nm; k++) {
      let a = act[k];
      const uk = u[k];
      for (let s = 0; s < this.nSub; s++) {
        const ac = Math.min(Math.max(a, 0), 1);
        const tau = uk - a > 0 ? dyn[2 * k] * (0.5 + 1.5 * ac) : dyn[2 * k + 1] / (0.5 + 1.5 * ac);
        a = a + dt * (uk - a) / Math.max(tau, 1e-10);
      }
      out[k] = a;
    }
    return out;
  }

  // action: the policy's 31 numbers. Writes this.ctrl (length nCtrl, the order of env._action_indices), tauCmd,
  // unmet (achieved - commanded, / scale) and achieved (/ scale) exactly as TorqueQPControl.generate_action.
  solve(action, data) {
    const ni = this.ni, nm = this.nm, A = this.A, Mp = this.Mp, G = this.G;
    for (let c = 0; c < ni; c++) this.tauCmd[c] = Math.min(Math.max(action[c], -1), 1) * this.scale[c];
    if (this.nExo) this.exoStep(action);
    for (let c = 0; c < ni; c++) this.target[c] = this.nExo && this.exoIdeal ? this.tauCmd[c] - this.exoTau[c] : this.tauCmd[c];
    this.gainBias(data.actuator_length, data.actuator_velocity);
    this.projectedArms(data);
    for (let c = 0; c < ni; c++) {
      let pb = 0;
      for (let k = 0; k < nm; k++) { A[c * nm + k] = Mp[k * ni + c] * G[k]; pb += Mp[k * ni + c] * this.b[k]; }
      this.r[c] = this.target[c] - pb;                       // active part of what the muscles must produce
    }
    for (let k = 0; k < nm; k++) {
      this.c[k] = 0.5 * this.mu * (-G[k]) * this.sagAbs[k];
      this.a0[k] = data.act[this.actAdr[k]];
    }
    this.actStep(this.a0, this.zeros, this.lo);
    this.actStep(this.a0, this.ones, this.tmp);
    for (let k = 0; k < nm; k++) this.D[k] = Math.max(this.tmp[k] - this.lo[k], 1e-4);
    // QP in x: A D, r - A lo, D (lo + c), D^2
    const ip = this.ip;
    for (let c = 0; c < ni; c++) {
      let s = 0;
      for (let k = 0; k < nm; k++) { ip.A[c * nm + k] = A[c * nm + k] * this.D[k]; s += A[c * nm + k] * this.lo[k]; }
      ip.r[c] = this.r[c] - s;
    }
    for (let k = 0; k < nm; k++) { ip.c[k] = this.D[k] * (this.lo[k] + this.c[k]); ip.q[k] = this.D[k] * this.D[k]; }
    ip.w.set(this.w);
    const x = ip.solveSoft();
    // a* = lo + D x, then the excitation that reaches it in one step
    const aStar = this.tmp;
    for (let k = 0; k < nm; k++) aStar[k] = this.lo[k] + this.D[k] * x[k];
    const ul = new Float64Array(nm), uh = new Float64Array(nm).fill(1), um = this.u, reach = new Float64Array(nm);
    for (let it = 0; it < this.nBisect; it++) {
      for (let k = 0; k < nm; k++) um[k] = 0.5 * (ul[k] + uh[k]);
      this.actStep(this.a0, um, reach);
      for (let k = 0; k < nm; k++) { if (reach[k] > aStar[k]) uh[k] = um[k]; else ul[k] = um[k]; }
    }
    this.ctrl.fill(0);
    for (let k = 0; k < nm; k++) this.ctrl[this.musPos[k]] = 0.5 * (ul[k] + uh[k]);
    for (let k = 0; k < this.nExo; k++) this.ctrl[this.exoPos[k]] = this.exoOut[k];
    for (let c = 0; c < ni; c++) {
      this.unmet[c] = ip.s[c] / this.scale[c];
      this.achieved[c] = this.tauCmd[c] / this.scale[c] + this.unmet[c];
    }
    return this.ctrl;
  }

  // write the muscle excitations into data.ctrl (the env's action indices)
  apply(data) {
    const ctrl = data.ctrl;
    for (let t = 0; t < this.nCtrl; t++) ctrl[this.actionIdx[t]] = this.ctrl[t];
  }
}

// min 1/2 x.diag(q).x + c.x + 1/2 sum_j w_j (A x - r)_j^2,  0 <= x <= 1   (QPRecruitment.solve_soft)
// Primal-dual interior point, Mehrotra predictor-corrector, fixed iteration count; the shortfall s = A x - r is
// eliminated through its multiplier (s = -y / w), so the reduced KKT matrix is A diag(1/H) A^T + diag(1/w).
// A (m x n, dense input) is used column-sparse: a muscle spans a few joints, so most of A is zero and
// A diag(1/H) A^T is a sum of small outer products. Same arithmetic as the dense form, in another order.
class IPM {
  constructor(m, n, nIter, eps) {
    this.m = m; this.n = n; this.nIter = nIter; this.eps = eps;
    this.A = new Float64Array(m * n); this.r = new Float64Array(m); this.c = new Float64Array(n);
    this.q = new Float64Array(n); this.w = new Float64Array(m); this.s = new Float64Array(m);
    this.colPtr = new Int32Array(n + 1); this.rowIdx = new Int32Array(m * n); this.val = new Float64Array(m * n);
    this.rs = new Float64Array(m); this.winv = new Float64Array(m); this.rho = new Float64Array(m);
    this.x = new Float64Array(n); this.y = new Float64Array(m); this.zl = new Float64Array(n); this.zu = new Float64Array(n);
    this.K = new Float64Array(m * m); this.H = new Float64Array(n); this.g = new Float64Array(n);
    this.rhs = new Float64Array(m); this.ax = new Float64Array(m); this.agh = new Float64Array(m);
    this.dx = new Float64Array(n); this.dy = new Float64Array(m); this.dzl = new Float64Array(n); this.dzu = new Float64Array(n);
    this.dxa = new Float64Array(n); this.dya = new Float64Array(m); this.dzla = new Float64Array(n); this.dzua = new Float64Array(n);
    this.ca = new Float64Array(n); this.cb = new Float64Array(n);
  }

  // row scaling (rho = row norm) and the column-sparse copy of the scaled A
  prepare() {
    const { m, n, A, rho, colPtr, rowIdx, val } = this;
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += A[j * n + i] * A[j * n + i];
      rho[j] = Math.sqrt(s) + 1e-9;
      this.rs[j] = this.r[j] / rho[j];
      this.winv[j] = 1 / (this.w[j] * rho[j] * rho[j]);
    }
    let nz = 0;
    for (let i = 0; i < n; i++) {
      colPtr[i] = nz;
      for (let j = 0; j < m; j++) {
        const a = A[j * n + i];
        if (a !== 0) { rowIdx[nz] = j; val[nz] = a / rho[j]; nz++; }
      }
    }
    colPtr[n] = nz;
  }

  newton(sigMu, ca, cb, dx, dy, dzl, dzu) {
    const { m, n, colPtr, rowIdx, val, q, c, x, y, zl, zu, H, g, K, rhs, winv, ax, agh } = this;
    K.fill(0); ax.fill(0); agh.fill(0);
    for (let i = 0; i < n; i++) {
      const xl = Math.max(x[i], TINY), xu = Math.max(1 - x[i], TINY);
      const p0 = colPtr[i], p1 = colPtr[i + 1];
      let aty = 0;
      for (let p = p0; p < p1; p++) aty += val[p] * y[rowIdx[p]];
      const rd = q[i] * x[i] + c[i] - aty - zl[i] + zu[i];
      const h = q[i] + zl[i] / xl + zu[i] / xu;
      H[i] = h;
      const gi = -rd + (-xl * zl[i] + sigMu - (ca ? ca[i] : 0)) / xl - (-xu * zu[i] + sigMu + (cb ? cb[i] : 0)) / xu;
      g[i] = gi;
      const ih = 1 / h, gh = gi * ih, xi = x[i];
      for (let p = p0; p < p1; p++) {
        const j = rowIdx[p], v = val[p];
        ax[j] += v * xi; agh[j] += v * gh;
        const vh = v * ih;
        for (let t = p0; t <= p; t++) K[j * m + rowIdx[t]] += vh * val[t];   // rows ascend within a column
      }
    }
    for (let j = 0; j < m; j++) {
      rhs[j] = -(ax[j] - this.rs[j] + winv[j] * y[j]) - agh[j];
      K[j * m + j] += winv[j] + this.eps;
    }
    cholSolve(K, rhs, m, dy);
    for (let i = 0; i < n; i++) {
      let aty = 0;
      for (let p = colPtr[i]; p < colPtr[i + 1]; p++) aty += val[p] * dy[rowIdx[p]];
      dx[i] = (g[i] + aty) / H[i];
      const xl = Math.max(x[i], TINY), xu = Math.max(1 - x[i], TINY);
      dzl[i] = (-xl * zl[i] + sigMu - (ca ? ca[i] : 0) - zl[i] * dx[i]) / xl;
      dzu[i] = (-xu * zu[i] + sigMu + (cb ? cb[i] : 0) + zu[i] * dx[i]) / xu;
    }
  }

  solveSoft() {
    const { m, n, x, y, zl, zu } = this;
    this.prepare();
    x.fill(0.5); y.fill(0); zl.fill(1); zu.fill(1);
    const { dxa, dya, dzla, dzua, dx, dy, dzl, dzu, ca, cb } = this;
    for (let it = 0; it < this.nIter; it++) {
      let mu = 0;
      for (let i = 0; i < n; i++) mu += Math.max(x[i], TINY) * zl[i] + Math.max(1 - x[i], TINY) * zu[i];
      mu /= 2 * n;
      this.newton(0, null, null, dxa, dya, dzla, dzua);                       // predictor
      let ap = Math.min(maxStepX(x, dxa, n), 1), ad = Math.min(maxStep(zl, dzla, n), maxStep(zu, dzua, n));
      let muAff = 0;
      for (let i = 0; i < n; i++) {
        const xl = Math.max(x[i], TINY), xu = Math.max(1 - x[i], TINY);
        muAff += (xl + ap * dxa[i]) * (zl[i] + ad * dzla[i]) + (xu - ap * dxa[i]) * (zu[i] + ad * dzua[i]);
      }
      muAff /= 2 * n;
      const sig = Math.min(Math.max(Math.pow(muAff / Math.max(mu, TINY), 3), 0), 1);
      for (let i = 0; i < n; i++) { ca[i] = dxa[i] * dzla[i]; cb[i] = dxa[i] * dzua[i]; }
      this.newton(sig * mu, ca, cb, dx, dy, dzl, dzu);                         // corrector
      ap = 0.995 * Math.min(maxStepX(x, dx, n), 1);
      ad = 0.995 * Math.min(maxStep(zl, dzl, n), maxStep(zu, dzu, n));
      for (let i = 0; i < n; i++) {
        x[i] += ap * dx[i];
        zl[i] = Math.max(zl[i] + ad * dzl[i], TINY);
        zu[i] = Math.max(zu[i] + ad * dzu[i], TINY);
      }
      for (let j = 0; j < m; j++) y[j] += ad * dy[j];
    }
    for (let i = 0; i < n; i++) x[i] = Math.min(Math.max(x[i], 0), 1);
    const s = this.s;
    s.fill(0);
    for (let i = 0; i < n; i++) for (let p = this.colPtr[i]; p < this.colPtr[i + 1]; p++) {
      const j = this.rowIdx[p];
      s[j] += this.val[p] * this.rho[j] * x[i];
    }
    for (let j = 0; j < m; j++) s[j] -= this.r[j];
    return x;
  }
}

// min over i of -v_i / dv_i for dv_i < 0, capped at 1 (max_step)
function maxStep(v, dv, n) {
  let s = 1;
  for (let i = 0; i < n; i++) if (dv[i] < 0) s = Math.min(s, -v[i] / Math.min(dv[i], -TINY));
  return s;
}

// step limit for x in [0, 1]: min(max_step(xl, dx), max_step(xu, -dx))
function maxStepX(x, dx, n) {
  let s = 1;
  for (let i = 0; i < n; i++) {
    const xl = Math.max(x[i], TINY), xu = Math.max(1 - x[i], TINY), d = dx[i];
    if (d < 0) s = Math.min(s, -xl / Math.min(d, -TINY));
    else if (d > 0) s = Math.min(s, xu / Math.max(d, TINY));
  }
  return s;
}

// K (lower triangle, m x m) = L L^T; solve K z = b into out
function cholSolve(K, b, m, out) {
  const L = K;
  for (let j = 0; j < m; j++) {
    let d = L[j * m + j];
    for (let k = 0; k < j; k++) d -= L[j * m + k] * L[j * m + k];
    d = Math.sqrt(d);
    L[j * m + j] = d;
    for (let i = j + 1; i < m; i++) {
      let s = L[i * m + j];
      for (let k = 0; k < j; k++) s -= L[i * m + k] * L[j * m + k];
      L[i * m + j] = s / d;
    }
  }
  // (only the lower triangle K[i*m+k], k <= i, is read and overwritten). Forward / back substitution:
  for (let i = 0; i < m; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * m + k] * out[k];
    out[i] = s / L[i * m + i];
  }
  for (let i = m - 1; i >= 0; i--) {
    let s = out[i];
    for (let k = i + 1; k < m; k++) s -= L[k * m + i] * out[k];
    out[i] = s / L[i * m + i];
  }
}
