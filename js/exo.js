// The T-RO exoskeleton controller as trained (musclemimic/algorithms/ppo/exo_residual.py):
//   reading   [angle_r, angle_l, velocity_r, velocity_l] of the observed joints (the human observation's entries)
//   window    a buffer of history + delay readings, filled with the first reading at a reset; the network reads the
//             `history` readings that end `delay` control steps before the newest (ExoAssistWrapper._history_block)
//   network   SharedLegActorCritic: every leg's own [angle, velocity] x history, then the other leg's
//             (contralateral), normalised with the training statistics, through ONE shared MLP -> that leg's motor
//   output    the policy mean in [-1, 1] (x the user's assist scale); the control function maps it to the motor
//             range, low-passes it and takes it off the muscles (qp.js)

function silu(v) { return v / (1 + Math.exp(-v)); }

class LegNet {
  constructor(blob, n, eps) {
    this.layers = [];
    for (let i = 0; i < n; i++) {
      const K = blob[`Dense_${i}.kernel`], [nin, nout] = K.shape;
      const W = new Float32Array(nin * nout);
      for (let a = 0; a < nin; a++) for (let o = 0; o < nout; o++) W[o * nin + a] = K[a * nout + o];
      this.layers.push({ W, b: blob[`Dense_${i}.bias`], nin, nout,
        s: i < n - 1 ? blob[`LayerNorm_${i}.scale`] : null, lb: i < n - 1 ? blob[`LayerNorm_${i}.bias`] : null });
    }
    this.eps = eps;
  }

  forward(x) {
    for (const L of this.layers) {
      const y = new Float64Array(L.nout);
      for (let o = 0; o < L.nout; o++) {
        let s = L.b[o];
        const r = o * L.nin;
        for (let i = 0; i < L.nin; i++) s += L.W[r + i] * x[i];
        y[o] = s;
      }
      if (L.s) {
        let m = 0;
        for (let i = 0; i < y.length; i++) m += y[i];
        m /= y.length;
        let v = 0;
        for (let i = 0; i < y.length; i++) v += (y[i] - m) * (y[i] - m);
        const inv = 1 / Math.sqrt(v / y.length + this.eps);
        for (let i = 0; i < y.length; i++) y[i] = silu((y[i] - m) * inv * L.s[i] + L.lb[i]);
      }
      x = y;
    }
    return x;
  }
}

export class ExoController {
  constructor(blob) {
    const m = this.meta = blob.meta;
    if (m.arch !== "shared_leg" || m.activation !== "silu" || m.ko !== 1) throw new Error("unsupported exo actor");
    this.H = m.history; this.d = m.delay; this.L = m.buf_len; this.nc = m.n_channels;
    this.histInd = Int32Array.from(blob.hist_ind);
    this.mean = Float64Array.from(blob.in_mean);
    this.invStd = Float64Array.from(blob.in_var, (v) => 1 / Math.sqrt(v + m.obs_eps));
    this.net = new LegNet(blob, m.n_dense, m.layernorm_eps);
    this.buf = new Float64Array(this.L * this.nc);
    this.action = new Float64Array(2);
    this.enabled = true;
    this.scale = 1.0;
  }

  reading(obs, out = new Float64Array(this.nc)) {
    for (let c = 0; c < this.nc; c++) out[c] = obs[this.histInd[c]];
    return out;
  }

  reset(obs) {
    const r = this.reading(obs);
    for (let t = 0; t < this.L; t++) this.buf.set(r, t * this.nc);
  }

  push(obs) {
    this.buf.copyWithin(0, this.nc);
    this.buf.set(this.reading(obs), (this.L - 1) * this.nc);
  }

  // the H readings the network reads (oldest first), (H, nc) flattened
  block() {
    const s = (this.L - this.H - this.d) * this.nc;
    return this.buf.subarray(s, s + this.H * this.nc);
  }

  // mean action [right, left] from a window block (H x [ang_r, ang_l, vel_r, vel_l], raw readings)
  forwardBlock(block) {
    const H = this.H, nc = this.nc;
    const z = new Float64Array(H * nc);
    for (let i = 0; i < H * nc; i++) z[i] = (block[i] - this.mean[i]) * this.invStd[i];
    const legIn = (leg) => {                                // per step [angle, velocity] of that leg
      const v = new Float64Array(2 * H);
      for (let h = 0; h < H; h++) { v[2 * h] = z[h * nc + leg]; v[2 * h + 1] = z[h * nc + 2 + leg]; }
      return v;
    };
    const r = legIn(0), l = legIn(1);
    const cat = (a, b) => { const v = new Float64Array(a.length + b.length); v.set(a); v.set(b, a.length); return v; };
    const rin = this.meta.contralateral ? cat(r, l) : r, lin = this.meta.contralateral ? cat(l, r) : l;
    this.action[0] = this.net.forward(rin)[0];
    this.action[1] = this.net.forward(lin)[0];
    return this.action;
  }

  // the command this step: the network on the current window, x the assist scale (0 when switched off)
  act() {
    this.forwardBlock(this.block());
    const k = this.enabled ? this.scale : 0;
    return [this.action[0] * k, this.action[1] * k];
  }
}
