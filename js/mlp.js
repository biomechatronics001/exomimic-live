// The actors in plain JavaScript (float32 weights, float64 accumulation per output): the reference the ONNX
// session is checked against, and the fallback when onnxruntime-web is unavailable.
//   ResidualActor: musclemimic/algorithms/common/networks.py ActorCritic(use_residual) actor tower with its
//                  RunningMeanStd in front (frozen statistics).

function silu(v) { return v / (1 + Math.exp(-v)); }

class Dense {
  // flax kernel (in, out) -> row-major (out, in) for a contiguous dot product per output
  constructor(kernel, bias) {
    const [nin, nout] = kernel.shape;
    this.nin = nin; this.nout = nout;
    this.W = new Float32Array(nin * nout);
    for (let i = 0; i < nin; i++) for (let o = 0; o < nout; o++) this.W[o * nin + i] = kernel[i * nout + o];
    this.b = Float32Array.from(bias);
    this.y = new Float64Array(nout);
  }

  apply(x) {
    const { W, b, y, nin, nout } = this;
    for (let o = 0; o < nout; o++) {
      const r = o * nin;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
      for (; i + 3 < nin; i += 4) {
        s0 += W[r + i] * x[i]; s1 += W[r + i + 1] * x[i + 1]; s2 += W[r + i + 2] * x[i + 2]; s3 += W[r + i + 3] * x[i + 3];
      }
      for (; i < nin; i++) s0 += W[r + i] * x[i];
      y[o] = b[o] + s0 + s1 + s2 + s3;
    }
    return y;
  }
}

// flax LayerNorm (biased variance), in place
function layerNorm(x, scale, bias, eps) {
  const n = x.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) * (x[i] - m);
  const inv = 1 / Math.sqrt(v / n + eps);
  for (let i = 0; i < n; i++) x[i] = (x[i] - m) * inv * scale[i] + bias[i];
  return x;
}

export class ResidualActor {
  constructor(blob) {
    const m = blob.meta;
    if (m.arch !== "residual_gated" || m.activation !== "silu") throw new Error(`unsupported actor ${m.arch}/${m.activation}`);
    this.eps = m.layernorm_eps;
    this.obsDim = m.obs_dim; this.actionDim = m.action_dim;
    this.mean = Float64Array.from(blob.obs_mean);
    this.invStd = Float64Array.from(blob.obs_var, (v) => 1 / Math.sqrt(v + m.obs_eps));
    this.logStd = Float64Array.from(blob.log_std);
    const dl = (name) => ({ d: new Dense(blob[`${name}_dense.kernel`], blob[`${name}_dense.bias`]),
      s: blob[`${name}_ln.scale`], b: blob[`${name}_ln.bias`] });
    this.blocks = m.blocks.map((blk) => ({
      l0: dl(blk.layer0), l1: dl(blk.layer1),
      proj: blk.proj ? new Dense(blob[`${blk.proj}.kernel`], blob[`${blk.proj}.bias`]) : null,
      w: 1 / (1 + Math.exp(-blob[blk.gate][0])),
    }));
    this.tail = m.tail ? dl(m.tail) : null;
    this.out = new Dense(blob["output.kernel"], blob["output.bias"]);
    this.x = new Float64Array(this.obsDim);
  }

  // obs (length obsDim) -> mean action (Float64Array, length actionDim; not clipped, the QP clips)
  forward(obs) {
    let x = this.x;
    for (let i = 0; i < this.obsDim; i++) x[i] = (obs[i] - this.mean[i]) * this.invStd[i];
    for (const blk of this.blocks) {
      const h0 = Float64Array.from(blk.l0.d.apply(x));
      layerNorm(h0, blk.l0.s, blk.l0.b, this.eps);
      for (let i = 0; i < h0.length; i++) h0[i] = silu(h0[i]);
      const h = Float64Array.from(blk.l1.d.apply(h0));
      layerNorm(h, blk.l1.s, blk.l1.b, this.eps);
      const skip = blk.proj ? blk.proj.apply(x) : x;
      const nx = new Float64Array(h.length);
      for (let i = 0; i < h.length; i++) nx[i] = silu(skip[i] + blk.w * h[i]);
      x = nx;
    }
    if (this.tail) {
      const t = Float64Array.from(this.tail.d.apply(x));
      layerNorm(t, this.tail.s, this.tail.b, this.eps);
      for (let i = 0; i < t.length; i++) t[i] = silu(t[i]);
      x = t;
    }
    return Float64Array.from(this.out.apply(x));
  }
}
