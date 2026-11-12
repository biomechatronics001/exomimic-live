// The human actor: onnxruntime-web when it loads (about 5x faster), else the plain-JS network (mlp.js).
import { ResidualActor } from "./mlp.js";

export class HumanPolicy {
  // info: actor_info.json (obs_dim, log_std); blob: actor.{json,bin} for the plain-JS network (null with ONNX)
  static async create({ info, blob = null, onnxBytes = null, ort = null }) {
    const p = new HumanPolicy();
    p.obsDim = info.obs_dim;
    p.logStd = Float64Array.from(info.log_std);
    p.backend = "none";
    if (blob) { p.js = new ResidualActor(blob); p.backend = "js"; }
    if (ort && onnxBytes) {
      try {
        p.ort = ort;
        p.sess = await ort.InferenceSession.create(onnxBytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
        p.buf = new Float32Array(p.obsDim);
        p.backend = "onnx";
      } catch (e) {
        console.warn("onnxruntime unavailable:", e);
      }
    }
    return p;
  }

  // mean action (deterministic = the policy mean) or a sample of the policy's Gaussian
  async act(obs, deterministic = true, rand = Math.random) {
    let mean;
    if (this.backend === "onnx") {
      this.buf.set(obs);
      const out = await this.sess.run({ obs: new this.ort.Tensor("float32", this.buf, [1, this.obsDim]) });
      mean = Float64Array.from(out.action.data);
    } else {
      mean = this.js.forward(obs);
    }
    if (!deterministic) {
      for (let i = 0; i < mean.length; i++) mean[i] += Math.exp(this.logStd[i]) * gauss(rand);
    }
    return mean;
  }
}

function gauss(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
