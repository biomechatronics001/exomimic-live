// One simulated human, stepped exactly as the CPU env steps it (loco_mujoco/core/mujoco_base.py Mujoco.step):
//   last_action <- action; ctrl <- TorqueQP(action, data as left by the previous step); mj_step x n_substeps;
//   reference frame + 1; observation; terminal check.
import { loadBlob } from "./blob.js";
import { ExoController } from "./exo.js";
import { HumanObs } from "./obs.js";
import { TorqueQP } from "./qp.js";
import { CanonicalTerrain } from "./terrain.js";

export function clipFromBlob(b) {
  const [T, nq] = b.qpos.shape;
  return { id: b.meta.id, label: b.meta.label, length: T, nq, nv: b.qvel.shape[1], qpos: b.qpos, qvel: b.qvel,
    site_rpos: b.site_rpos, root_xpos: b.root_xpos, meta: b.meta };
}

// the terrain and the model-specific layout / QP of a variant ("human", "hip", "knee"); the exo variants add their
// controller. `withActor` also loads the human network's weights (the plain-JS network; tests, fallback).
export async function loadHumanAssets(io, variant = "human", withActor = false) {
  const base = "assets/" + variant;
  const [layout, terrain, qp, actor, exo] = await Promise.all([
    io.text(base + "/obs_layout.json").then(JSON.parse), io.text("assets/human/terrain.json").then(JSON.parse),
    loadBlob(base + "/qp", io), withActor ? loadBlob("assets/human/actor", io) : null,
    variant === "human" ? null : loadBlob(base + "/exo_actor", io)]);
  return { variant, layout, terrain, qp, actor, exo };
}

export class HumanSim {
  constructor(mujoco, model, assets) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.layout = assets.layout;
    this.nSub = assets.layout.n_substeps;
    this.dt = assets.layout.control_dt;
    this.terrain = new CanonicalTerrain(assets.terrain);
    this.obsB = new HumanObs(assets.layout, this.terrain);
    this.qp = new TorqueQP(assets.qp);
    this.nAct = this.qp.ni;
    this.exo = assets.exo ? new ExoController(assets.exo) : null;
    this.fullAction = new Float64Array(this.nAct + (this.exo ? 2 : 0));
    this.st = { clip: null, t: 0, lastAction: new Float64Array(this.nAct), achieved: new Float64Array(this.nAct) };
    this.obs = null;
    this.fallen = false;
    this.term = null;
  }

  setClip(clip) { this.st.clip = clip; }

  // TrajInitialStateHandler: the reference state of `frame` (world frame, rezero_root_xy false), forward kinematics
  reset(frame = 0) {
    const { mujoco, model, data, st } = this;
    const clip = st.clip;
    mujoco.mj_resetData(model, data);
    data.qpos.set(clip.qpos.subarray(frame * clip.nq, (frame + 1) * clip.nq));
    data.qvel.set(clip.qvel.subarray(frame * clip.nv, (frame + 1) * clip.nv));
    mujoco.mj_forward(model, data);
    st.t = frame;
    st.lastAction.fill(0);
    st.achieved.fill(0);
    this.qp.reset();
    this.fallen = false;
    this.obs = this.obsB.build(data, st);
    this.term = this.obsB.terminal(data, st);
    if (this.exo) this.exo.reset(this.obs);
    return this.obs;
  }

  // one control step for the human's `action` (31 torques); with a device its controller adds the motor command
  // from its sensor window (`exoAction` overrides it, for the parity tests)
  step(action, exoAction = null) {
    const { mujoco, model, data, st } = this;
    const full = this.fullAction;
    for (let i = 0; i < this.nAct; i++) full[i] = action[i];
    if (this.exo) {
      const e = exoAction || this.exo.act();
      full[this.nAct] = e[0]; full[this.nAct + 1] = e[1];
    }
    st.lastAction.set(full.subarray(0, this.nAct));
    this.qp.solve(full, data);
    this.qp.apply(data);
    for (let s = 0; s < this.nSub; s++) mujoco.mj_step(model, data);
    st.achieved.set(this.qp.achieved);
    st.t = Math.min(st.t + 1, st.clip.length - 1);
    this.obs = this.obsB.build(data, st);
    this.term = this.obsB.terminal(data, st);
    this.fallen = this.term.fallen;
    if (this.exo) this.exo.push(this.obs);
    return this.obs;
  }

  // the motors' torque this step (N*m, after the output filter), [right, left]
  get exoTorque() { return this.exo ? [this.qp.exoOut[0], this.qp.exoOut[1]] : [0, 0]; }

  get clipEnded() { return this.st.t >= this.st.clip.length - 1; }
}
