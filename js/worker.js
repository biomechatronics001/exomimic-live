// The simulation, off the page's main thread: MuJoCo WASM + the QP + the policy networks step in real time here;
// the page only draws. Messages in: init / variant / clip / restart / step / set. Messages out: progress, ready, scene
// (geometry, once per model), state (poses, muscle paths, device torque arrows, new plot samples, stats), status, error.
import loadMujoco from "../vendor/mujoco/mujoco.js";
import { loadBlob } from "./blob.js";
import { loadModel, rebasedMeshVerts } from "./model.js";
import { HumanPolicy } from "./policy.js";
import { HumanSim, clipFromBlob, loadHumanAssets } from "./sim.js";

const MESH = 7, BOX = 6, PLANE = 0;
const S = {
  mujoco: null, policy: null, clips: [], clipCache: new Map(), variants: new Map(), sim: null, scene: null,
  clip: null, playing: true, speed: 1, deterministic: true, loop: true, ghost: false, muscles: true,
  assist: true, scale: 1, arrows: null, arrowsOn: true, pendingSteps: 0,
  simTime: 0, acc: 0, last: 0, fellAt: -1, stepMs: 0, rtf: 0, samples: [], timer: null,
};
let bytes = 0;
const post = (m, t) => self.postMessage(m, t || []);

// fetch with retries: a CDN right after a deploy answers some requests with 5xx or drops them for a few seconds;
// the final error names the file
async function get(p) {
  for (let k = 0; ; k++) {
    try {
      const r = await fetch(p);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r;
    } catch (e) {
      if (k >= 6 || String(e.message).startsWith("HTTP 4")) {
        throw new Error(`${p.replace(BASE, "")}: ${e.message} (after ${k + 1} tries)`);
      }
      console.warn(`retrying ${p}: ${e.message}`);
      await new Promise((res) => setTimeout(res, 300 * 2 ** k));
    }
  }
}
const BASE = new URL("../", import.meta.url).href;            // the site root
const io = {
  text: async (p) => (await get(BASE + p)).text(),
  bytes: async (p) => { const b = new Uint8Array(await (await get(BASE + p)).arrayBuffer()); bytes += b.byteLength; return b; },
};
const progress = (msg, frac) => post({ type: "progress", msg, frac, mb: bytes / 1e6 });

async function init() {
  progress("Loading the physics engine (MuJoCo WebAssembly)…", 0.05);
  const [mujoco, ort] = await Promise.all([
    loadMujoco(),
    import("../vendor/ort/ort.wasm.min.mjs").catch((e) => { console.warn("onnxruntime-web failed to load", e); return null; }),
  ]);
  S.mujoco = mujoco;
  if (ort) {
    ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
    ort.env.wasm.numThreads = 1;
  }
  progress("Loading the human model and its policy…", 0.15);
  const [index, , onnx, info] = await Promise.all([
    io.text("assets/clips/index.json").then(JSON.parse),
    loadVariant("human", (d, n) => progress(`Loading the musculoskeletal model (${d}/${n} files)…`, 0.15 + 0.45 * d / n)),
    ort ? io.bytes("assets/human/actor.onnx") : Promise.resolve(null),
    io.text("assets/human/actor_info.json").then(JSON.parse),
  ]);
  S.clips = index;
  progress("Preparing the policy network…", 0.8);
  S.policy = await HumanPolicy.create({ info, onnxBytes: onnx, ort });
  if (S.policy.backend !== "onnx") {
    progress("Loading the policy network (JavaScript fallback)…", 0.85);
    S.policy = await HumanPolicy.create({ info, blob: await loadBlob("assets/human/actor", io) });
  }
  await selectVariant("human");
  await selectClip(index[0].id);
  post({ type: "ready", clips: index, backend: S.policy.backend });
  S.last = performance.now();
  loop();
}

async function loadVariant(variant, onFile) {
  if (S.variants.has(variant)) return S.variants.get(variant);
  const assets = await loadHumanAssets(io, variant);
  const { model, geomFrame0 } = await loadModel(S.mujoco, `assets/model/${variant}`, io, onFile);
  const v = { variant, model, assets, geomFrame0 };
  S.variants.set(variant, v);
  return v;
}

async function selectVariant(variant) {
  const was = S.playing;
  S.playing = false;
  if (variant !== "human") post({ type: "status", msg: "Loading the device…" });
  const v = await loadVariant(variant, (d, n) => post({ type: "status", msg: `Loading the device model (${d}/${n})…` }));
  if (S.sim) { S.sim.data.delete(); S.ghostData.delete(); }
  S.sim = new HumanSim(S.mujoco, v.model, v.assets);
  S.ghostData = new S.mujoco.MjData(v.model);
  if (S.sim.exo) { S.sim.exo.enabled = S.assist; S.sim.exo.scale = S.scale; }
  S.variant = variant;
  S.scene = sceneOf(v.model, v.geomFrame0);
  S.scene.desc.arrows = S.arrows = arrowsOf(v.model, variant);
  musclesOf(v.model);
  hipMuscles();
  post({ type: "scene", variant, scene: S.scene.desc, exo: S.sim.exo ? S.sim.exo.meta.label : null },
    S.scene.transfer);
  if (S.clip) restart();
  post({ type: "status", msg: "" });
  S.playing = was;
}

// what the page draws: bone meshes (group 0), the stair / ramp boxes, the floor, the device's parts
function sceneOf(m, frame0) {
  const mj = S.mujoco, OBJ = mj.mjtObj, name = (t, i) => mj.mj_id2name(m, t.value, i) || "";
  const geoms = [], meshes = {}, transfer = [];
  for (let i = 0; i < m.ngeom; i++) {
    const type = m.geom_type[i], group = m.geom_group[i], gname = name(OBJ.mjOBJ_GEOM, i);
    const exo = gname.startsWith("exo") || name(OBJ.mjOBJ_BODY, m.geom_bodyid[i]).startsWith("exo");
    let kind = null;
    if (type === MESH && (group === 0 || exo)) kind = "mesh";
    else if (type === BOX && group === 0 && m.geom_bodyid[i] === 0) kind = "box";
    else if (type === PLANE) kind = "plane";
    if (!kind) continue;
    const g = { id: i, kind, exo, name: gname, size: Array.from(m.geom_size.subarray(3 * i, 3 * i + 3)) };
    if (kind === "mesh") {
      const id = m.geom_dataid[i];
      const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
      const own = rebasedMeshVerts(m, i, m.mesh_vert.subarray(3 * va, 3 * (va + vn)), frame0);
      const key = own ? `g${i}` : id;          // a rebased mesh is this geom's own copy
      g.mesh = key;
      if (!meshes[key]) {
        meshes[key] = { vert: own || Float32Array.from(m.mesh_vert.subarray(3 * va, 3 * (va + vn))),
          face: Uint32Array.from(m.mesh_face.subarray(3 * fa, 3 * (fa + fn))) };
        transfer.push(meshes[key].vert.buffer, meshes[key].face.buffer);
      }
    }
    geoms.push(g);
  }
  const ids = Int32Array.from(geoms.map((g) => g.id));
  const ghostIds = Int32Array.from(geoms.filter((g) => g.kind === "mesh" && !g.exo).map((g) => g.id));
  const pelvis = mj.mj_name2id(m, OBJ.mjOBJ_BODY.value, "pelvis");
  return { ids, ghostIds, pelvis, desc: { geoms, meshes, ghostCount: ghostIds.length, maxSeg: 2 * (m.nwrap + m.ntendon) }, transfer };
}

// the torque arrows: the joint each motor drives, in the order of the device torque [right, left], and how far
// outside the joint the arc is drawn (clear of the device's parts)
function arrowsOf(m, variant) {
  const qp = S.sim.qp;
  if (!S.sim.exo || !qp.nExo) return null;
  const mj = S.mujoco, ACT = mj.mjtObj.mjOBJ_ACTUATOR.value;
  const joints = Array.from(qp.exoNames, (n) => m.actuator_trnid[2 * mj.mj_name2id(m, ACT, n)]);
  const knee = variant === "knee";
  return { joints, offset: knee ? 0.1 : 0.13, radius: knee ? 0.08 : 0.11,
    max: S.sim.exo.meta.torque_limit || 15, names: ["R", "L"] };
}

// per arrow: centre (3), rotation axis (3, right-hand rule for a positive torque), start direction (3, along the
// limb the joint moves, in the plane of the arc), torque (N*m, + = the joint's positive direction = flexion)
function arrowState() {
  const d = S.sim.data, m = S.sim.model, a = S.arrows, tau = S.sim.exoTorque, out = new Float32Array(10 * a.joints.length);
  const p = 3 * S.scene.pelvis;
  a.joints.forEach((j, k) => {
    const c = d.xanchor.subarray(3 * j, 3 * j + 3), u = d.xaxis.subarray(3 * j, 3 * j + 3), b = 9 * m.jnt_bodyid[j];
    // outward: the side of the pelvis the joint is on, along its axis
    const side = Math.sign((c[0] - d.xpos[p]) * u[0] + (c[1] - d.xpos[p + 1]) * u[1] + (c[2] - d.xpos[p + 2]) * u[2]) || 1;
    // the limb: minus the moving body's y axis (OpenSim: y runs up the segment), without its part along the axis
    let s = [-d.xmat[b + 1], -d.xmat[b + 4], -d.xmat[b + 7]];
    const su = s[0] * u[0] + s[1] * u[1] + s[2] * u[2];
    s = [s[0] - su * u[0], s[1] - su * u[1], s[2] - su * u[2]];
    const sn = Math.hypot(s[0], s[1], s[2]) || 1;
    const o = 10 * k, off = side * a.offset;
    out.set([c[0] + off * u[0], c[1] + off * u[1], c[2] + off * u[2], u[0], u[1], u[2], s[0] / sn, s[1] / sn, s[2] / sn,
      tau[k]], o);
  });
  return out;
}

function musclesOf(m) {
  const mj = S.mujoco, TEN = mj.mjtTrn.mjTRN_TENDON.value, MUS = mj.mjtDyn.mjDYN_MUSCLE.value;
  S.tendonAct = new Int32Array(m.ntendon).fill(-1);
  for (let a = 0; a < m.nu; a++) {
    if (m.actuator_trntype[a] === TEN && m.actuator_dyntype[a] === MUS) S.tendonAct[m.actuator_trnid[2 * a]] = m.actuator_actadr[a];
  }
}

// the muscles acting on each hip's flexion (moment arm at the start pose), for the activation plot
function hipMuscles() {
  const m = S.sim.model, d = S.sim.data, mj = S.mujoco;
  mj.mj_forward(m, d);
  const jid = (n) => mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT.value, n);
  const dof = ["hip_flexion_r", "hip_flexion_l"].map((n) => m.jnt_dofadr[jid(n)]);
  const sets = [[], []], MUS = mj.mjtDyn.mjDYN_MUSCLE.value;
  for (let a = 0; a < m.nu; a++) {
    if (m.actuator_dyntype[a] !== MUS) continue;
    const adr = d.moment_rowadr[a], n = d.moment_rownnz[a];
    for (let t = 0; t < n; t++) {
      if (Math.abs(d.actuator_moment[adr + t]) < 1e-4) continue;
      const j = d.moment_colind[adr + t];
      if (j === dof[0]) sets[0].push(m.actuator_actadr[a]);
      if (j === dof[1]) sets[1].push(m.actuator_actadr[a]);
    }
  }
  S.hipAct = sets;
  S.qadr = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"].map((n) => m.jnt_qposadr[jid(n)]);
}

async function selectClip(id) {
  let clip = S.clipCache.get(id);
  if (!clip) {
    post({ type: "status", msg: "Loading the motion…" });
    clip = clipFromBlob(await loadBlob(`assets/clips/${id}`, io));
    S.clipCache.set(id, clip);
    post({ type: "status", msg: "" });
  }
  S.clip = clip;
  // where the clip starts (index.json "start"; run: a gait phase the policy enters smoothly, export_clips.DEMO_START)
  S.clipStart = (S.clips.find((c) => c.id === id) || {}).start || 0;
  restart();
}

function restart() {
  S.sim.setClip(S.clip);
  S.sim.reset(S.clipStart || 0);
  S.simTime = 0; S.acc = 0; S.fellAt = -1;
  S.samples.length = 0;
  post({ type: "restart" });
  sample();
  publish();
}

function sample() {
  const d = S.sim.data, q = S.qadr, deg = 180 / Math.PI;
  const mean = (ids) => ids.reduce((s, i) => s + Math.min(Math.max(d.act[i], 0), 1), 0) / Math.max(ids.length, 1);
  const [er, el] = S.sim.exoTorque;
  S.samples.push(S.simTime, d.qpos[q[0]] * deg, d.qpos[q[1]] * deg, d.qpos[q[2]] * deg, d.qpos[q[3]] * deg,
    er, el, 100 * mean(S.hipAct[0]), 100 * mean(S.hipAct[1]));
}

// poses of the drawn geoms (xpos 3 + xmat 9 each), the ghost's, the muscle paths, the new plot samples
function publish() {
  const d = S.sim.data, sc = S.scene, n = sc.ids.length;
  const pose = new Float32Array(12 * n);
  for (let k = 0; k < n; k++) {
    const i = sc.ids[k];
    pose.set(d.geom_xpos.subarray(3 * i, 3 * i + 3), 12 * k);
    pose.set(d.geom_xmat.subarray(9 * i, 9 * i + 9), 12 * k + 3);
  }
  let ghost = null;
  if (S.ghost) {
    const g = S.ghostData, c = S.sim.st.clip, t = S.sim.st.t, gi = sc.ghostIds;
    g.qpos.set(c.qpos.subarray(t * c.nq, (t + 1) * c.nq));
    S.mujoco.mj_kinematics(S.sim.model, g);
    ghost = new Float32Array(12 * gi.length);
    for (let k = 0; k < gi.length; k++) {
      ghost.set(g.geom_xpos.subarray(3 * gi[k], 3 * gi[k] + 3), 12 * k);
      ghost.set(g.geom_xmat.subarray(9 * gi[k], 9 * gi[k] + 9), 12 * k + 3);
    }
  }
  let seg = null, segAct = null, nseg = 0;
  if (S.muscles) {
    const m = S.sim.model, wx = d.wrap_xpos, wo = d.wrap_obj, adr = d.ten_wrapadr, num = d.ten_wrapnum, act = d.act;
    seg = new Float32Array(sc.desc.maxSeg * 3);
    segAct = new Float32Array(sc.desc.maxSeg);
    for (let t = 0; t < m.ntendon; t++) {
      const ai = S.tendonAct[t], a = ai >= 0 ? act[ai] : 0;
      for (let j = adr[t]; j < adr[t] + num[t] - 1; j++) {
        if (wo[j] === -2 || wo[j + 1] === -2) continue;          // pulley: the path restarts
        for (let e = 0; e < 2; e++) {
          seg.set(wx.subarray(3 * (j + e), 3 * (j + e) + 3), 3 * nseg);
          segAct[nseg++] = a;
        }
      }
    }
  }
  // the body's forward direction, for the camera presets: the pelvis' x axis in the horizontal plane (OpenSim
  // frame: x forward, y up, z right; checked against the reference walking direction)
  const p = 3 * sc.pelvis, R = 9 * sc.pelvis;
  let fx = d.xmat[R], fy = d.xmat[R + 3];
  const fn = Math.hypot(fx, fy) || 1; fx /= fn; fy /= fn;
  const samples = Float32Array.from(S.samples);
  S.samples.length = 0;
  const arrows = S.arrows && S.arrowsOn ? arrowState() : null;
  const tr = [pose.buffer, samples.buffer];
  if (ghost) tr.push(ghost.buffer);
  if (seg) tr.push(seg.buffer, segAct.buffer);
  if (arrows) tr.push(arrows.buffer);
  post({ type: "state", pose, ghost, seg, segAct, nseg, samples, arrows, pelvis: [d.xpos[p], d.xpos[p + 1], d.xpos[p + 2]], fwd: [fx, fy],
    stats: { t: S.simTime, rtf: S.rtf, stepMs: S.stepMs, playing: S.playing, backend: S.policy.backend } }, tr);
}

async function stepOnce() {
  const sim = S.sim, t0 = performance.now();
  sim.step(await S.policy.act(sim.obs, S.deterministic));
  S.stepMs = 0.95 * S.stepMs + 0.05 * (performance.now() - t0);
  S.simTime += sim.dt;
  sample();
  if (sim.fallen) {
    S.fellAt = performance.now();
    post({ type: "status", msg: "Lost the reference (tracking error over 0.5 m): restarting the motion." });
    return false;
  }
  if (sim.clipEnded) {
    if (S.loop) restart();
    else { S.playing = false; post({ type: "status", msg: "End of the motion." }); }
    return false;
  }
  return true;
}

// real-time stepping: catch up with the wall clock (x speed), publish a frame about every 16 ms
async function loop() {
  const now = performance.now(), dtWall = Math.min((now - S.last) / 1000, 0.25);
  S.last = now;
  if (S.fellAt >= 0 && now - S.fellAt > 900) { restart(); post({ type: "status", msg: "" }); }
  if (S.playing && S.fellAt < 0) {
    const dt = S.sim.dt, before = S.simTime;
    S.acc = Math.min(S.acc + dtWall * S.speed, 6 * dt);
    const t0 = performance.now();
    while (S.acc >= dt && performance.now() - t0 < 14) {
      S.acc -= dt;
      if (!(await stepOnce())) { S.acc = 0; break; }
    }
    if (S.acc >= dt) S.acc = dt;                  // could not keep up: run slower than real time
    if (dtWall > 0) S.rtf = 0.9 * S.rtf + 0.1 * Math.max(0, S.simTime - before) / dtWall;
  } else {
    S.rtf *= 0.9;
    // single steps while paused, run here so that only this loop ever steps the simulation
    if (S.pendingSteps > 0 && S.fellAt < 0) {
      S.pendingSteps--;
      await stepOnce();
    }
  }
  publish();
  const spent = performance.now() - now;
  S.timer = setTimeout(loop, Math.max(0, 16 - spent));
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") await init();
    else if (m.type === "variant") await selectVariant(m.variant);
    else if (m.type === "clip") await selectClip(m.id);
    else if (m.type === "restart") { restart(); post({ type: "status", msg: "" }); }
    else if (m.type === "step") { S.playing = false; S.pendingSteps = Math.min(S.pendingSteps + 1, 50); }
    else if (m.type === "set") {
      for (const k of ["playing", "speed", "deterministic", "loop", "ghost", "muscles", "assist", "scale", "arrowsOn"]) if (k in m) S[k] = m[k];
      if (m.playing) S.pendingSteps = 0;
      if (S.sim && S.sim.exo) { S.sim.exo.enabled = S.assist; S.sim.exo.scale = S.scale; }
      if ("playing" in m) S.last = performance.now();
    }
  } catch (err) {
    console.error(err);
    post({ type: "error", msg: err.message || String(err) });
  }
};
