// Load an exported model (export/export_model.py) into MuJoCo WASM: every file of the model directory goes into a
// virtual file system, model.xml is compiled, and the exact float arrays of the training model are written back
// (model_patch: the XML writer rounds numbers to ~1e-7).
import { loadBlob } from "./blob.js";

// files.json maps every file the XML names to its URL relative to the model directory (asset files are shared by
// the variants under assets/model/shared). `onFile(done, total)` reports download progress. `raw` skips
// site_override.json (export/sidesite_compat.mjs, which computes it).
export async function loadModel(mujoco, dir, io, onFile = null, { raw = false } = {}) {
  const files = JSON.parse(await io.text(dir + "/files.json"));
  const vfs = new mujoco.MjVFS();
  const names = Object.keys(files);
  let done = 0, next = 0;
  const worker = async () => {                    // at most 8 requests in flight
    while (next < names.length) {
      const n = names[next++];
      const b = await io.bytes(dir + "/" + files[n]);
      vfs.addBuffer(n, b);
      if (onFile) onFile(++done, names.length);
    }
  };
  await Promise.all(Array.from({ length: Math.min(8, names.length) }, worker));
  const model = mujoco.MjModel.from_xml_path("model.xml", vfs);
  vfs.delete();
  // the geom frames this compile computed for its own mesh vertices (MuJoCo recentres every mesh on its centroid and
  // principal axes): the patch below writes back the training model's frames, which belong to the full-resolution
  // meshes; a decimated mesh is drawn with these instead (worker.js sceneOf)
  const geomFrame0 = { pos: Float64Array.from(model.geom_pos), quat: Float64Array.from(model.geom_quat) };
  const patch = await loadBlob(dir + "/model_patch", io);
  const applied = [];
  for (const [k, v] of Object.entries(patch)) {
    if (k === "meta") continue;
    const dst = model[k];
    if (!dst || dst.length !== v.length) { console.warn(`model_patch: ${k} not applied (size)`); continue; }
    dst.set(v);
    applied.push(k);
  }
  // MuJoCo 3.17+ picks a sidesite's wrapping side differently from 3.4 (the training model); these sidesites are
  // moved so every tendon keeps the training model's path. Required: without it TMAJ / LAT2 wrap the other way.
  if (!raw) {
    const ov = JSON.parse(await io.text(dir + "/site_override.json"));
    for (const [n, { pos }] of Object.entries(ov.sites)) {
      const i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, n);
      if (i < 0) throw new Error(`site_override.json: no site ${n}`);
      model.site_pos.set(pos, 3 * i);
    }
  }
  return { model, applied, geomFrame0 };
}

// row-major rotation matrix of a MuJoCo quaternion (w, x, y, z)
function quatMat(q, k) {
  const w = q[4 * k], x = q[4 * k + 1], y = q[4 * k + 2], z = q[4 * k + 3];
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}

// vertices of mesh geom i in the frame its (patched) geom_pos / geom_quat place: the compile recentred the mesh for
// frame0, the patch restored the full-resolution mesh's frame; v' = Rp^T (R0 v + p0 - pp). Null when the two agree
// (every mesh that was not decimated: the patch only undid the XML's rounding there).
export function rebasedMeshVerts(m, i, vert, frame0) {
  const p0 = frame0.pos.subarray(3 * i, 3 * i + 3), pp = m.geom_pos.subarray(3 * i, 3 * i + 3);
  const R0 = quatMat(frame0.quat, i), Rp = quatMat(m.geom_quat, i);
  let dev = Math.hypot(p0[0] - pp[0], p0[1] - pp[1], p0[2] - pp[2]);
  for (let k = 0; k < 9; k++) dev = Math.max(dev, Math.abs(R0[k] - Rp[k]));
  if (dev < 1e-6) return null;
  const out = new Float32Array(vert.length);
  for (let v = 0; v < vert.length; v += 3) {
    const x = vert[v], y = vert[v + 1], z = vert[v + 2];
    const b0 = R0[0] * x + R0[1] * y + R0[2] * z + p0[0] - pp[0];      // body frame, minus the patched origin
    const b1 = R0[3] * x + R0[4] * y + R0[5] * z + p0[1] - pp[1];
    const b2 = R0[6] * x + R0[7] * y + R0[8] * z + p0[2] - pp[2];
    out[v] = Rp[0] * b0 + Rp[3] * b1 + Rp[6] * b2;
    out[v + 1] = Rp[1] * b0 + Rp[4] * b1 + Rp[7] * b2;
    out[v + 2] = Rp[2] * b0 + Rp[5] * b1 + Rp[8] * b2;
  }
  return out;
}
