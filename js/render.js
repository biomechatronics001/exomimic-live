// three.js view of the simulation the worker runs: the scene description (meshes, boxes, the floor, the device's
// parts) arrives once per model; each frame brings the poses (xpos + row-major xmat per drawn geom), the muscle
// paths with their activations, the device torque arrows, and optionally the reference "ghost".
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

// muscle colour by activation: [idle, mid, full] (linear RGB); full at MUSCLE_FULL (index.html's legend says 35 %)
export const MUSCLE_FULL = 0.35;
const MUSCLE_WIDTH = 2;            // CSS pixels
const MUSCLE_RAMP_LIGHT = [[0.74, 0.76, 0.79], [0.99, 0.55, 0.06], [0.84, 0.04, 0.06]];
const MUSCLE_RAMP_DARK = [[0.24, 0.26, 0.30], [1.0, 0.56, 0.08], [1.0, 0.16, 0.12]];

// A device torque as a curved arrow around the joint axis: the arc starts along the limb and sweeps the way the
// torque turns it (right-hand rule about the axis), its length is |torque| / max of SWEEP, the cone marks the
// direction, the label gives the value.
const SWEEP = (300 * Math.PI) / 180, N_ARC = 48, N_RING = 10;

class TorqueArrow {
  // below: the label goes under the arc instead of above it (the two sides' arcs line up in a side view)
  constructor(color, name, radius, below) {
    this.name = name;
    this.radius = radius;
    this.below = below;
    this.tube = 0.009;
    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, emissive: color, emissiveIntensity: 0.35 });
    const n = (N_ARC + 1) * N_RING;
    this.pos = new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage);
    this.nrm = new THREE.BufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage);
    const idx = [];
    for (let i = 0; i < N_ARC; i++) {
      for (let j = 0; j < N_RING; j++) {
        const a = i * N_RING + j, b = i * N_RING + (j + 1) % N_RING, c = a + N_RING, e = b + N_RING;
        idx.push(a, c, b, b, c, e);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", this.pos);
    g.setAttribute("normal", this.nrm);
    g.setIndex(idx);
    this.arc = new THREE.Mesh(g, mat);
    this.arc.frustumCulled = false;
    this.headLen = 0.045;
    this.head = new THREE.Mesh(new THREE.ConeGeometry(0.024, this.headLen, 20), mat);
    this.label = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, transparent: true }));
    this.label.renderOrder = 10;
    this.labelCanvas = document.createElement("canvas");
    this.labelCanvas.width = 256; this.labelCanvas.height = 72;
    this.label.material.map = new THREE.CanvasTexture(this.labelCanvas);
    this.label.material.map.colorSpace = THREE.SRGBColorSpace;
    this.label.scale.set(0.2, 0.056, 1);
    this.color = `#${new THREE.Color(color).getHexString()}`;
    this.text = "";
    this.group.add(this.arc, this.head, this.label);
  }

  // A: the worker's 10 numbers for this arrow (centre, axis, start, torque); max: the torque of a full SWEEP
  set(A, o, max, dark) {
    const tau = A[o + 9], mag = Math.abs(tau);
    this.group.visible = mag >= 0.2;
    if (!this.group.visible) return;
    // frame: x along the limb, z the rotation axis of this torque's sign, y = z x x
    const x = [A[o + 6], A[o + 7], A[o + 8]], sg = tau >= 0 ? 1 : -1, z = [sg * A[o + 3], sg * A[o + 4], sg * A[o + 5]];
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    const e = this.group.matrix.elements;
    e[0] = x[0]; e[1] = x[1]; e[2] = x[2]; e[3] = 0;
    e[4] = y[0]; e[5] = y[1]; e[6] = y[2]; e[7] = 0;
    e[8] = z[0]; e[9] = z[1]; e[10] = z[2]; e[11] = 0;
    e[12] = A[o]; e[13] = A[o + 1]; e[14] = A[o + 2]; e[15] = 1;
    this.group.matrixWorldNeedsUpdate = true;
    // the arc, from the limb to where the cone starts
    const R = this.radius, head = this.headLen / R;
    const theta = Math.max(Math.min(mag / max, 1) * SWEEP, head + 0.25), end = theta - head;   // a stub at least
    const P = this.pos.array, Nn = this.nrm.array;
    for (let i = 0; i <= N_ARC; i++) {
      const t = (end * i) / N_ARC, ct = Math.cos(t), st = Math.sin(t);
      for (let j = 0; j < N_RING; j++) {
        const f = (2 * Math.PI * j) / N_RING, cf = Math.cos(f), sf = Math.sin(f), k = 3 * (i * N_RING + j);
        Nn[k] = cf * ct; Nn[k + 1] = cf * st; Nn[k + 2] = sf;
        P[k] = (R + this.tube * cf) * ct; P[k + 1] = (R + this.tube * cf) * st; P[k + 2] = this.tube * sf;
      }
    }
    this.pos.needsUpdate = this.nrm.needsUpdate = true;
    // the cone along the tangent at the arc's end (ConeGeometry points along +y; rotating +y by t about z gives the
    // tangent (-sin t, cos t))
    const ce = Math.cos(end), se = Math.sin(end), h = this.headLen / 2;
    this.head.position.set(R * ce - se * h, R * se + ce * h, 0);
    this.head.rotation.set(0, 0, end);
    // the value, straight above (right) or below (left) the arc: world up in this frame is (x_z, y_z, z_z)
    const lr = (this.below ? -1 : 1) * (R + 0.075);
    this.label.position.set(lr * x[2], lr * y[2], lr * z[2]);
    const text = `${this.name} ${mag.toFixed(1)} N·m ${tau >= 0 ? "flex" : "ext"}`;
    if (text !== this.text || dark !== this.dark) {
      this.text = text; this.dark = dark;
      const g = this.labelCanvas.getContext("2d"), W = this.labelCanvas.width, H = this.labelCanvas.height;
      g.clearRect(0, 0, W, H);
      g.fillStyle = dark ? "rgba(20,23,28,0.82)" : "rgba(255,255,255,0.85)";
      g.beginPath(); g.roundRect(2, 2, W - 4, H - 4, 14); g.fill();
      g.fillStyle = this.color;
      g.font = "600 36px system-ui, -apple-system, Segoe UI, sans-serif";
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(text, W / 2, H / 2 + 1);
      this.label.material.map.needsUpdate = true;
    }
  }
}

export class Viewer {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 200);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0, 0.9);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.follow = true;
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.7);
    this.scene.add(this.hemi);
    const sun = this.sun = new THREE.DirectionalLight(0xffffff, 2.1);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 0.5, far: 20 });
    sun.shadow.bias = -0.0004;
    this.scene.add(sun, sun.target);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.pelvis = [0, 0, 0.9];
    this.fwd = [1, 0];
    this.camPreset = "persp";
    this.setTheme(false);
    window.addEventListener("resize", () => this.resize());
    this.resize();
  }

  setTheme(dark) {
    this.dark = dark;
    this.scene.background = new THREE.Color(dark ? 0x14171c : 0xeef1f5);
    this.scene.fog = new THREE.Fog(this.scene.background, 12, 36);
    if (this.floorTex) { this.floorTex.image = this._floorCanvas(dark); this.floorTex.needsUpdate = true; }
    this.hemi.groundColor.set(dark ? 0x30343c : 0x8a8f99);
  }

  resize() {
    const c = this.renderer.domElement, w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.muscleMat) this.muscleMat.resolution.set(w, h);        // line width in CSS pixels
  }

  _floorCanvas(dark) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 256;
    const g = cv.getContext("2d");
    g.fillStyle = dark ? "#22262d" : "#e2e6eb";
    g.fillRect(0, 0, 256, 256);
    g.fillStyle = dark ? "#292e36" : "#d5dae0";
    g.fillRect(0, 0, 128, 128); g.fillRect(128, 128, 128, 128);
    return cv;
  }

  build(desc) {
    this.root.clear();
    const bone = new THREE.MeshStandardMaterial({ color: 0xe9e2d0, roughness: 0.62 });
    const stair = new THREE.MeshStandardMaterial({ color: 0xa9aeb6, roughness: 0.9 });
    const exo = new THREE.MeshStandardMaterial({ color: 0x2f3742, roughness: 0.35, metalness: 0.55 });
    // neutral steel: blue / orange mean right / left (plots, torque arrows), so the device itself stays grey
    const exoAccent = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.38, metalness: 0.45 });
    const geo = {};
    for (const [id, m] of Object.entries(desc.meshes)) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(m.vert, 3));
      g.setIndex(new THREE.BufferAttribute(m.face, 1));
      g.computeVertexNormals();
      geo[id] = g;
    }
    this.objs = [];
    for (const d of desc.geoms) {
      let o;
      if (d.kind === "mesh") {
        o = new THREE.Mesh(geo[d.mesh], d.exo ? (/base|link/i.test(d.name) ? exoAccent : exo) : bone);
        o.castShadow = true;
      } else if (d.kind === "box") {
        o = new THREE.Mesh(new THREE.BoxGeometry(2 * d.size[0], 2 * d.size[1], 2 * d.size[2]), stair);
        o.castShadow = o.receiveShadow = true;
      } else {
        const tex = this.floorTex = new THREE.CanvasTexture(this._floorCanvas(this.dark));
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(60, 60);
        tex.anisotropy = 8;
        tex.colorSpace = THREE.SRGBColorSpace;
        o = new THREE.Mesh(new THREE.PlaneGeometry(120, 120), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
        o.receiveShadow = true;
      }
      o.matrixAutoUpdate = false;
      this.root.add(o);
      this.objs.push(o);
    }
    // ghost: the bone meshes again, translucent
    const gm = new THREE.MeshStandardMaterial({ color: 0x3d8bfd, transparent: true, opacity: 0.22, depthWrite: false });
    this.ghost = new THREE.Group();
    this.ghostObjs = [];
    for (const d of desc.geoms) {
      if (d.kind !== "mesh" || d.exo) continue;
      const o = new THREE.Mesh(geo[d.mesh], gm);
      o.matrixAutoUpdate = false;
      o.renderOrder = 2;
      this.ghost.add(o);
      this.ghostObjs.push(o);
    }
    this.ghost.visible = false;
    this.root.add(this.ghost);
    // muscles: line segments MUSCLE_WIDTH px wide (screen-space quads; plain GL lines stay 1 px), colour from
    // activation. The interleaved buffers hold [start xyz, end xyz] per segment, as the worker sends them.
    const n = desc.maxSeg;                                  // vertices, two per segment
    const lg = new LineSegmentsGeometry();
    lg.setPositions(new Float32Array(3 * n));
    lg.setColors(new Float32Array(3 * n));
    this.segPos = lg.attributes.instanceStart.data;
    this.segCol = lg.attributes.instanceColorStart.data;
    this.muscleMat = new LineMaterial({ vertexColors: true, linewidth: MUSCLE_WIDTH });
    this.muscles = new LineSegments2(lg, this.muscleMat);
    this.muscles.frustumCulled = false;
    this.root.add(this.muscles);
    this.resize();
    // device torque arrows, right / left in the plots' colours
    this.arrows = null;
    if (desc.arrows) {
      const a = desc.arrows;
      this.arrowMax = a.max;
      this.arrows = a.names.map((nm, k) => new TorqueArrow(k === 0 ? 0x0b6fd1 : 0xd26b00, nm, a.radius, k === 1));
      for (const t of this.arrows) this.root.add(t.group);
    }
  }

  static _pose(obj, P, k) {
    const e = obj.matrix.elements, b = 12 * k;
    // three.js is column-major; MuJoCo xmat is row-major
    e[0] = P[b + 3]; e[4] = P[b + 4]; e[8] = P[b + 5]; e[12] = P[b];
    e[1] = P[b + 6]; e[5] = P[b + 7]; e[9] = P[b + 8]; e[13] = P[b + 1];
    e[2] = P[b + 9]; e[6] = P[b + 10]; e[10] = P[b + 11]; e[14] = P[b + 2];
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;
    obj.matrixWorldNeedsUpdate = true;
  }

  update(s) {
    if (!this.objs) return;
    for (let k = 0; k < this.objs.length; k++) Viewer._pose(this.objs[k], s.pose, k);
    this.ghost.visible = !!s.ghost;
    if (s.ghost) for (let k = 0; k < this.ghostObjs.length; k++) Viewer._pose(this.ghostObjs[k], s.ghost, k);
    this.muscles.visible = !!s.seg;
    if (s.seg) {
      const P = this.segPos.array, C = this.segCol.array;
      P.set(s.seg.subarray(0, 3 * s.nseg));
      // idle muscles neutral grey (recede), active ones orange -> red (MUSCLE_FULL activation and above)
      const [rest, mid, hot] = this.dark ? MUSCLE_RAMP_DARK : MUSCLE_RAMP_LIGHT;
      for (let k = 0; k < s.nseg; k++) {
        const a = Math.min(1, Math.max(s.segAct[k], 0) / MUSCLE_FULL) ** 0.75;
        const [c0, c1, t] = a < 0.5 ? [rest, mid, 2 * a] : [mid, hot, 2 * a - 1];
        C[3 * k] = c0[0] + (c1[0] - c0[0]) * t;
        C[3 * k + 1] = c0[1] + (c1[1] - c0[1]) * t;
        C[3 * k + 2] = c0[2] + (c1[2] - c0[2]) * t;
      }
      this.muscles.geometry.instanceCount = s.nseg / 2;
      this.segPos.needsUpdate = this.segCol.needsUpdate = true;
    }
    if (this.arrows) {
      this.arrows.forEach((t, k) => {
        if (s.arrows) t.set(s.arrows, 10 * k, this.arrowMax, this.dark);
        else t.group.visible = false;
      });
    }
    const [x, y, z] = s.pelvis;
    if (this.follow) {        // track the pelvis; height smoothed (stairs, squats) so the view does not bob per step
      const t = this.controls.target, dx = x - t.x, dy = y - t.y, dz = 0.08 * (z - 0.08 - t.z);
      t.x = x; t.y = y; t.z += dz;
      this.camera.position.x += dx; this.camera.position.y += dy; this.camera.position.z += dz;
    }
    this.pelvis = s.pelvis;
    this.fwd = s.fwd;
    this.sun.position.set(x + 2.5, y - 3.5, 6);
    this.sun.target.position.set(x, y, 0);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // camera presets around the pelvis, relative to the body's heading (`fwd`: the pelvis' x axis, from the worker)
  preset(kind) {
    this.camPreset = kind;
    const t = this.controls.target, c = this.camera.position;
    const [fx, fy] = this.fwd, lx = -fy, ly = fx;                    // forward, the body's left
    const v = { persp: [2.4, -2.2, 0.8], side: [0, -3.6, 0.3], front: [3.6, 0, 0.3], top: [0.01, 0, 5.5] }[kind];
    // v = [forward, left, up] offsets; "side" looks at the right side
    c.set(t.x + v[0] * fx + v[1] * lx, t.y + v[0] * fy + v[1] * ly, t.z + v[2]);
  }
}
