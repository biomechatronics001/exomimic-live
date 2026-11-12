// The page: controls, plots and the 3D view. The simulation runs in a Web Worker (worker.js); this thread only
// sends the user's settings and draws the latest state the worker published.
import { Viewer } from "./render.js";
import { StripChart } from "./plots.js";

const $ = (id) => document.getElementById(id);
const RIGHT = ["#0b6fd1", "#5aa9f0"], LEFT = ["#d26b00", "#f2a446"];   // [light, dark]
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
// pages served before 2026-11-12 registered a cross-origin-isolation service worker (coi-sw.js); retire it
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations().then((rs) => rs.forEach((r) => r.unregister())).catch(() => {});
}
const ui = { state: null, firstScene: true, ready: false, playing: true };
const set = (o) => worker.postMessage({ type: "set", ...o });

function theme() {
  const r = document.documentElement;
  const dark = r.dataset.theme === "dark" || (!r.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const css = getComputedStyle(r);
  return { dark, ink: css.getPropertyValue("--ink").trim(), muted: css.getPropertyValue("--muted").trim(),
    grid: css.getPropertyValue("--grid").trim() };
}

const viewer = new Viewer($("view"));
const charts = {
  hip: new StripChart($("plot-hip"), { title: "Hip flexion (deg)", series: [{ name: "R", color: RIGHT }, { name: "L", color: LEFT }] }),
  knee: new StripChart($("plot-knee"), { title: "Knee flexion (deg)", series: [{ name: "R", color: RIGHT }, { name: "L", color: LEFT }] }),
  exo: new StripChart($("plot-exo"), { title: "Device torque (N·m, + = flexion)", range: [-15, 15], series: [{ name: "R", color: RIGHT }, { name: "L", color: LEFT }] }),
  act: new StripChart($("plot-act"), { title: "Hip-muscle activation, mean (%)", series: [{ name: "R", color: RIGHT, digits: 0 }, { name: "L", color: LEFT, digits: 0 }] }),
};

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") {
    $("load-msg").textContent = m.msg;
    $("load-bar").style.width = `${Math.round(100 * m.frac)}%`;
    $("load-mb").textContent = `${m.mb.toFixed(0)} MB`;
  } else if (m.type === "ready") {
    buildClipList(m.clips);
    ui.ready = true;
    $("load-bar").style.width = "100%";
    $("loading").classList.add("done");
  } else if (m.type === "scene") {
    viewer.build(m.scene);
    viewer.setTheme(theme().dark);
    $("device-controls").hidden = !m.exo;
    $("plot-exo").parentElement.hidden = !m.exo;
    $("device-note").textContent = m.exo || "No device: the human alone.";
    ui.needPreset = ui.firstScene;
  } else if (m.type === "state") {
    ui.state = m;
    const s = m.samples;
    for (let k = 0; k + 8 < s.length; k += 9) {
      charts.hip.push(s[k], [s[k + 1], s[k + 2]]);
      charts.knee.push(s[k], [s[k + 3], s[k + 4]]);
      charts.exo.push(s[k], [s[k + 5], s[k + 6]]);
      charts.act.push(s[k], [s[k + 7], s[k + 8]]);
    }
  } else if (m.type === "restart") {
    for (const c of Object.values(charts)) c.clear();
  } else if (m.type === "status") {
    $("status").textContent = m.msg;
  } else if (m.type === "error") {
    if (!ui.ready) { $("load-msg").textContent = `Could not start: ${m.msg}`; $("loading").classList.add("error"); }
    else $("status").textContent = `Error: ${m.msg}`;
  }
};
worker.onerror = (e) => { $("load-msg").textContent = `Could not start: ${e.message}`; $("loading").classList.add("error"); };

function frame() {
  const s = ui.state;
  if (s) {
    viewer.update(s);
    if (ui.needPreset) { viewer.preset("persp"); ui.needPreset = false; ui.firstScene = false; }
    const st = s.stats;
    $("stat").textContent = `${st.t.toFixed(st.playing ? 1 : 2)} s · ${st.playing ? `${st.rtf.toFixed(2)}× real time` : "paused"} · ` +
      `${st.stepMs.toFixed(1)} ms/step · ${st.backend === "onnx" ? "ONNX" : "JS"} network`;
  }
  viewer.render();
  const th = theme();
  for (const c of Object.values(charts)) if (!c.cv.parentElement.hidden) c.draw(th);
  requestAnimationFrame(frame);
}

function buildClipList(clips) {
  const sel = $("clip"), groups = {};
  for (const c of clips) (groups[c.category] ||= []).push(c);
  const names = { walk: "Walking", run: "Running", squat: "Squat", stairs: "Stairs", slope: "Ramp" };
  sel.textContent = "";
  for (const [cat, cs] of Object.entries(groups)) {
    const og = document.createElement("optgroup");
    og.label = names[cat] || cat;
    for (const c of cs) {
      const o = document.createElement("option");
      o.value = c.id;
      o.textContent = `${c.label} (${c.seconds.toFixed(0)} s)`;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
}

function setPlaying(p) {
  ui.playing = p;
  $("play").textContent = p ? "Pause" : "Play";
  set({ playing: p });
}

$("clip").onchange = () => worker.postMessage({ type: "clip", id: $("clip").value });
for (const b of document.querySelectorAll("[data-device]")) {
  b.onclick = () => {
    document.querySelectorAll("[data-device]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    worker.postMessage({ type: "variant", variant: b.dataset.device });
  };
}
$("assist").onchange = () => set({ assist: $("assist").checked });
$("arrows").onchange = () => set({ arrowsOn: $("arrows").checked });
$("scale").oninput = () => {
  $("scale-v").textContent = `${Math.round(100 * Number($("scale").value))}%`;
  set({ scale: Number($("scale").value) });
};
$("play").onclick = () => setPlaying(!ui.playing);
// one control step (10 ms); pauses first
function stepOnce() {
  if (ui.playing) setPlaying(false);
  worker.postMessage({ type: "step" });
}
$("step").onclick = stepOnce;
$("restart").onclick = () => worker.postMessage({ type: "restart" });
$("speed").onchange = () => set({ speed: Number($("speed").value) });
$("stochastic").onchange = () => set({ deterministic: !$("stochastic").checked });
$("loop").onchange = () => set({ loop: $("loop").checked });
$("muscles").onchange = () => set({ muscles: $("muscles").checked });
$("ghost").onchange = () => set({ ghost: $("ghost").checked });
$("follow").onchange = () => { viewer.follow = $("follow").checked; };
for (const b of document.querySelectorAll("[data-cam]")) b.onclick = () => viewer.preset(b.dataset.cam);
$("theme").onclick = () => {
  document.documentElement.dataset.theme = theme().dark ? "light" : "dark";
  viewer.setTheme(theme().dark);
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => viewer.setTheme(theme().dark));
// keys: Enter play / pause, Space one step (held: keeps stepping), R restart, 1-4 cameras. They work with a button
// or checkbox focused too; that control's own Enter / Space action is suppressed (keyup) so it does not fire twice.
const typing = (t) => t.tagName === "SELECT" || t.tagName === "TEXTAREA" ||
  (t.tagName === "INPUT" && !["checkbox", "range", "radio"].includes(t.type));
document.addEventListener("keydown", (e) => {
  if (typing(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Enter") { e.preventDefault(); if (!e.repeat) setPlaying(!ui.playing); }
  else if (e.code === "Space") { e.preventDefault(); stepOnce(); }
  else if (e.key === "r" || e.key === "R") worker.postMessage({ type: "restart" });
  else if (["1", "2", "3", "4"].includes(e.key)) viewer.preset(["persp", "side", "front", "top"][Number(e.key) - 1]);
});
document.addEventListener("keyup", (e) => {
  if (!typing(e.target) && (e.key === "Enter" || e.code === "Space")) e.preventDefault();
});

// MuJoCo 4's default WebAssembly build is single-threaded and onnxruntime runs one thread: no cross-origin isolation
// (SharedArrayBuffer) is needed, so any static host works as is
worker.postMessage({ type: "init" });
requestAnimationFrame(frame);
