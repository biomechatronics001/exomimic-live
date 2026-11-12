// Small strip charts on a canvas: the last few seconds of a few signals, right / left in two fixed hues, a legend,
// a recessive zero line and the current values as text.
export class StripChart {
  constructor(canvas, { title, unit, series, window = 4, range = null }) {
    this.cv = canvas; this.title = title; this.unit = unit; this.series = series;
    this.window = window; this.range = range;
    this.t = []; this.v = series.map(() => []);
  }

  clear() { this.t.length = 0; for (const s of this.v) s.length = 0; }

  push(t, values) {
    this.t.push(t);
    values.forEach((x, i) => this.v[i].push(x));
    const t0 = t - this.window;
    let k = 0;
    while (k < this.t.length && this.t[k] < t0) k++;
    if (k > 0) { this.t.splice(0, k); for (const s of this.v) s.splice(0, k); }
  }

  draw(theme) {
    const cv = this.cv, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const padL = 34, padR = 8, padT = 20, padB = 6, w = W - padL - padR, h = H - padT - padB;
    g.font = "600 11px system-ui, sans-serif";
    g.fillStyle = theme.ink;
    g.textBaseline = "alphabetic";
    g.fillText(`${this.title}`, 0, 12);
    // legend + current values (text in ink, a colour swatch carries identity)
    let x = W;
    g.font = "11px system-ui, sans-serif";
    for (let i = this.series.length - 1; i >= 0; i--) {
      const s = this.series[i], cur = this.v[i].length ? this.v[i][this.v[i].length - 1] : NaN;
      const txt = `${s.name} ${Number.isFinite(cur) ? cur.toFixed(s.digits ?? 1) : "–"}`;
      const tw = g.measureText(txt).width;
      x -= tw;
      g.fillStyle = theme.muted;
      g.fillText(txt, x, 12);
      x -= 12;
      g.fillStyle = s.color[theme.dark ? 1 : 0];
      g.fillRect(x, 5, 8, 8);
      x -= 10;
    }
    if (!this.t.length) return;
    let lo = Infinity, hi = -Infinity;
    if (this.range) [lo, hi] = this.range;
    else for (const s of this.v) for (const y of s) { if (y < lo) lo = y; if (y > hi) hi = y; }
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    const span = hi - lo; lo -= 0.08 * span; hi += 0.08 * span;
    const t1 = this.t[this.t.length - 1], t0 = t1 - this.window;
    const X = (t) => padL + ((t - t0) / this.window) * w, Y = (y) => padT + (1 - (y - lo) / (hi - lo)) * h;
    // axis labels + zero line
    g.fillStyle = theme.muted;
    g.font = "10px system-ui, sans-serif";
    g.textAlign = "right";
    g.fillText(fmtTick(hi), padL - 4, padT + 8);
    g.fillText(fmtTick(lo), padL - 4, padT + h);
    g.textAlign = "left";
    g.strokeStyle = theme.grid;
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, padT); g.lineTo(padL, padT + h); g.stroke();
    if (lo < 0 && hi > 0) { g.beginPath(); g.moveTo(padL, Y(0)); g.lineTo(padL + w, Y(0)); g.stroke(); }
    g.lineWidth = 2;
    g.lineJoin = "round";
    this.series.forEach((s, i) => {
      const v = this.v[i];
      g.strokeStyle = s.color[theme.dark ? 1 : 0];
      g.setLineDash(s.dash || []);
      g.beginPath();
      for (let k = 0; k < v.length; k++) { const px = X(this.t[k]), py = Y(v[k]); k ? g.lineTo(px, py) : g.moveTo(px, py); }
      g.stroke();
    });
    g.setLineDash([]);
  }
}

function fmtTick(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(0) : v.toFixed(1);
}
