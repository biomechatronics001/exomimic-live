// Ground height of the canonical stair / ramp pieces (loco_mujoco/core/terrain/canonical.py height_at_xy): the
// exact top surface of every piece inside its y extent and x span, the floor (0) everywhere else.
export class CanonicalTerrain {
  constructor(spec) { this.pieces = spec.pieces || []; }

  height(x, y) {
    let h = 0;
    for (const g of this.pieces) {
      if (Math.abs(y - g.y_center) > g.half_width) continue;
      const ahead = x - g.start_x;
      let span, top;
      if (g.kind === "stairs") {
        span = g.n_steps * g.step_run + g.length;
        top = Math.min(Math.max(Math.floor(ahead / g.step_run) + 1, 1), g.n_steps) * g.step_rise;
      } else {
        span = 2 * g.length;
        top = Math.min(ahead, g.length) * Math.tan(g.slope_rad);
      }
      if (ahead >= 0 && ahead <= span) h = Math.max(h, top);
    }
    return h;
  }
}
