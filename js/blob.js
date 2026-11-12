// Reader for the array containers written by export/blob.py: <name>.json (index) + <name>.bin (raw arrays).
const CTOR = { f8: Float64Array, f4: Float32Array, i4: Int32Array, u1: Uint8Array };

export function parseBlob(index, buffer) {
  const out = {};
  for (const [k, e] of Object.entries(index.arrays)) {
    const n = e.shape.reduce((a, b) => a * b, 1);
    if (e.dtype === "f2") {
      const h = new Uint16Array(buffer, e.offset, n);
      const f = new Float32Array(n);
      for (let i = 0; i < n; i++) f[i] = halfToFloat(h[i]);
      out[k] = f;
    } else {
      out[k] = new CTOR[e.dtype](buffer, e.offset, n);
    }
    out[k].shape = e.shape;
  }
  out.meta = index.meta || {};
  return out;
}

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

// fetchText / fetchBytes are injected so the same code runs in the browser (fetch) and in node (fs)
export async function loadBlob(base, io) {
  const index = JSON.parse(await io.text(base + ".json"));
  const buf = await io.bytes(base + ".bin");
  return parseBlob(index, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
