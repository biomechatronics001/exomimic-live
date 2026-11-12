# ExoMimic Live

**Open:** https://biomechatronics001.github.io/exomimic-live/

A 354-muscle human with learned hip and knee exoskeleton controllers, simulated in your browser. Nothing to install
(desktop Chrome or Edge).

## How to use

- **Motion**: choose walk, run, squat, stairs or ramp.
- **Device**: None, Hip exo or Knee exo. Turn the assist on or off, or scale it.
- **Torque arrows**: the curved arrow at each driven joint is the motor torque. The arrowhead shows the direction
  (flex / ext), the arc length the size (300° = 15 N·m). Blue is right, orange is left.
- **Muscles**: colour is activation, grey (idle) to orange to red (35 % and above).
- **Camera**: 3/4, Side, Front, Top. The side view shows the torque arrows best.

| Key | Action |
|---|---|
| Enter | play / pause |
| Space | one 10 ms step (hold to keep stepping) |
| R | restart |
| 1–4 | camera views |

## Run locally

```bash
python3 -m http.server 8000      # in this folder, then open http://localhost:8000/
```

## Credits

MuJoCo (Apache-2.0), onnxruntime-web (MIT), three.js (MIT). Musculoskeletal model from MyoSuite (Apache-2.0).
Reference motions retargeted from AMASS (KIT, ACCAD, BMLmovi), for non-commercial research use only.
