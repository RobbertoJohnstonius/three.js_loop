import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
  const group = new THREE.Group();

  // Position-based hash → [0, 1]
  function hashPos(x, y, z) {
    const h1 = Math.sin(x * 127.1 + y * 311.7 + z * 74.7);
    const h2 = Math.sin(x * 269.5 + y * 183.3 + z * 246.1);
    return ((h1 + h2) * 0.5 + 1) * 0.5;
  }

  // Assign per-face flat vertex colors from centroid + palette
  function applyFaceColors(geo, palette) {
    geo = geo.toNonIndexed();
    geo.computeVertexNormals();
    const fpos = geo.attributes.position;
    const nFaces = Math.floor(fpos.count / 3);
    const colors = new Float32Array(fpos.count * 3);
    for (let f = 0; f < nFaces; f++) {
      const i0=f*3, i1=f*3+1, i2=f*3+2;
      const cx = (fpos.getX(i0)+fpos.getX(i1)+fpos.getX(i2)) / 3;
      const cy = (fpos.getY(i0)+fpos.getY(i1)+fpos.getY(i2)) / 3;
      const cz = (fpos.getZ(i0)+fpos.getZ(i1)+fpos.getZ(i2)) / 3;
      const t = hashPos(cx, cy, cz);
      const r = t < 0.5
        ? palette[0][0] + (palette[1][0]-palette[0][0])*(t*2)
        : palette[1][0] + (palette[2][0]-palette[1][0])*((t-0.5)*2);
      const g = t < 0.5
        ? palette[0][1] + (palette[1][1]-palette[0][1])*(t*2)
        : palette[1][1] + (palette[2][1]-palette[1][1])*((t-0.5)*2);
      const b = t < 0.5
        ? palette[0][2] + (palette[1][2]-palette[0][2])*(t*2)
        : palette[1][2] + (palette[2][2]-palette[1][2])*((t-0.5)*2);
      for (const idx of [i0,i1,i2]) { colors[idx*3]=r; colors[idx*3+1]=g; colors[idx*3+2]=b; }
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    // Spherical UVs
    const uvAttr = new THREE.Float32BufferAttribute(fpos.count * 2, 2);
    for (let i = 0; i < fpos.count; i++) {
      const x=fpos.getX(i), y=fpos.getY(i), z=fpos.getZ(i);
      const len = Math.sqrt(x*x+y*y+z*z)||1;
      uvAttr.setXY(i, 0.5+Math.atan2(z/len,x/len)/(2*Math.PI),
                      0.5-Math.asin(Math.max(-1,Math.min(1,y/len)))/Math.PI);
    }
    geo.setAttribute('uv', uvAttr);
    return geo;
  }

  function stdMat() {
    return new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.85, metalness: 0.02 });
  }

  // ── TRUNK — bottle-shaped via CylinderGeometry vertex displacement ──────────
  let trunkGeo = new THREE.CylinderGeometry(0.12, 0.18, 1.3, 8, 12, false);
  trunkGeo.deleteAttribute('uv');

  const tpos = trunkGeo.attributes.position;
  for (let i = 0; i < tpos.count; i++) {
    const x = tpos.getX(i), y = tpos.getY(i), z = tpos.getZ(i);
    const yN = (y + 0.65) / 1.3;
    const t = (yN - 0.35) / 0.30;
    const bulge = Math.exp(-t * t * 2.0) * 2.2;
    tpos.setXYZ(i, x * (1 + bulge), y, z * (1 + bulge));
  }
  tpos.needsUpdate = true;

  const trunkPal = [[0.50, 0.44, 0.40], [0.62, 0.56, 0.52], [0.72, 0.66, 0.62]];
  trunkGeo = applyFaceColors(trunkGeo, trunkPal);
  const trunk = new THREE.Mesh(trunkGeo, stdMat());
  trunk.position.y = -0.35;
  trunk.castShadow = true; trunk.receiveShadow = true;
  group.add(trunk);

  // ── CANOPY — displaced icosahedra for leaf clusters ─────────────────────────
  const leafPal = [[0.30, 0.40, 0.20], [0.38, 0.50, 0.24], [0.48, 0.60, 0.28]];

  const clusters = [
    { pos: [0, 0.80, 0], r: 0.42 },
    { pos: [ 0.45, 0.60, 0.15], r: 0.30 },
    { pos: [-0.45, 0.62, 0.10], r: 0.30 },
    { pos: [ 0.20, 0.65,-0.42], r: 0.28 },
    { pos: [-0.20, 0.60, 0.42], r: 0.28 },
    { pos: [ 0.55, 0.50,-0.25], r: 0.24 },
    { pos: [-0.55, 0.50, 0.20], r: 0.24 },
  ];

  for (const { pos, r } of clusters) {
    let lGeo = new THREE.IcosahedronGeometry(r, 1);
    lGeo.deleteAttribute('uv');
    lGeo = mergeVertices(lGeo);

    const lp = lGeo.attributes.position;
    for (let i = 0; i < lp.count; i++) {
      const x = lp.getX(i), y = lp.getY(i), z = lp.getZ(i);
      const disp = (hashPos(x, y, z) * 0.08 - 0.04);
      const len = Math.sqrt(x*x+y*y+z*z)||1;
      lp.setXYZ(i, x+(x/len)*disp, y+(y/len)*disp, z+(z/len)*disp);
    }
    lp.needsUpdate = true;

    lGeo = applyFaceColors(lGeo, leafPal);
    const leaf = new THREE.Mesh(lGeo, stdMat());
    leaf.position.set(...pos);
    leaf.castShadow = true; leaf.receiveShadow = true;
    group.add(leaf);
  }

  return group;
}