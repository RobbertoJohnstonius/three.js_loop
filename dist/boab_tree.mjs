import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
  const group = new THREE.Group();

  function hashPos(x, y, z) {
    const h1 = Math.sin(x * 127.1 + y * 311.7 + z * 74.7);
    const h2 = Math.sin(x * 269.5 + y * 183.3 + z * 246.1);
    return ((h1 + h2) * 0.5 + 1) * 0.5;
  }

  function applyFaceColors(geo, palette) {
    geo = geo.toNonIndexed();
    geo.computeVertexNormals();
    const fpos = geo.attributes.position;
    const nFaces = Math.floor(fpos.count / 3);
    const colors = new Float32Array(fpos.count * 3);
    for (let f = 0; f < nFaces; f++) {
      const i0=f*3, i1=f*3+1, i2=f*3+2;
      const cx=(fpos.getX(i0)+fpos.getX(i1)+fpos.getX(i2))/3;
      const cy=(fpos.getY(i0)+fpos.getY(i1)+fpos.getY(i2))/3;
      const cz=(fpos.getZ(i0)+fpos.getZ(i1)+fpos.getZ(i2))/3;
      const t = hashPos(cx, cy, cz);
      const r = t<0.5 ? palette[0][0]+(palette[1][0]-palette[0][0])*(t*2) : palette[1][0]+(palette[2][0]-palette[1][0])*((t-0.5)*2);
      const g = t<0.5 ? palette[0][1]+(palette[1][1]-palette[0][1])*(t*2) : palette[1][1]+(palette[2][1]-palette[1][1])*((t-0.5)*2);
      const b = t<0.5 ? palette[0][2]+(palette[1][2]-palette[0][2])*(t*2) : palette[1][2]+(palette[2][2]-palette[1][2])*((t-0.5)*2);
      for (const idx of [i0,i1,i2]) { colors[idx*3]=r; colors[idx*3+1]=g; colors[idx*3+2]=b; }
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const uvAttr = new THREE.Float32BufferAttribute(fpos.count * 2, 2);
    for (let i = 0; i < fpos.count; i++) {
      const x=fpos.getX(i), y=fpos.getY(i), z=fpos.getZ(i);
      const len=Math.sqrt(x*x+y*y+z*z)||1;
      uvAttr.setXY(i, 0.5+Math.atan2(z/len,x/len)/(2*Math.PI),
                      0.5-Math.asin(Math.max(-1,Math.min(1,y/len)))/Math.PI);
    }
    geo.setAttribute('uv', uvAttr);
    return geo;
  }

  function stdMat() {
    return new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.85, metalness: 0.02 });
  }

  const trunkPal = [[0.38, 0.28, 0.20], [0.54, 0.42, 0.32], [0.66, 0.54, 0.42]];
  const leafPal  = [[0.26, 0.38, 0.14], [0.34, 0.50, 0.20], [0.44, 0.62, 0.26]];

  // ── TRUNK ───────────────────────────────────────────────────────────────────
  // Taller trunk so it dominates the silhouette like a real boab.
  // No base flare — the barrel profile alone reads as boab.
  let trunkGeo = new THREE.CylinderGeometry(0.11, 0.24, 1.15, 10, 16, false);
  trunkGeo.deleteAttribute('uv');
  trunkGeo = mergeVertices(trunkGeo);

  const tpos = trunkGeo.attributes.position;
  for (let i = 0; i < tpos.count; i++) {
    const x=tpos.getX(i), y=tpos.getY(i), z=tpos.getZ(i);
    const yN = (y + 0.575) / 1.15; // [0=bottom, 1=crown]
    // Barrel bulge: peak at 45% up, fading to zero at crown and base
    const tb = (yN - 0.45) / 0.30;
    const bulge = Math.exp(-tb * tb * 2.0) * 1.6;
    tpos.setXYZ(i, x * (1 + bulge), y, z * (1 + bulge));
  }
  tpos.needsUpdate = true;
  trunkGeo = applyFaceColors(trunkGeo, trunkPal);

  const trunk = new THREE.Mesh(trunkGeo, stdMat());
  trunk.name = 'trunk';
  trunk.position.y = -0.575; // bottom at y=-1.15, crown at y=0
  trunk.castShadow = true; trunk.receiveShadow = true;
  group.add(trunk);

  // ── CROWN BRANCHES ──────────────────────────────────────────────────────────
  // Branches at ~50° from vertical — enough upward lift to keep canopy clear
  // of the trunk, but wide spread to match boab silhouette.
  // All branch base origins at the crown (world y=0).
  const branchDefs = [
    { d: new THREE.Vector3(0,      1,      0),     len: 0.28, r: 0.075 }, // central stub
    { d: new THREE.Vector3(0.766,  0.643,  0),     len: 0.42, r: 0.068 }, // +X  50°
    { d: new THREE.Vector3(-0.766, 0.643,  0),     len: 0.42, r: 0.068 }, // -X
    { d: new THREE.Vector3(0,      0.643,  0.766), len: 0.40, r: 0.062 }, // +Z
    { d: new THREE.Vector3(0,      0.643, -0.766), len: 0.40, r: 0.062 }, // -Z
    { d: new THREE.Vector3(0.542,  0.643,  0.542), len: 0.36, r: 0.056 }, // +X+Z
    { d: new THREE.Vector3(-0.542, 0.643, -0.542), len: 0.36, r: 0.056 }, // -X-Z
  ];

  const branchTips = [];
  const up = new THREE.Vector3(0, 1, 0);

  for (const { d, len, r } of branchDefs) {
    const dn = d.clone().normalize();
    let bGeo = new THREE.CylinderGeometry(r * 0.50, r, len, 6, 2, false);
    bGeo.deleteAttribute('uv');
    bGeo = mergeVertices(bGeo);
    bGeo = applyFaceColors(bGeo, trunkPal);

    const branch = new THREE.Mesh(bGeo, stdMat());
    branch.name = 'trunk';
    branch.position.set(dn.x*len*0.5, dn.y*len*0.5, dn.z*len*0.5);
    if (dn.x !== 0 || dn.z !== 0) branch.quaternion.setFromUnitVectors(up, dn);
    branch.castShadow = true; branch.receiveShadow = true;
    group.add(branch);

    branchTips.push(new THREE.Vector3(dn.x*len, dn.y*len, dn.z*len));
  }

  // ── CANOPY CLUSTERS ─────────────────────────────────────────────────────────
  // detail=2 icosahedra (80 faces each) give smooth Realistic mode without big polygons.
  // Smaller radii + fewer fill clusters = more open canopy, more light below.
  const primaryRadii = [0.30, 0.26, 0.26, 0.24, 0.24, 0.22, 0.22];

  const fillClusters = [
    { pos: new THREE.Vector3( 0.22, 0.42,  0.22), r: 0.19 },
    { pos: new THREE.Vector3(-0.22, 0.42, -0.22), r: 0.19 },
    { pos: new THREE.Vector3( 0.20, 0.42, -0.20), r: 0.17 },
    { pos: new THREE.Vector3(-0.20, 0.42,  0.20), r: 0.17 },
  ];

  const allClusters = [
    ...branchTips.map((pos, i) => ({ pos, r: primaryRadii[i] })),
    ...fillClusters,
  ];

  let lcg = 31337;
  function lcgNext() {
    lcg = (lcg * 1664525 + 1013904223) & 0xffffffff;
    return (lcg >>> 0) / 0xffffffff;
  }

  for (const { pos, r } of allClusters) {
    // detail=2: edge ≈ r×0.54, max safe disp = 0.40×r×0.54 ≈ 0.22r
    let lGeo = new THREE.IcosahedronGeometry(r, 2);
    lGeo.deleteAttribute('uv');
    lGeo = mergeVertices(lGeo);

    const lp = lGeo.attributes.position;
    const maxD = r * 0.18;
    for (let i = 0; i < lp.count; i++) {
      const disp = lcgNext() * maxD * 2 - maxD;
      const x=lp.getX(i), y=lp.getY(i), z=lp.getZ(i);
      const len=Math.sqrt(x*x+y*y+z*z)||1;
      lp.setXYZ(i, x+(x/len)*disp, y+(y/len)*disp, z+(z/len)*disp);
    }
    lp.needsUpdate = true;

    lGeo = applyFaceColors(lGeo, leafPal);
    const leaf = new THREE.Mesh(lGeo, stdMat());
    leaf.name = 'foliage';
    leaf.position.copy(pos);
    leaf.castShadow = true; leaf.receiveShadow = true;
    group.add(leaf);
  }

  return group;
}
