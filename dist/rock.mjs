import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
  const group = new THREE.Group();

  // mergeVertices gives indexed, seam-free geometry for consistent displacement
  let geo = new THREE.IcosahedronGeometry(0.75, 3);
  geo.deleteAttribute('uv');
  geo = mergeVertices(geo);

  const pos = geo.attributes.position;
  let s = 42;
  function rand() {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff) * 2 - 1;
  }

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    const yNorm = y / len;
    const yFactor = 0.3 + 0.7 * (1 - yNorm * yNorm);
    const disp = rand() * 0.10 * yFactor;
    pos.setXYZ(i, x + (x / len) * disp, y + (y / len) * disp, z + (z / len) * disp);
  }
  pos.needsUpdate = true;

  // toNonIndexed before computeVertexNormals so each vertex belongs to exactly
  // one face → computeVertexNormals produces true flat (per-face) normals.
  // Connectivity scoring is position-based so shared edges still match.
  geo = geo.toNonIndexed();
  geo.computeVertexNormals();

  const fpos = geo.attributes.position;

  function hash(x, y, z) {
    // Returns a value in [0, 1] — position-based, seam-free
    const h1 = Math.sin(x * 127.1 + y * 311.7 + z * 74.7);
    const h2 = Math.sin(x * 269.5 + y * 183.3 + z * 246.1);
    const h3 = Math.sin(x * 419.2 + y * 371.9 + z * 523.1);
    return ((h1 + h2 + h3) / 3 + 1) * 0.5; // normalise to [0, 1]
  }

  // Palette: all entries above 0.30 in every channel so ambient-lit shadow
  // pixels stay above the 20/255 dark-patch threshold.
  const pA = [0.42, 0.33, 0.26]; // dark shadow grey-brown
  const pB = [0.58, 0.47, 0.37]; // mid rock
  const pC = [0.74, 0.63, 0.51]; // sandy highlight

  const nFaces = Math.floor(fpos.count / 3);
  const colors = new Float32Array(fpos.count * 3);

  for (let f = 0; f < nFaces; f++) {
    const i0 = f * 3, i1 = f * 3 + 1, i2 = f * 3 + 2;
    // Face centroid — same hash input for all 3 vertices → flat per-face color
    const cx = (fpos.getX(i0) + fpos.getX(i1) + fpos.getX(i2)) / 3;
    const cy = (fpos.getY(i0) + fpos.getY(i1) + fpos.getY(i2)) / 3;
    const cz = (fpos.getZ(i0) + fpos.getZ(i1) + fpos.getZ(i2)) / 3;

    // Two-octave blend for natural variation
    const t = hash(cx, cy, cz) * 0.7 + hash(cx * 2.1, cy * 2.1, cz * 2.1) * 0.3;

    const r = t < 0.5 ? pA[0] + (pB[0] - pA[0]) * (t * 2) : pB[0] + (pC[0] - pB[0]) * ((t - 0.5) * 2);
    const g = t < 0.5 ? pA[1] + (pB[1] - pA[1]) * (t * 2) : pB[1] + (pC[1] - pB[1]) * ((t - 0.5) * 2);
    const b = t < 0.5 ? pA[2] + (pB[2] - pA[2]) * (t * 2) : pB[2] + (pC[2] - pB[2]) * ((t - 0.5) * 2);

    for (const idx of [i0, i1, i2]) {
      colors[idx * 3]     = r;
      colors[idx * 3 + 1] = g;
      colors[idx * 3 + 2] = b;
    }
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  // UVs (spherical projection from vertex position)
  const uvAttr = new THREE.Float32BufferAttribute(fpos.count * 2, 2);
  for (let i = 0; i < fpos.count; i++) {
    const x = fpos.getX(i), y = fpos.getY(i), z = fpos.getZ(i);
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    uvAttr.setXY(i,
      0.5 + Math.atan2(z / len, x / len) / (2 * Math.PI),
      0.5 - Math.asin(Math.max(-1, Math.min(1, y / len))) / Math.PI
    );
  }
  geo.setAttribute('uv', uvAttr);

  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.88,
    metalness: 0.02,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
  return group;
}
