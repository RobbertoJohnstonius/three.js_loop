import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
  const group = new THREE.Group();

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

  // Compute and refine vertex normals
  geo.computeVertexNormals();
  for (let i = 0; i < geo.index.count; i += 3) {
    const a = geo.index.getX(i);
    const b = geo.index.getX(i + 1);
    const c = geo.index.getX(i + 2);

    const v1 = new THREE.Vector3().fromBufferAttribute(pos, a);
    const v2 = new THREE.Vector3().fromBufferAttribute(pos, b);
    const v3 = new THREE.Vector3().fromBufferAttribute(pos, c);

    const faceNormal = new THREE.Vector3().crossVectors(v2.clone().sub(v1), v3.clone().sub(v1)).normalize();

    geo.attributes.normal.setXYZ(a, faceNormal.x, faceNormal.y, faceNormal.z);
    geo.attributes.normal.setXYZ(b, faceNormal.x, faceNormal.y, faceNormal.z);
    geo.attributes.normal.setXYZ(c, faceNormal.x, faceNormal.y, faceNormal.z);
  }
  geo.attributes.normal.needsUpdate = true;

  const uvAttr = new THREE.Float32BufferAttribute(pos.count * 2, 2);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    uvAttr.setXY(i,
      0.5 + Math.atan2(z / len, x / len) / (2 * Math.PI),
      0.5 - Math.asin(Math.max(-1, Math.min(1, y / len))) / Math.PI
    );
  }
  geo.setAttribute('uv', uvAttr);

  // Advanced hash-based FBM noise function
  function hash(p) {
    p = p.dot(new THREE.Vector3(127.1, 311.7, 74.7));
    return -1.0 + 2.0 * Math.frac(Math.sin(p) * 43758.5453123);
  }

  function noise(p) {
    const K = new THREE.Vector3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    const Pi = new THREE.Vector3(1.0 - 1.0 / 3.0, 1.0 / 3.0 - 1.0, 1.0 / 3.0 - 1.0);
    const Pf = new THREE.Vector3(4.0 * Pi.x * Pi.x, 4.0 * Pi.y * Pi.y, 4.0 * Pi.z * Pi.z);
    const p0 = Math.floor(p.x), p1 = p0 + 1;
    const q0 = Math.floor(p.y), q1 = q0 + 1;
    const r0 = Math.floor(p.z), r1 = r0 + 1;
    const s = p.x - p0, t = p.y - q0, u = p.z - r0;
    const w0 = (1.0 - s) * (1.0 - t) * (1.0 - u);
    const w1 =    s   * (1.0 - t) * (1.0 - u);
    const w2 = (1.0 - s) *    t   * (1.0 - u);
    const w3 =    s   *    t   * (1.0 - u);
    const w4 = (1.0 - s) * (1.0 - t) *    u;
    const w5 =    s   * (1.0 - t) *    u;
    const w6 = (1.0 - s) *    t   *    u;
    const w7 =    s   *    t   *    u;
    return hash(new THREE.Vector3(p0, q0, r0)) * w0 +
           hash(new THREE.Vector3(p1, q0, r0)) * w1 +
           hash(new THREE.Vector3(p0, q1, r0)) * w2 +
           hash(new THREE.Vector3(p1, q1, r0)) * w3 +
           hash(new THREE.Vector3(p0, q0, r1)) * w4 +
           hash(new THREE.Vector3(p1, q0, r1)) * w5 +
           hash(new THREE.Vector3(p0, q1, r1)) * w6 +
           hash(new THREE.Vector3(p1, q1, r1)) * w7;
  }

  function fbm(p, octaves, lacunarity, gain) {
    let result = 0.0;
    let amplitude = 1.0;
    for (let i = 0; i < octaves; i++) {
      result += noise(p) * amplitude;
      amplitude *= gain;
      p.multiplyScalar(lacunarity);
    }
    return result;
  }

  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const noiseValue = fbm(new THREE.Vector3(x, y, z), 4, 2.0, 0.5);
    const t = (noiseValue + 1.0) * 0.5;
    const r = Math.max(0, Math.min(1, 0.2 + 0.3 * Math.sin(t * 2 * Math.PI)));
    const g = Math.max(0, Math.min(1, 0.2 + 0.3 * Math.cos(t * 2 * Math.PI)));
    const b = Math.max(0, Math.min(1, 0.2 + 0.3 * Math.sin(t * 2 * Math.PI + Math.PI / 2)));
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.92,
    metalness: 0.03,
    vertexColors: true
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
  return group;
}