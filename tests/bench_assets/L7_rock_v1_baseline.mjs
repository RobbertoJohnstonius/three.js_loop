/**
 * rock_v1.mjs — starter procedural rock asset for the threejs_loop.
 * Hand-authored baseline; the loop will improve from here.
 */
import * as THREE from 'three';

export function createAsset() {
  const group = new THREE.Group();

  // Main body: slightly irregular icosahedron
  const geo = new THREE.IcosahedronGeometry(0.8, 1);

  // Displace vertices randomly for a rocky silhouette
  const pos = geo.attributes.position;
  const seed = 42;
  let s = seed;
  function rand() {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff) * 2 - 1;
  }

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const len = Math.sqrt(x * x + y * y + z * z);
    const disp = 1 + rand() * 0.25;
    pos.setXYZ(i, x / len * 0.8 * disp, y / len * 0.8 * disp, z / len * 0.8 * disp);
  }

  geo.computeVertexNormals();

  // Flat-ish bottom: push down vertices below y = -0.3
  for (let i = 0; i < pos.count; i++) {
    if (pos.getY(i) < -0.3) pos.setY(i, -0.4);
  }
  geo.computeVertexNormals();

  // UV: simple spherical projection
  const uvAttr = new THREE.Float32BufferAttribute(pos.count * 2, 2);
  for (let i = 0; i < pos.count; i++) {
    const nx = pos.getX(i);
    const ny = pos.getY(i);
    const nz = pos.getZ(i);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const u = 0.5 + Math.atan2(nz / len, nx / len) / (2 * Math.PI);
    const v = 0.5 - Math.asin(Math.max(-1, Math.min(1, ny / len))) / Math.PI;
    uvAttr.setXY(i, u, v);
  }
  geo.setAttribute('uv', uvAttr);

  const mat = new THREE.MeshStandardMaterial({
    color: 0x7a7068,
    roughness: 0.92,
    metalness: 0.04,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);

  return group;
}
