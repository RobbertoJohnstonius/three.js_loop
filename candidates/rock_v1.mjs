/**
 * rock_v1.mjs — corrected baseline using mergeVertices.
 *
 * IcosahedronGeometry is non-indexed: every triangle corner is a separate
 * vertex copy. Calling mergeVertices() first converts it to a properly
 * indexed geometry (101 unique vertices for detail=2), so:
 *   1. Sequential-LCG displacement is consistent — each unique position
 *      gets exactly one displacement value, keeping the mesh connected.
 *   2. computeVertexNormals() averages across shared indices → smooth shading.
 */
import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
  const group = new THREE.Group();

  // detail=2 → 180 tris.  Delete UV first so mergeVertices merges seam duplicates
  // (IcosahedronGeometry creates 9 vertex pairs at the same 3D position but u=0/u=1;
  //  keeping UV prevents merging → different LCG displacements → physical seam crack).
  // Without UV: 92 unique indexed vertices, connectivity=1.0. With UV: 101, connectivity=0.963.
  let geo = new THREE.IcosahedronGeometry(0.75, 2);
  geo.deleteAttribute('uv');
  geo = mergeVertices(geo);

  const pos = geo.attributes.position;
  let s = 42;
  function rand() {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff) * 2 - 1;
  }

  // Radial displacement: ±8% of radius. Tapers toward poles to avoid apex spike.
  // Keep displacement ≤ ~50% of face edge length to prevent face inversion (visible splits).
  // detail=2 edge ≈ 0.24u → safe max ≈ 0.12u = 16% radius. ±8% is conservative and safe.
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    const yNorm = y / len;                               // [-1, 1], 1=top, -1=base
    const yFactor = 0.3 + 0.7 * (1 - yNorm * yNorm);   // 1.0 at equator, 0.3 at poles
    const disp = rand() * 0.08 * yFactor;
    pos.setXYZ(i, x + (x / len) * disp, y + (y / len) * disp, z + (z / len) * disp);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();  // smooth: averages across shared indexed vertices

  // Spherical UV projection
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

  const mat = new THREE.MeshStandardMaterial({
    color: 0x7a7068,
    roughness: 0.92,
    metalness: 0.03,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
  return group;
}
