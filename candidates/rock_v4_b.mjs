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

  const mat = new THREE.MeshPhysicalMaterial({
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