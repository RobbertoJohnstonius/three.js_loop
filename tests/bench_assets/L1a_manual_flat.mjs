import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.BufferGeometry();
  const verts = new Float32Array([
    -0.5, -0.5, 0,   0.5, -0.5, 0,   0.5, 0.5, 0,
    -0.5, -0.5, 0,   0.5,  0.5, 0,  -0.5, 0.5, 0,
  ]);
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.8 }));
}
