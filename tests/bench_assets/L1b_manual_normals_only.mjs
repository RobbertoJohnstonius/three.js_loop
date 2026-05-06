import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.IcosahedronGeometry(1, 0);   // 20 faces = 40 tris
  geo.computeVertexNormals();
  // Intentionally omit UVs
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.85 }));
}
