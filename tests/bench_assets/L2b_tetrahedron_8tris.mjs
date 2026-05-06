import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.TetrahedronGeometry(1, 0),
    new THREE.MeshStandardMaterial({ color: 0x776655, roughness: 0.9 })
  );
}
