import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.85, metalness: 0.05 })
  );
}
