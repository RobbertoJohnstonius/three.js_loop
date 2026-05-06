import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.8, metalness: 0.1 })
  );
}
