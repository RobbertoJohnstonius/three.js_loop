import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.TorusGeometry(0.7, 0.3, 16, 32),
    new THREE.MeshStandardMaterial({ color: 0x996655, roughness: 0.7, metalness: 0.2 })
  );
}
