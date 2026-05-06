import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.CylinderGeometry(0.4, 0.6, 1.5, 16, 3),
    new THREE.MeshStandardMaterial({ color: 0x775533, roughness: 0.9 })
  );
}
