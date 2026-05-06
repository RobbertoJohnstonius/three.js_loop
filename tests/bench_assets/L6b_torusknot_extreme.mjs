import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.7, 0.25, 200, 32),
    new THREE.MeshStandardMaterial({ color: 0x997766, roughness: 0.7, metalness: 0.1 })
  );
}
