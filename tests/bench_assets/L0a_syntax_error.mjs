import * as THREE from 'three';

export function createAsset( {   // missing ) — syntax error
  return new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial()
  );
}
