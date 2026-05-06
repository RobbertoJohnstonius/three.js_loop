import * as THREE from 'three';

export function createAsset() {
  return new THREE.Group();   // no Mesh children — validator rejects
}
