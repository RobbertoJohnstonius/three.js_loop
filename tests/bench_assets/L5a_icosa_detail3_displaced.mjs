import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.IcosahedronGeometry(1, 3);
  const pos = geo.attributes.position;

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const noise = Math.sin(x * 4.1) * 0.07 + Math.cos(y * 3.7) * 0.06 + Math.sin(z * 5.3) * 0.05;
    const scale = 1 + noise;
    pos.setXYZ(i, x * scale, y * scale, z * scale);
  }
  geo.computeVertexNormals();

  const uvs = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const len = Math.sqrt(x*x + y*y + z*z);
    uvs[i*2]   = 0.5 + Math.atan2(z/len, x/len) / (2 * Math.PI);
    uvs[i*2+1] = 0.5 - Math.asin(Math.max(-1, Math.min(1, y/len))) / Math.PI;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

  return new THREE.Mesh(geo,
    new THREE.MeshStandardMaterial({ color: 0x887766, roughness: 0.85, metalness: 0.05 })
  );
}
