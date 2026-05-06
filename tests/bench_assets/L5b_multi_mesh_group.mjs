import * as THREE from 'three';

export function createAsset() {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x776655, roughness: 0.9, metalness: 0.0 });

  const configs = [
    [0.8, 3, [0, 0, 0]],
    [0.5, 2, [0.6, 0.1, 0.3]],
    [0.4, 2, [-0.5, 0.15, -0.3]],
  ];

  for (const [radius, detail, pos] of configs) {
    const geo = new THREE.IcosahedronGeometry(radius, detail);
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
      const noise = Math.sin(x * 5) * 0.08 + Math.cos(y * 4) * 0.06;
      positions.setXYZ(i, x * (1 + noise), y * (1 + noise), z * (1 + noise));
    }
    geo.computeVertexNormals();
    const uvs = new Float32Array(positions.count * 2);
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
      const len = Math.sqrt(x*x + y*y + z*z);
      uvs[i*2]   = 0.5 + Math.atan2(z/len, x/len) / (2 * Math.PI);
      uvs[i*2+1] = 0.5 - Math.asin(Math.max(-1, Math.min(1, y/len))) / Math.PI;
    }
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...pos);
    group.add(mesh);
  }
  return group;
}
