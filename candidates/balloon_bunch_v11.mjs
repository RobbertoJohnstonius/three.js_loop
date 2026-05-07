import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
    const radius = 0.15;
    const detail = 1;

    // 9 exact colors matching reference photo
    const balloonColors = [
        new THREE.Color(0x006400), // dark-green
        new THREE.Color(0x00aa00), // green
        new THREE.Color(0x2050d0), // blue
        new THREE.Color(0xe020a0), // pink
        new THREE.Color(0xe02020), // red
        new THREE.Color(0x8020c0), // purple
        new THREE.Color(0x20c0e0), // light-blue
        new THREE.Color(0xe0d020), // yellow
        new THREE.Color(0xe06010), // orange
    ];

    // Dome cluster — natural height variation 0.76→1.00 so all are visible from camera
    const positions = [
        [0.0,  1.00,  0.0 ],  // top center
        [0.22, 0.88,  0.0 ],  // right
        [-0.22,0.88,  0.0 ],  // left
        [0.0,  0.88,  0.22],  // front
        [0.0,  0.88, -0.22],  // back
        [0.18, 0.76,  0.18],  // right-front lower
        [-0.18,0.76,  0.18],  // left-front lower
        [0.18, 0.76, -0.18],  // right-back lower
        [-0.18,0.76, -0.18],  // left-back lower
    ];

    const group = new THREE.Group();

    const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.10,
        metalness: 0.0,
        vertexColors: true,
    });

    positions.forEach(([bx, by, bz], i) => {
        let geo = new THREE.IcosahedronGeometry(radius, detail);
        geo.deleteAttribute('uv');
        geo = mergeVertices(geo);

        // Slight Y stretch for teardrop shape
        const verts = geo.attributes.position.array;
        for (let v = 0; v < verts.length; v += 3) {
            const y = verts[v + 1];
            verts[v + 1] *= 1.2 + 0.15 * (y + radius) / (2 * radius);
        }
        geo.computeVertexNormals();

        // Spherical UVs
        const uvs = [];
        for (let v = 0; v < verts.length; v += 3) {
            uvs.push(
                0.5 + Math.atan2(verts[v + 2], verts[v]) / (2 * Math.PI),
                0.5 - Math.asin(Math.max(-1, Math.min(1, verts[v + 1] / radius))) / Math.PI,
            );
        }
        geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

        // Solid per-balloon vertex color
        const col = balloonColors[i];
        const cols = new Float32Array(verts.length);
        for (let v = 0; v < verts.length; v += 3) {
            cols[v] = col.r; cols[v + 1] = col.g; cols[v + 2] = col.b;
        }
        geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));

        const balloon = new THREE.Mesh(geo, material);
        balloon.position.set(bx, by, bz);
        balloon.name = 'balloon';
        group.add(balloon);

        // Stick: slightly converges toward gather point at y≈0.30
        // (looks like a bouquet held in hand, not a wild fan)
        const gx = bx * 0.15, gz = bz * 0.15;
        const pts = [
            new THREE.Vector3(bx, by - radius, bz),
            new THREE.Vector3(gx, 0.30, gz),
        ];
        const stick = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: 0xdddddd }),
        );
        stick.name = 'string';
        group.add(stick);
    });

    return group;
}
