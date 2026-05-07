import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
    const radius = 0.15;
    const detail = 1;
    const balloonColors = [
        new THREE.Color(0x006400), // #006400 green
        new THREE.Color(0x00aa00), // #00aa00 green
        new THREE.Color(0x2050d0), // #2050d0 blue
        new THREE.Color(0xe020a0), // #e020a0 magenta
        new THREE.Color(0xe02020), // #e02020 red
        new THREE.Color(0x8020c0), // #8020c0 purple
        new THREE.Color(0x20c0e0), // #20c0e0 teal
        new THREE.Color(0xe0d020), // #e0d020 yellow
        new THREE.Color(0xe06010)  // #e06010 orange
    ];

    const group = new THREE.Group();

    const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.10,
        metalness: 0.0,
        vertexColors: true
    });

    const createBalloon = (color, position, uniformY) => {
        let geometry = new THREE.IcosahedronGeometry(radius, detail);
        geometry.deleteAttribute('uv');
        geometry = mergeVertices(geometry);

        // Displace vertices slightly to create a teardrop shape
        const vertices = geometry.attributes.position.array;
        for (let i = 0; i < vertices.length; i += 3) {
            const y = vertices[i + 1];
            const scale = 1.2 + (0.15 * (y + radius) / (2 * radius));
            vertices[i + 1] *= scale;
        }
        geometry.computeVertexNormals();

        // Set UV coordinates
        const uvs = [];
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];
            const u = 0.5 + Math.atan2(z, x) / (2 * Math.PI);
            const v = 0.5 - Math.asin(y) / Math.PI;
            uvs.push(u, v);
        }
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

        // Set vertex colors
        const colors = new Float32Array(vertices.length);
        for (let i = 0; i < vertices.length; i += 3) {
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const balloon = new THREE.Mesh(geometry, material);
        balloon.position.set(position[0], uniformY, position[2]);
        balloon.name = 'balloon';
        group.add(balloon);

        // Create and add the string
        const pts = [new THREE.Vector3(position[0], uniformY - 0.15, position[2]), new THREE.Vector3(position[0], -0.55, position[2])];
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffffff }));
        line.name = 'string';
        group.add(line);
    };

    const positions = [
        [0.0, 0.82, 0.0],
        [0.22, 0.80, 0.12],
        [-0.22, 0.78, 0.12],
        [0.12, 0.88, -0.18],
        [-0.12, 0.72, -0.18],
        [0.28, 0.84, 0.0],
        [-0.28, 0.76, 0.0],
        [0.0, 0.90, 0.22],
        [0.0, 0.68, -0.22]
    ];

    // Calculate the average y-coordinate
    const averageY = positions.reduce((sum, pos) => sum + pos[1], 0) / positions.length;

    for (let i = 0; i < 9; i++) {
        createBalloon(balloonColors[i], positions[i], averageY);
    }

    return group;
}