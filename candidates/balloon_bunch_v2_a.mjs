import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export function createAsset() {
    const radius = 0.15;
    const detail = 2;
    const balloonColors = [
        new THREE.Color(0xff0000), // red
        new THREE.Color(0xffa500), // orange
        new THREE.Color(0xffff00), // yellow
        new THREE.Color(0x008000), // green
        new THREE.Color(0x008080), // teal
        new THREE.Color(0x0000ff), // blue
        new THREE.Color(0x800080), // purple
        new THREE.Color(0xff00ff), // magenta
        new THREE.Color(0xffffff), // white
        new THREE.Color(0x006400)  // dark-green
    ];

    const group = new THREE.Group();

    const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.15,
        metalness: 0.0,
        vertexColors: true
    });

    const createBalloon = (color, position) => {
        const geometry = new THREE.IcosahedronGeometry(radius, detail);
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
        balloon.position.set(...position);
        balloon.name = 'balloon';
        group.add(balloon);

        // Create and add the string
        const stringGeometry = new THREE.CylinderGeometry(0.01, 0.01, 1.5, 4, 1, false);
        const stringMaterial = new THREE.MeshStandardMaterial({ color: 0x808080, roughness: 0.75, metalness: 0.0 });
        const string = new THREE.Mesh(stringGeometry, stringMaterial);
        string.position.set(0, -radius - 0.75, 0);
        string.rotation.x = -Math.PI / 2;
        balloon.add(string);
    };

    const positions = [
        [0.0, 1.0, 0.0],
        [0.7, 0.7, -0.5],
        [-0.7, 0.7, 0.5],
        [0.4, 0.4, 0.8],
        [-0.4, 0.4, -0.8],
        [0.8, 0.1, 0.4],
        [-0.8, 0.1, -0.4],
        [0.2, 0.1, -0.2],
        [-0.2, 0.1, 0.2],
        [0.6, 0.5, 0.2],
        [-0.6, 0.5, -0.2],
        [0.0, 0.8, 0.0]
    ];

    for (let i = 0; i < 12; i++) {
        const color = balloonColors[i % balloonColors.length];
        const position = positions[i];
        createBalloon(color, position);
    }

    return group;
}