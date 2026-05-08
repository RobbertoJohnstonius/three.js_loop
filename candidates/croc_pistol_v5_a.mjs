import * as THREE from 'three';

export function createAsset() {
    const group = new THREE.Object3D();

    function hash(cx, cy, cz) {
        const h = Math.sin(cx * 127.1 + cy * 311.7 + cz * 74.7)
                + Math.sin(cx * 269.5 + cy * 183.3 + cz * 246.1)
                + Math.sin(cx * 113.5 + cy * 271.9 + cz * 124.6);
        return (h / 3.0 + 1.0) * 0.5;
    }

    function posNoise(geo, scale) {
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
            pos.setXYZ(i,
                x + Math.sin(x * 127.1 + y * 311.7 + z *  74.7) * scale,
                y + Math.sin(x * 269.5 + y * 183.3 + z * 246.1) * scale,
                z + Math.sin(x * 113.5 + y * 271.9 + z * 124.6) * scale
            );
        }
        pos.needsUpdate = true;
    }

    function applyFaceColors(geo, palette) {
        const ng = geo.toNonIndexed();
        ng.computeVertexNormals();
        const pos = ng.attributes.position;
        const nFaces = Math.floor(pos.count / 3);
        const colors = new Float32Array(pos.count * 3);
        for (let f = 0; f < nFaces; f++) {
            const i0 = f * 3, i1 = f * 3 + 1, i2 = f * 3 + 2;
            const cx = (pos.getX(i0) + pos.getX(i1) + pos.getX(i2)) / 3;
            const cy = (pos.getY(i0) + pos.getY(i1) + pos.getY(i2)) / 3;
            const cz = (pos.getZ(i0) + pos.getZ(i1) + pos.getZ(i2)) / 3;
            const t = hash(cx, cy, cz);
            const [r, g, b] = palette(t);
            for (const idx of [i0, i1, i2]) {
                colors[idx * 3] = r; colors[idx * 3 + 1] = g; colors[idx * 3 + 2] = b;
            }
        }
        ng.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        return ng;
    }

    // ── Body / slide — 6×6×6 subdivisions → 216 tris ─────────────────────────
    let bodyGeo = new THREE.BoxGeometry(1.56, 0.42, 0.44, 6, 6, 6);
    posNoise(bodyGeo, 0.020);
    bodyGeo = applyFaceColors(bodyGeo, t => [
        0.29 + t * 0.12,  // teal-jade R
        0.56 + t * 0.24,  // teal-jade G  (vivid — hue 145°)
        0.44 + t * 0.14,  // teal-jade B
    ]);
    const bodyMesh = new THREE.Mesh(bodyGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.22, metalness: 0.60
    }));
    bodyMesh.name = 'body';
    bodyMesh.position.set(0, 0.28, 0);
    group.add(bodyMesh);

    // ── Gold ornamental rail — 6×12×6 subdivisions → 432 tris ────────────────
    let railGeo = new THREE.BoxGeometry(1.58, 0.12, 0.46, 6, 12, 6);
    posNoise(railGeo, 0.014);
    railGeo = applyFaceColors(railGeo, t => [
        0.80 + t * 0.12,  // gold R
        0.64 + t * 0.10,  // gold G  (hue 45°)
        0.16 + t * 0.07,  // gold B
    ]);
    const railMesh = new THREE.Mesh(railGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.10, metalness: 0.92
    }));
    railMesh.name = 'rail';
    railMesh.position.set(0, 0.55, 0);
    group.add(railMesh);

    // ── Grip — 6×12×6 subdivisions → 216 tris, GREEN dominant scale ──────────
    let gripGeo = new THREE.BoxGeometry(0.50, 0.64, 0.43, 6, 12, 6);
    gripGeo.rotateZ(-Math.PI / 12);
    posNoise(gripGeo, 0.016);
    gripGeo = applyFaceColors(gripGeo, t => {
        if (t < 0.70) {
            const b2 = t / 0.70;
            return [0.25 + b2 * 0.10, 0.46 + b2 * 0.18, 0.30 + b2 * 0.14];  // teal-green
        } else if (t < 0.88) {
            const b2 = (t - 0.70) / 0.18;
            return [0.22 + b2 * 0.08, 0.44 + b2 * 0.04, 0.46 + b2 * 0.16];  // blue-green
        } else {
            const b2 = (t - 0.88) / 0.12;
            return [0.36 + b2 * 0.10, 0.25 + b2 * 0.04, 0.48 + b2 * 0.08];  // subtle purple
        }
    });
    const gripMesh = new THREE.Mesh(gripGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.35, metalness: 0.10
    }));
    gripMesh.name = 'grip';
    gripMesh.position.set(-0.28, -0.22, 0);
    group.add(gripMesh);

    // ── Guard — TALL visible orange arc below body/grip junction ─────────────
    let guardGeo = new THREE.BoxGeometry(0.46, 0.1, 0.44, 3, 1, 1);
    posNoise(guardGeo, 0.013);
    guardGeo = applyFaceColors(guardGeo, t => [
        0.82 + t * 0.06,  // vivid orange R  (hue ~25°)
        0.41 + t * 0.08,  // orange G
        0.12 + t * 0.04,  // orange B
    ]);
    const guardMesh = new THREE.Mesh(guardGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.15, metalness: 0.70
    }));
    guardMesh.name = 'guard';
    guardMesh.position.set(-0.06, -0.2, 0);
    group.add(guardMesh);

    // ── Barrel — SHORT FAT stub, rotated to +X axis ───────────────────────────
    let barrelGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.22, 10, 1);
    barrelGeo.rotateZ(Math.PI / 2);
    posNoise(barrelGeo, 0.016);
    barrelGeo = applyFaceColors(barrelGeo, t => [
        0.30 + t * 0.06,
        0.39 + t * 0.07,
        0.43 + t * 0.07,
    ]);
    const barrelMesh = new THREE.Mesh(barrelGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.25, metalness: 0.65
    }));
    barrelMesh.name = 'barrel';
    barrelMesh.position.set(0.89, 0.34, 0);
    group.add(barrelMesh);

    // ── Muzzle ring — larger disc at barrel tip, adds silhouette detail ───────
    let muzzleGeo = new THREE.CylinderGeometry(0.17, 0.14, 0.06, 8, 1);
    muzzleGeo.rotateZ(Math.PI / 2);
    posNoise(muzzleGeo, 0.012);
    muzzleGeo = applyFaceColors(muzzleGeo, t => [0.28 + t * 0.06, 0.36 + t * 0.06, 0.40 + t * 0.06]);
    const muzzleMesh = new THREE.Mesh(muzzleGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.28, metalness: 0.60
    }));
    muzzleMesh.name = 'barrel';
    muzzleMesh.position.set(1.00, 0.34, 0);
    group.add(muzzleMesh);

    // ── Trigger — thin bar inside the guard arc ───────────────────────────────
    let triggerGeo = new THREE.BoxGeometry(0.06, 0.18, 0.04, 1, 3, 1);
    posNoise(triggerGeo, 0.008);
    triggerGeo = applyFaceColors(triggerGeo, t => [0.24 + t * 0.06, 0.22 + t * 0.05, 0.20 + t * 0.05]);
    const triggerMesh = new THREE.Mesh(triggerGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.30, metalness: 0.50
    }));
    triggerMesh.name = 'trigger';
    triggerMesh.position.set(-0.04, -0.14, 0);
    group.add(triggerMesh);

    // ── Hammer / rear sight ───────────────────────────────────────────────────
    let hammerGeo = new THREE.BoxGeometry(0.14, 0.12, 0.44, 2, 1, 1);
    posNoise(hammerGeo, 0.011);
    hammerGeo = applyFaceColors(hammerGeo, t => [0.22 + t * 0.06, 0.36 + t * 0.08, 0.30 + t * 0.07]);
    const hammerMesh = new THREE.Mesh(hammerGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.28, metalness: 0.55
    }));
    hammerMesh.name = 'hammer';
    hammerMesh.position.set(-0.69, 0.54, 0);
    group.add(hammerMesh);

    // ── Orange Guard Element ───────────────────────────────────────────────────
    let orangeGuardGeo = new THREE.BoxGeometry(0.46, 0.1, 0.44, 3, 1, 1);
    posNoise(orangeGuardGeo, 0.013);
    orangeGuardGeo = applyFaceColors(orangeGuardGeo, t => [
        0.82 + t * 0.06,  // vivid orange R  (hue ~25°)
        0.41 + t * 0.08,  // orange G
        0.12 + t * 0.04,  // orange B
    ]);
    const orangeGuardMesh = new THREE.Mesh(orangeGuardGeo, new THREE.MeshStandardMaterial({
        color: 0xffffff, vertexColors: true, roughness: 0.18, metalness: 0.75
    }));
    orangeGuardMesh.name = 'orangeGuard';
    orangeGuardMesh.position.set(-0.06, -0.3, 0);
    group.add(orangeGuardMesh);

    return group;
}