import * as THREE from 'three';

export class VoxelRenderer {

    constructor() {

        this.container =
            document.getElementById('viewport');

        if (!this.container) {
            throw new Error('Missing #viewport');
        }

        this.scene = new THREE.Scene();
        this.scene.background =
            new THREE.Color(0x101116);

        this.camera =
            new THREE.PerspectiveCamera(
                45,
                1,
                0.01,
                100
            );

        this.camera.position.set(0, -4, 1.4);

        this.renderer =
            new THREE.WebGLRenderer({
                antialias: true
            });

        this.renderer.setPixelRatio(
            Math.min(window.devicePixelRatio || 1, 2)
        );

        this.renderer.outputColorSpace =
            THREE.SRGBColorSpace;

        this.container.innerHTML = '';
        this.container.appendChild(
            this.renderer.domElement
        );

        this.group = new THREE.Group();
        this.scene.add(this.group);

        const hemi =
            new THREE.HemisphereLight(
                0xfff4e6,
                0x303040,
                2.2
            );

        this.scene.add(hemi);

        const key =
            new THREE.DirectionalLight(
                0xffe8d0,
                3
            );

        key.position.set(3, -4, 6);
        this.scene.add(key);

        const fill =
            new THREE.DirectionalLight(
                0x9bbcff,
                1.2
            );

        fill.position.set(-4, 2, 3);
        this.scene.add(fill);

        this.center = new THREE.Vector3(0, 0, 1.0);

        this.rotationX = 0;
        this.rotationY = 0;
        this.distance = 3.2;

        this.dragging = false;
        this.lastX = 0;
        this.lastY = 0;

        this.setupControls();
        this.resize();

        window.addEventListener(
            'resize',
            () => this.resize()
        );

        this.animate();
    }

    setupControls() {

        const canvas =
            this.renderer.domElement;

        canvas.addEventListener(
            'pointerdown',
            e => {

                this.dragging = true;
                this.lastX = e.clientX;
                this.lastY = e.clientY;

                canvas.setPointerCapture(
                    e.pointerId
                );
            }
        );

        canvas.addEventListener(
            'pointermove',
            e => {

                if (!this.dragging) return;

                const dx =
                    e.clientX - this.lastX;

                const dy =
                    e.clientY - this.lastY;

                this.rotationY += dx * 0.0025;
                this.rotationX += dy * 0.008;

                this.rotationX =
                    Math.max(
                        -1.4,
                        Math.min(
                            1.4,
                            this.rotationX
                        )
                    );

                this.lastX = e.clientX;
                this.lastY = e.clientY;
            }
        );

        canvas.addEventListener(
            'pointerup',
            () => {
                this.dragging = false;
            }
        );

        canvas.addEventListener(
            'pointercancel',
            () => {
                this.dragging = false;
            }
        );

        canvas.addEventListener(
            'wheel',
            e => {

                e.preventDefault();

                this.distance *=
                    Math.exp(e.deltaY * 0.001);

                this.distance =
                    Math.max(
                        1.5,
                        Math.min(
                            10,
                            this.distance
                        )
                    );
            },
            { passive: false }
        );

        window.addEventListener(
            'keydown',
            e => {

                if (e.key === '0') {

                    this.rotationX = 0;
        this.rotationY = 0;
                    this.distance = 3.2;
                }
            }
        );
    }

    resize() {

        const width =
            this.container.clientWidth || 800;

        const height =
            this.container.clientHeight || 600;

        this.camera.aspect =
            width / height;

        this.camera.updateProjectionMatrix();

        this.renderer.setSize(
            width,
            height,
            false
        );
    }

    clear() {

        while (this.group.children.length) {

            const object =
                this.group.children.pop();

            if (object.geometry) {
                object.geometry.dispose();
            }

            if (object.material) {
                object.material.dispose();
            }
        }
    }

    getColor(r, g, b) {

        let R = r / 255;
        let G = g / 255;
        let B = b / 255;

        const brightness =
            (R + G + B) / 3;

        /*
         * Prevent accidental white/gray appearance.
         * Preserve strong clothing colors.
         */
        if (
            brightness > 0.92 &&
            Math.abs(R - G) < 0.08 &&
            Math.abs(G - B) < 0.08
        ) {

            R = 0.72;
            G = 0.52;
            B = 0.40;
        }

        /*
         * Slightly warm skin-like rendering
         * without forcing every voxel to skin color.
         */
        if (
            R > G * 1.08 &&
            G > B * 1.15 &&
            R > 0.35
        ) {

            R = Math.min(1, R * 1.05);
            G = Math.min(1, G * 0.96);
            B = Math.min(1, B * 0.90);
        }

        return new THREE.Color(R, G, B);
    }

    setFrame(frame, header) {

        this.clear();

        if (!frame || !header) return;

        const data = frame.data;
        const count = frame.voxelCount;

        if (!data || !count) return;

        const positions =
            new Float32Array(count * 3);

        const colors =
            new Float32Array(count * 3);

        const gx = header.gridX;
        const gy = header.gridY;
        const voxelSize = header.voxelSize;

        let minZ = Infinity;

        for (let i = 0; i < count; i++) {

            const p = i * 6;
            const q = i * 3;

            const x = data[p];
            const y = data[p + 1];
            const z = data[p + 2];

            positions[q] =
                (x - gx / 2) * voxelSize;

            positions[q + 1] =
                (y - gy / 2) * voxelSize;

            positions[q + 2] =
                z * voxelSize;

            minZ =
                Math.min(
                    minZ,
                    positions[q + 2]
                );

            const c =
                this.getColor(
                    data[p + 3],
                    data[p + 4],
                    data[p + 5]
                );

            colors[q] = c.r;
            colors[q + 1] = c.g;
            colors[q + 2] = c.b;
        }

        /*
         * Shift model so its feet sit near the floor.
         */
        for (let i = 0; i < count; i++) {

            positions[i * 3 + 2] -= minZ;
        }

        const geometry =
            new THREE.BufferGeometry();

        geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(
                positions,
                3
            )
        );

        geometry.setAttribute(
            'color',
            new THREE.BufferAttribute(
                colors,
                3
            )
        );

        /*
         * Larger overlapping points make the voxel cloud
         * visually continuous instead of separated dots.
         */
        const material =
            new THREE.PointsMaterial({
                size: voxelSize * 2.8,
                vertexColors: true,
                sizeAttenuation: true,
                transparent: false,
                alphaTest: 0.05
            });

        const points =
            new THREE.Points(
                geometry,
                material
            );

        this.group.add(points);
    }

    animate() {

        requestAnimationFrame(
            () => this.animate()
        );

        const r = this.distance;

        const x =
            Math.sin(this.rotationY) *
            Math.cos(this.rotationX) *
            r;

        const y =
            -Math.cos(this.rotationY) *
            Math.cos(this.rotationX) *
            r;

        const z =
            Math.sin(this.rotationX) *
            r;

        this.camera.position.set(
            this.center.x + x,
            this.center.y + y,
            this.center.z + z
        );

        this.camera.lookAt(
            this.center
        );

        this.renderer.render(
            this.scene,
            this.camera
        );
    }
}


