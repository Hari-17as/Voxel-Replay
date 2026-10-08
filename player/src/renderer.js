import * as THREE from 'three';

export class VoxelRenderer {

    constructor() {

        this.container =
            document.getElementById('viewport');

        if (!this.container) {
            throw new Error('Missing #viewport');
        }

        // ----------------------------------------------------
        // SCENE
        // ----------------------------------------------------

        this.scene = new THREE.Scene();

        this.scene.background =
            new THREE.Color(0x101116);

        // ----------------------------------------------------
        // CAMERA
        // ----------------------------------------------------

        this.camera =
            new THREE.PerspectiveCamera(
                45,
                1,
                0.01,
                100
            );

        this.camera.position.set(
            0,
            -4,
            1.4
        );

        // ----------------------------------------------------
        // WEBGL
        // ----------------------------------------------------

        this.renderer =
            new THREE.WebGLRenderer({
                antialias: true
            });

        this.renderer.setPixelRatio(
            Math.min(
                window.devicePixelRatio || 1,
                2
            )
        );

        this.renderer.outputColorSpace =
            THREE.SRGBColorSpace;

        this.renderer.shadowMap.enabled = true;

        this.container.innerHTML = '';

        this.container.appendChild(
            this.renderer.domElement
        );

        // ----------------------------------------------------
        // CHARACTER GROUP
        // ----------------------------------------------------

        this.group =
            new THREE.Group();

        this.scene.add(
            this.group
        );

        // ----------------------------------------------------
        // LIGHTING
        // ----------------------------------------------------

        const hemi =
            new THREE.HemisphereLight(
                0xfff4e6,
                0x303040,
                2.0
            );

        this.scene.add(
            hemi
        );

        const key =
            new THREE.DirectionalLight(
                0xffe8d0,
                3.0
            );

        key.position.set(
            3,
            -4,
            6
        );

        key.castShadow = true;

        this.scene.add(
            key
        );

        const fill =
            new THREE.DirectionalLight(
                0x9bbcff,
                1.0
            );

        fill.position.set(
            -4,
            2,
            3
        );

        this.scene.add(
            fill
        );

        // ----------------------------------------------------
        // CAMERA CONTROL
        // ----------------------------------------------------

        this.center =
            new THREE.Vector3(
                0,
                0,
                1.05
            );

        this.rotationX = 0;

        this.rotationY = 0;

        this.distance = 3.4;

        this.dragging = false;

        this.lastX = 0;

        this.lastY = 0;

        // ----------------------------------------------------
        // CURRENT CHARACTER
        // ----------------------------------------------------

        this.currentObject = null;

        this.currentFrame = null;

        this.currentHeader = null;

        // ----------------------------------------------------
        // CONTROLS
        // ----------------------------------------------------

        this.setupControls();

        this.resize();

        window.addEventListener(
            'resize',
            () => this.resize()
        );

        // ----------------------------------------------------
        // RENDER LOOP
        // ----------------------------------------------------

        this.animate();
    }


    // ========================================================
    // CONTROLS
    // ========================================================

    setupControls() {

        const canvas =
            this.renderer.domElement;

        canvas.addEventListener(
            'pointerdown',
            e => {

                this.dragging = true;

                this.lastX =
                    e.clientX;

                this.lastY =
                    e.clientY;

                canvas.setPointerCapture(
                    e.pointerId
                );
            }
        );

        canvas.addEventListener(
            'pointermove',
            e => {

                if (!this.dragging) {
                    return;
                }

                const dx =
                    e.clientX -
                    this.lastX;

                const dy =
                    e.clientY -
                    this.lastY;

                // Slow rotation.

                this.rotationY +=
                    dx * 0.0025;

                this.rotationX +=
                    dy * 0.0025;

                this.rotationX =
                    Math.max(
                        -1.2,
                        Math.min(
                            1.2,
                            this.rotationX
                        )
                    );

                this.lastX =
                    e.clientX;

                this.lastY =
                    e.clientY;
            }
        );

        canvas.addEventListener(
            'pointerup',
            e => {

                this.dragging = false;

                try {
                    canvas.releasePointerCapture(
                        e.pointerId
                    );
                } catch (_) {}
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
                    Math.exp(
                        e.deltaY * 0.001
                    );

                this.distance =
                    Math.max(
                        1.5,
                        Math.min(
                            10,
                            this.distance
                        )
                    );
            },
            {
                passive: false
            }
        );

        window.addEventListener(
            'keydown',
            e => {

                if (e.key === '0') {

                    this.rotationX = 0;

                    this.rotationY = 0;

                    this.distance = 3.4;
                }
            }
        );
    }


    // ========================================================
    // RESIZE
    // ========================================================

    resize() {

        const width =
            this.container.clientWidth ||
            800;

        const height =
            this.container.clientHeight ||
            600;

        this.camera.aspect =
            width / height;

        this.camera.updateProjectionMatrix();

        this.renderer.setSize(
            width,
            height,
            false
        );
    }


    // ========================================================
    // REMOVE CURRENT FRAME
    // ========================================================

    clear() {

        if (!this.currentObject) {
            return;
        }

        this.group.remove(
            this.currentObject
        );

        this.disposeObject(
            this.currentObject
        );

        this.currentObject = null;
    }


    // ========================================================
    // DISPOSE
    // ========================================================

    disposeObject(object) {

        object.traverse(
            child => {

                if (child.geometry) {

                    child.geometry.dispose();
                }

                if (child.material) {

                    if (
                        Array.isArray(
                            child.material
                        )
                    ) {

                        child.material.forEach(
                            material => {

                                material.dispose();
                            }
                        );

                    } else {

                        child.material.dispose();
                    }
                }
            }
        );
    }


    // ========================================================
    // COLOR
    // ========================================================

    getColor(
        r,
        g,
        b
    ) {

        let R =
            r / 255;

        let G =
            g / 255;

        let B =
            b / 255;

        // Keep actual colors.
        //
        // Only suppress nearly-white background
        // colours that sometimes enter the volume.

        const brightness =
            (R + G + B) / 3;

        const spread =
            Math.max(
                R,
                G,
                B
            )
            -
            Math.min(
                R,
                G,
                B
            );

        if (
            brightness > 0.94 &&
            spread < 0.06
        ) {

            R = 0.68;

            G = 0.50;

            B = 0.40;
        }

        return new THREE.Color(
            R,
            G,
            B
        );
    }


    // ========================================================
    // BUILD CURRENT FRAME
    // ========================================================

    setFrame(
        frame,
        header
    ) {

        // IMPORTANT:
        //
        // Completely remove the previous frame.
        //
        // Therefore the scene can contain
        // ONLY ONE CHARACTER at a time.

        this.clear();

        if (
            !frame ||
            !header
        ) {
            return;
        }

        const data =
            frame.data;

        const count =
            frame.voxelCount;

        if (
            !data ||
            !count
        ) {
            return;
        }

        const voxelSize =
            header.voxelSize;

        // ----------------------------------------------------
        // CREATE INSTANCED CUBES
        // ----------------------------------------------------

        const geometry =
            new THREE.BoxGeometry(
                voxelSize * 0.94,
                voxelSize * 0.94,
                voxelSize * 0.94
            );

        const material =
            new THREE.MeshStandardMaterial({
                vertexColors: false,
                roughness: 0.82,
                metalness: 0.02
            });

        const mesh =
            new THREE.InstancedMesh(
                geometry,
                material,
                count
            );

        mesh.instanceMatrix.setUsage(
            THREE.DynamicDrawUsage
        );

        mesh.castShadow = true;

        mesh.receiveShadow = true;

        // ----------------------------------------------------
        // POSITIONING
        // ----------------------------------------------------

        const gx =
            header.gridX;

        const gy =
            header.gridY;

        const temp =
            new THREE.Object3D();

        let minZ =
            Infinity;

        // First pass:
        // find lowest voxel.

        for (
            let i = 0;
            i < count;
            i++
        ) {

            const p =
                i * 6;

            const z =
                data[p + 2] *
                voxelSize;

            if (
                z < minZ
            ) {

                minZ = z;
            }
        }

        // ----------------------------------------------------
        // INSTANCES
        // ----------------------------------------------------

        for (
            let i = 0;
            i < count;
            i++
        ) {

            const p =
                i * 6;

            const x =
                data[p];

            const y =
                data[p + 1];

            const z =
                data[p + 2];

            // Center X/Y.
            //
            // Z starts at the floor.

            temp.position.set(

                (
                    x -
                    gx / 2
                )
                *
                voxelSize,

                (
                    y -
                    gy / 2
                )
                *
                voxelSize,

                (
                    z *
                    voxelSize
                )
                -
                minZ
            );

            temp.rotation.set(
                0,
                0,
                0
            );

            temp.scale.set(
                1,
                1,
                1
            );

            temp.updateMatrix();

            mesh.setMatrixAt(
                i,
                temp.matrix
            );

            // ------------------------------------------------
            // VOXEL COLOR
            // ------------------------------------------------

            const color =
                this.getColor(
                    data[p + 3],
                    data[p + 4],
                    data[p + 5]
                );

            mesh.setColorAt(
                i,
                color
            );
        }

        mesh.instanceMatrix.needsUpdate =
            true;

        if (
            mesh.instanceColor
        ) {

            mesh.instanceColor.needsUpdate =
                true;
        }

        // ----------------------------------------------------
        // ADD ONLY THIS CHARACTER
        // ----------------------------------------------------

        this.group.add(
            mesh
        );

        this.currentObject =
            mesh;

        this.currentFrame =
            frame;

        this.currentHeader =
            header;
    }


    // ========================================================
    // ANIMATION LOOP
    // ========================================================

    animate() {

        requestAnimationFrame(
            () => this.animate()
        );

        const radius =
            this.distance;

        const x =
            Math.sin(
                this.rotationY
            )
            *
            Math.cos(
                this.rotationX
            )
            *
            radius;

        const y =
            -Math.cos(
                this.rotationY
            )
            *
            Math.cos(
                this.rotationX
            )
            *
            radius;

        const z =
            Math.sin(
                this.rotationX
            )
            *
            radius;

        this.camera.position.set(

            this.center.x +
            x,

            this.center.y +
            y,

            this.center.z +
            z
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