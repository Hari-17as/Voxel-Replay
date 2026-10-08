import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class SceneRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.currentMode = 'mesh'; // 'mesh' or 'voxel'

    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c0d10);

    // Camera setup
    const aspect = canvas.clientWidth / canvas.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 50.0);
    this.camera.position.set(0.0, -3.2, 1.4);

    // WebGL setup
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0.0, 0.0, 1.05);
    this.controls.maxPolarAngle = Math.PI / 2 + 0.1; // Restrict camera going below floor

    this.setupLighting();
    this.setupGroundGrid();

    // Render nodes
    this.meshNode = null;
    this.pointsNode = null;
    this.geometry = new THREE.BufferGeometry();

    // Standard Smooth Shaded Human Material
    this.meshMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.65,
      metalness: 0.1,
      side: THREE.DoubleSide
    });

    // Voxel Point Cloud Material
    this.pointsMaterial = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: true,
      sizeAttenuation: true
    });

    window.addEventListener('resize', () => this.onWindowResize());
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xfff5ea, 1.4);
    keyLight.position.set(2.5, -3.0, 4.0);
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xbad7ff, 0.9);
    fillLight.position.set(-3.0, 2.0, 2.5);
    this.scene.add(fillLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.7);
    backLight.position.set(0.0, 4.0, 1.5);
    this.scene.add(backLight);
  }

  setupGroundGrid() {
    const grid = new THREE.GridHelper(6.0, 24, 0x00f0ff, 0x1f2430);
    grid.position.z = 0.0;
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);
  }

  setRenderMode(mode) {
    this.currentMode = mode;
    if (this.meshNode) this.meshNode.visible = (mode === 'mesh');
    if (this.pointsNode) this.pointsNode.visible = (mode === 'voxel');
  }

  updateMeshFrame(frameData) {
    if (!frameData || frameData.verts.length === 0) {
      if (this.meshNode) this.meshNode.visible = false;
      return { vertices: 0, triangles: 0 };
    }

    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry();

    this.geometry.setAttribute('position', new THREE.BufferAttribute(frameData.verts, 3));
    this.geometry.setAttribute('normal', new THREE.BufferAttribute(frameData.normals, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(frameData.colors, 3, true));
    this.geometry.setIndex(new THREE.BufferAttribute(frameData.indices, 1));

    if (!this.meshNode) {
      this.meshNode = new THREE.Mesh(this.geometry, this.meshMaterial);
      this.scene.add(this.meshNode);
    } else {
      this.meshNode.geometry = this.geometry;
    }

    this.meshNode.visible = (this.currentMode === 'mesh');
    return {
      vertices: frameData.verts.length / 3,
      triangles: frameData.indices.length / 3
    };
  }

  updateVoxelFrame(voxelData, boundsMin, boundsMax) {
    if (!voxelData || voxelData.length === 0) {
      if (this.pointsNode) this.pointsNode.visible = false;
      return;
    }

    const count = voxelData.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    const sx = (boundsMax[0] - boundsMin[0]) / 127.0;
    const sy = (boundsMax[1] - boundsMin[1]) / 127.0;
    const sz = (boundsMax[2] - boundsMin[2]) / 191.0;

    for (let i = 0; i < count; i++) {
      const v = voxelData[i];
      positions[i * 3 + 0] = boundsMin[0] + v.x * sx;
      positions[i * 3 + 1] = boundsMin[1] + v.y * sy;
      positions[i * 3 + 2] = boundsMin[2] + v.z * sz;

      colors[i * 3 + 0] = v.r / 255.0;
      colors[i * 3 + 1] = v.g / 255.0;
      colors[i * 3 + 2] = v.b / 255.0;
    }

    const voxelGeo = new THREE.BufferGeometry();
    voxelGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    voxelGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    if (!this.pointsNode) {
      this.pointsNode = new THREE.Points(voxelGeo, this.pointsMaterial);
      this.scene.add(this.pointsNode);
    } else {
      this.pointsNode.geometry.dispose();
      this.pointsNode.geometry = voxelGeo;
    }

    this.pointsNode.visible = (this.currentMode === 'voxel');
  }

  onWindowResize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}