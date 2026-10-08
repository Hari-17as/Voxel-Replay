import { SceneRenderer } from './renderer.js';

class VoxelReplayApp {
  constructor() {
    this.canvas = document.getElementById('viewport');
    this.renderer = new SceneRenderer(this.canvas);

    this.meshAnimation = null;
    this.v4dAnimation = null;

    this.isPlaying = true;
    this.currentFrameIdx = 0;
    this.lastFrameTime = performance.now();
    this.fpsCounter = 0;
    this.fpsTimer = performance.now();

    this.initUI();
    this.loadAssets();
    this.animate();
  }

  initUI() {
    this.playBtn = document.getElementById('btn-play');
    this.timeline = document.getElementById('timeline-slider');
    this.modeSelector = document.getElementById('mode-selector');
    this.lblFrame = document.getElementById('telemetry-frame');
    this.lblFps = document.getElementById('telemetry-fps');
    this.lblVerts = document.getElementById('telemetry-verts');
    this.lblTris = document.getElementById('telemetry-tris');
    this.lblStatus = document.getElementById('status-badge');

    this.playBtn.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      this.playBtn.textContent = this.isPlaying ? 'PAUSE' : 'PLAY';
    });

    this.timeline.addEventListener('input', (e) => {
      this.isPlaying = false;
      this.playBtn.textContent = 'PLAY';
      this.currentFrameIdx = parseInt(e.target.value, 10);
      this.displayCurrentFrame();
    });

    this.modeSelector.addEventListener('change', (e) => {
      this.renderer.setRenderMode(e.target.value);
    });
  }

  async loadAssets() {
    this.lblStatus.textContent = 'LOADING GEOMETRY...';
    try {
      const vmeshRes = await fetch('/reconstructed_mesh.vmesh');
      if (vmeshRes.ok) {
        const buffer = await vmeshRes.arrayBuffer();
        this.meshAnimation = this.parseVMesh(buffer);
        this.timeline.max = this.meshAnimation.frameCount - 1;
        this.lblStatus.textContent = 'ONLINE (MESH)';
      }
    } catch (err) {
      console.warn("VMesh load failed, attempting V4D fallback.", err);
    }

    try {
      const v4dRes = await fetch('/reconstructed_color.v4d');
      if (v4dRes.ok) {
        const buf = await v4dRes.arrayBuffer();
        this.v4dAnimation = this.parseV4D(buf);
      }
    } catch (err) {
      console.warn("V4D fallback load failed.", err);
    }

    if (!this.meshAnimation && !this.v4dAnimation) {
      this.lblStatus.textContent = 'NO ASSETS FOUND';
    } else {
      this.displayCurrentFrame();
    }
  }

  parseVMesh(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== 'VMES') throw new Error('Invalid VMESH magic identifier');

    const frameCount = view.getUint32(6, true);
    const fps = view.getFloat32(10, true);
    const boundsMin = [view.getFloat32(14, true), view.getFloat32(18, true), view.getFloat32(22, true)];
    const boundsMax = [view.getFloat32(26, true), view.getFloat32(30, true), view.getFloat32(34, true)];

    let offset = 38;
    const frames = [];

    for (let f = 0; f < frameCount; f++) {
      const timestamp = view.getFloat32(offset, true);
      const vertCount = view.getUint32(offset + 4, true);
      const indexCount = view.getUint32(offset + 8, true);
      offset += 12;

      let verts = new Float32Array(0);
      let normals = new Float32Array(0);
      let colors = new Uint8Array(0);
      let indices = new Uint32Array(0);

      if (vertCount > 0) {
        verts = new Float32Array(buffer, offset, vertCount * 3);
        offset += vertCount * 3 * 4;

        normals = new Float32Array(buffer, offset, vertCount * 3);
        offset += vertCount * 3 * 4;

        colors = new Uint8Array(buffer, offset, vertCount * 3);
        offset += vertCount * 3;

        indices = new Uint32Array(buffer, offset, indexCount);
        offset += indexCount * 4;
      }

      frames.push({ timestamp, verts, normals, colors, indices });
    }

    return { frameCount, fps, boundsMin, boundsMax, frames };
  }

  parseV4D(buffer) {
    const view = new DataView(buffer);
    const frameCount = view.getUint32(12, true);
    const fps = view.getFloat32(16, true);
    let offset = 24;
    const frames = [];

    for (let f = 0; f < frameCount; f++) {
      const ts = view.getFloat32(offset, true);
      const vCount = view.getUint32(offset + 4, true);
      offset += 8;

      const voxels = [];
      for (let i = 0; i < vCount; i++) {
        voxels.push({
          x: view.getUint8(offset),
          y: view.getUint8(offset + 1),
          z: view.getUint8(offset + 2),
          r: view.getUint8(offset + 3),
          g: view.getUint8(offset + 4),
          b: view.getUint8(offset + 5)
        });
        offset += 6;
      }
      frames.push({ timestamp: ts, voxels });
    }
    return { frameCount, fps, frames };
  }

  displayCurrentFrame() {
    this.timeline.value = this.currentFrameIdx;
    this.lblFrame.textContent = `${String(this.currentFrameIdx + 1).padStart(2, '0')}/${this.timeline.max + 1}`;

    if (this.meshAnimation && this.meshAnimation.frames[this.currentFrameIdx]) {
      const stats = this.renderer.updateMeshFrame(this.meshAnimation.frames[this.currentFrameIdx]);
      this.lblVerts.textContent = stats.vertices.toLocaleString();
      this.lblTris.textContent = stats.triangles.toLocaleString();
    }

    if (this.v4dAnimation && this.v4dAnimation.frames[this.currentFrameIdx]) {
      this.renderer.updateVoxelFrame(
        this.v4dAnimation.frames[this.currentFrameIdx].voxels,
        [-1.1, -1.1, 0.0],
        [1.1, 1.1, 2.2]
      );
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const now = performance.now();
    const targetFps = this.meshAnimation ? this.meshAnimation.fps : 15.0;
    const interval = 1000.0 / targetFps;

    if (this.isPlaying && this.meshAnimation) {
      if (now - this.lastFrameTime >= interval) {
        this.currentFrameIdx = (this.currentFrameIdx + 1) % this.meshAnimation.frameCount;
        this.displayCurrentFrame();
        this.lastFrameTime = now;
      }
    }

    // Telemetry FPS update
    this.fpsCounter++;
    if (now - this.fpsTimer >= 1000.0) {
      this.lblFps.textContent = `${this.fpsCounter} FPS`;
      this.fpsCounter = 0;
      this.fpsTimer = now;
    }

    this.renderer.render();
  }
}

window.addEventListener('DOMContentLoaded', () => {
  new VoxelReplayApp();
});