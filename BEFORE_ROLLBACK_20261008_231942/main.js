import { VoxelRenderer } from './renderer.js';

const $ = (id) => document.getElementById(id);

const status = $('status');
const timeline = $('timeline');
const play = $('playPause');
const timeLabel = $('timeLabel');
const stats = $('stats');

let data = null;
let viewer = null;
let current = 0;
let playing = false;
let last = performance.now();
let acc = 0;

async function decodeV4D() {
    const url = `/reconstructed_color.v4d?cache=${Date.now()}`;

    const response = await fetch(url, {
        cache: 'no-store'
    });

    if (!response.ok) {
        throw new Error(`V4D HTTP ${response.status}`);
    }

    const buffer = await response.arrayBuffer();

    if (buffer.byteLength < 32) {
        throw new Error('V4D file is too small');
    }

    const view = new DataView(buffer);

    const magic = String.fromCharCode(
        view.getUint8(0),
        view.getUint8(1),
        view.getUint8(2),
        view.getUint8(3)
    );

    if (magic !== 'V4D1') {
        throw new Error(`Invalid V4D magic: ${magic}`);
    }

    let offset = 4;

    const version = view.getUint16(offset, true);
    offset += 2;

    if (version !== 1) {
        throw new Error(`Unsupported V4D version: ${version}`);
    }

    const grid_x = view.getUint16(offset, true);
    offset += 2;

    const grid_y = view.getUint16(offset, true);
    offset += 2;

    const grid_z = view.getUint16(offset, true);
    offset += 2;

    const frame_count = view.getUint32(offset, true);
    offset += 4;

    const fps = view.getFloat32(offset, true);
    offset += 4;

    const voxel_size = view.getFloat32(offset, true);
    offset += 4;

    const frames = [];

    for (let i = 0; i < frame_count; i++) {

        if (offset + 8 > buffer.byteLength) {
            throw new Error(`Unexpected end of V4D at frame ${i}`);
        }

        const timestamp = view.getFloat32(offset, true);
        offset += 4;

        const voxel_count = view.getUint32(offset, true);
        offset += 4;

        const byte_count = voxel_count * 6;

        if (offset + byte_count > buffer.byteLength) {
            throw new Error(`Invalid voxel data at frame ${i}`);
        }

        const voxels = new Uint8Array(
            buffer,
            offset,
            byte_count
        );

        offset += byte_count;

        frames.push({
            timestamp,
            voxels
        });
    }

    console.log('V4D LOADED');
    console.log({
        grid_x,
        grid_y,
        grid_z,
        frame_count,
        fps,
        voxel_size,
        fileBytes: buffer.byteLength
    });

    return {
        header: {
            grid_x,
            grid_y,
            grid_z,
            frame_count,
            fps,
            voxel_size
        },
        frames
    };
}

function showFrame(index) {

    if (!data || !data.frames.length) {
        return;
    }

    current = Math.max(
        0,
        Math.min(data.frames.length - 1, index)
    );

    const frame = data.frames[current];

    viewer.setFrame(
        frame.voxels,
        data.header
    );

    timeline.value = current;

    const duration =
        data.frames[data.frames.length - 1].timestamp;

    timeLabel.textContent =
        `${frame.timestamp.toFixed(2)} / ${duration.toFixed(2)} s`;

    stats.textContent =
        `Frame ${current + 1}/${data.frames.length} • ` +
        `${Math.floor(frame.voxels.length / 6).toLocaleString()} voxels • ` +
        `${data.header.fps.toFixed(2)} FPS`;
}

async function loadV4D() {

    try {

        status.textContent = 'LOADING 4D HUMAN...';

        data = await decodeV4D();

        if (!viewer) {
            viewer = new VoxelRenderer(
                $('viewport')
            );
        }

        timeline.min = 0;
        timeline.max = data.frames.length - 1;
        timeline.step = 1;
        timeline.value = 0;

        showFrame(0);

        status.textContent =
            `4D HUMAN READY • ${data.frames.length} FRAMES`;

    } catch (error) {

        console.error('V4D LOAD ERROR:', error);

        status.textContent =
            `ERROR: ${error.message}`;

        stats.textContent =
            'Could not load reconstructed_color.v4d';

    }
}

play.onclick = () => {

    if (!data) {
        return;
    }

    playing = !playing;

    play.textContent =
        playing ? '⏸ PAUSE' : '▶ PLAY';
};

timeline.oninput = (event) => {

    if (!data) {
        return;
    }

    showFrame(
        Number(event.target.value)
    );
};

function animationLoop(now) {

    const dt = now - last;
    last = now;

    if (
        playing &&
        data &&
        data.frames.length
    ) {

        acc += dt;

        const interval =
            1000 / data.header.fps;

        if (acc >= interval) {

            acc = 0;

            const next =
                current + 1 >= data.frames.length
                    ? 0
                    : current + 1;

            showFrame(next);
        }
    }

    requestAnimationFrame(animationLoop);
}

requestAnimationFrame(animationLoop);

loadV4D();