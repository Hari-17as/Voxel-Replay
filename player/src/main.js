import { VoxelRenderer } from './renderer.js';

const viewer = new VoxelRenderer(
    document.getElementById('viewer')
);

let frames = [];
let header = null;
let currentFrame = 0;
let playing = false;
let lastTime = 0;

const status =
    document.getElementById('status');

const timeLabel =
    document.getElementById('timeLabel');

const timeline =
    document.getElementById('timeline');

const stats =
    document.getElementById('stats');

function setStatus(text) {
    if (status) status.textContent = text;
}

function decodeV4D(buffer) {
    const dv = new DataView(buffer);
    const bytes = new Uint8Array(buffer);

    const magic =
        String.fromCharCode(
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3]
        );

    if (magic !== 'V4D1') {
        throw new Error('Invalid V4D file');
    }

    let p = 4;

    const version = dv.getUint16(p, true);
    p += 2;

    const gridX = dv.getUint16(p, true);
    p += 2;

    const gridY = dv.getUint16(p, true);
    p += 2;

    const gridZ = dv.getUint16(p, true);
    p += 2;

    const frameCount = dv.getUint32(p, true);
    p += 4;

    const fps = dv.getFloat32(p, true);
    p += 4;

    const voxelSize = dv.getFloat32(p, true);
    p += 4;

    const decodedFrames = [];

    for (let f = 0; f < frameCount; f++) {

        const timestamp =
            dv.getFloat32(p, true);

        p += 4;

        const voxelCount =
            dv.getUint32(p, true);

        p += 4;

        const data =
            new Uint8Array(
                buffer,
                p,
                voxelCount * 6
            );

        p += voxelCount * 6;

        decodedFrames.push({
            timestamp,
            voxelCount,
            data
        });
    }

    return {
        version,
        gridX,
        gridY,
        gridZ,
        frameCount,
        fps,
        voxelSize,
        frames: decodedFrames
    };
}

function showFrame(index) {

    if (!frames.length) return;

    index = Math.max(
        0,
        Math.min(index, frames.length - 1)
    );

    currentFrame = index;

    viewer.setFrame(
        frames[index],
        header
    );

    const t =
        frames[index].timestamp;

    if (timeLabel) {
        timeLabel.textContent =
            `${t.toFixed(2)} s`;
    }

    if (timeline) {
        timeline.value = index;
    }

    if (stats) {
        stats.textContent =
            `Frame ${index + 1}/${frames.length} • ` +
            `${frames[index].voxelCount.toLocaleString()} voxels`;
    }
}

async function loadV4D() {

    try {

        setStatus('LOADING 4D HUMAN...');

        const response =
            await fetch(
                `/reconstructed_color.v4d?cache=${Date.now()}`,
                {
                    cache: 'no-store'
                }
            );

        if (!response.ok) {
            throw new Error(
                `V4D HTTP ${response.status}`
            );
        }

        const buffer =
            await response.arrayBuffer();

        header =
            decodeV4D(buffer);

        frames = header.frames;

        console.log('V4D LOADED');
        console.log(header);

        if (timeline) {
            timeline.min = 0;
            timeline.max =
                Math.max(0, frames.length - 1);
            timeline.step = 1;
            timeline.value = 0;
        }

        showFrame(0);

        setStatus(
            `4D HUMAN READY • ${frames.length} FRAMES`
        );

    } catch (error) {

        console.error(error);

        setStatus(
            `ERROR: ${error.message}`
        );
    }
}

if (timeline) {

    timeline.addEventListener(
        'input',
        () => {
            showFrame(
                Number(timeline.value)
            );
        }
    );
}

const playButton =
    document.getElementById('playPause') ||
    document.getElementById('play');

if (playButton) {

    playButton.addEventListener(
        'click',
        () => {

            playing = !playing;

            playButton.textContent =
                playing ? '⏸ PAUSE' : '▶ PLAY';

            lastTime = performance.now();
        }
    );
}

function animationLoop(now) {

    if (
        playing &&
        frames.length > 0
    ) {

        const frameDuration =
            1000 / (header?.fps || 12);

        if (
            now - lastTime >= frameDuration
        ) {

            const steps =
                Math.floor(
                    (now - lastTime) /
                    frameDuration
                );

            currentFrame =
                (currentFrame + steps) %
                frames.length;

            showFrame(currentFrame);

            lastTime = now;
        }
    }

    requestAnimationFrame(animationLoop);
}

requestAnimationFrame(animationLoop);

loadV4D();

