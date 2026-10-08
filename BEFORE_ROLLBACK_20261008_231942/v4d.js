const textDecoder = new TextDecoder();

function readHeader(view, offset = 0) {
  const magic = textDecoder.decode(
    new Uint8Array(view.buffer, view.byteOffset + offset, 4)
  );
  offset += 4;

  if (magic !== "V4D1") {
    throw new Error("Not a V4D1 file.");
  }

  const version = view.getUint16(offset, true);
  offset += 2;

  if (version !== 1) {
    throw new Error(`Unsupported V4D version: ${version}`);
  }

  const gridX = view.getUint16(offset, true); offset += 2;
  const gridY = view.getUint16(offset, true); offset += 2;
  const gridZ = view.getUint16(offset, true); offset += 2;
  const frameCount = view.getUint32(offset, true); offset += 4;
  const fps = view.getFloat32(offset, true); offset += 4;
  const voxelSize = view.getFloat32(offset, true); offset += 4;

  return {
    version,
    gridX,
    gridY,
    gridZ,
    frameCount,
    fps,
    voxelSize,
    offset
  };
}

export function decodeV4D(buffer) {
  const view = new DataView(buffer);
  const header = readHeader(view);

  let offset = header.offset;
  const frames = [];

  for (let frameIndex = 0; frameIndex < header.frameCount; frameIndex++) {
    const timestamp = view.getFloat32(offset, true);
    offset += 4;

    const voxelCount = view.getUint32(offset, true);
    offset += 4;

    const voxels = new Uint8Array(voxelCount * 6);

    for (let i = 0; i < voxels.length; i++) {
      voxels[i] = view.getUint8(offset++);
    }

    frames.push({
      timestamp,
      voxels
    });
  }

  return { header, frames };
}
