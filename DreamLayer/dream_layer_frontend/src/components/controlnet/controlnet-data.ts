// Control types available in ControlNet
export const controlTypes = [
  { id: "all", label: "All" },
  { id: "canny", label: "Canny" },
  { id: "depth", label: "Depth" },
  { id: "ip-adapter", label: "IP-Adapter" },
  { id: "inpaint", label: "Inpaint" },
  { id: "instantid", label: "Instant-ID" },
  { id: "instructp2p", label: "InstructP2P" },
  { id: "lineart", label: "Lineart" },
  { id: "mlsd", label: "MLSD" },
  { id: "normalmap", label: "NormalMap" },
  { id: "openpose", label: "OpenPose" },
  { id: "recolor", label: "Recolor" },
  { id: "reference", label: "Reference" },
  { id: "revision", label: "Revision" },
  { id: "scribble", label: "Scribble" },
  { id: "segmentation", label: "Segmentation" },
  { id: "shuffle", label: "Shuffle" },
  { id: "softedge", label: "SoftEdge" },
  { id: "sparsectrl", label: "SparseCtrl" },
  { id: "t2i-adapter", label: "T2I-Adapter" },
  { id: "tile", label: "Tile" },
];

// Pre-processor models
export const preprocessors = [
  { id: "none", label: "None" },
  { id: "dw_openpose_full", label: "dw_openpose_full" },
  { id: "depth_midas", label: "depth_midas" },
  { id: "depth_zoe", label: "depth_zoe" },
  { id: "openpose_full", label: "openpose_full" },
  { id: "scribble_pidinet", label: "scribble_pidinet" },
  { id: "scribble_xdog", label: "scribble_xdog" },
  { id: "segmentation", label: "segmentation" },
  { id: "lineart_anime", label: "lineart_anime" },
  { id: "lineart_realistic", label: "lineart_realistic" },
  { id: "mlsd", label: "mlsd" },
  { id: "normal_map", label: "normal_map" },
  { id: "canny", label: "canny" },
];

// Preprocessors that support thresholds
export const preprocessorsWithThresholds = ["canny", "mlsd", "scribble_pidinet", "scribble_xdog"];

// Control modes
export const controlModes = [
  { id: "balanced", label: "Balanced" },
  { id: "prompt", label: "My prompt is more important" },
  { id: "control", label: "ControlNet is more important" },
];

// Resize modes
export const resizeModes = [
  { id: "just_resize", label: "Just Resize" },
  { id: "crop_resize", label: "Crop and Resize" },
  { id: "resize_fill", label: "Resize and Fill" },
];
