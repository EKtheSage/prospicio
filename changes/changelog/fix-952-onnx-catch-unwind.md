### Models

- `OnnxModel::from_bytes` returns an error, not a panic, for a malformed
  ONNX file that fails after the parse (a constant whose dims overflow) (#952, #1016).
