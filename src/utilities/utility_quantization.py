def quantize_feature_map(feature_map, num_bits=8):
    # Define range for quantization based on num_bits
    quantization_levels = 2 ** num_bits - 1  # 255 for 8-bit quantization

    min_val = feature_map.min()
    max_val = feature_map.max()

    # Avoid divide by zero
    scale = (max_val - min_val) / quantization_levels if max_val != min_val else 1.0

    quantized_map = ((feature_map - min_val) / scale).round().clamp(0, quantization_levels).byte()

    return quantized_map, min_val, scale