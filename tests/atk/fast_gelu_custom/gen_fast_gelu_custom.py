"""Deterministic ATK matrix generator for fast_gelu_custom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None

CASES = json.loads(r'''
[
  {
    "id": 0,
    "default_seed": 20260813,
    "name": "fast_gelu_custom",
    "aclnn_name": "FastGelu",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_fast_gelu_custom",
    "expected_error_msg": null,
    "backward": false,
    "standard": {
      "acc": {
        "cv_fused_double_benchmark": {
          "max_re_ratio": 5,
          "avg_re_ratio": 1.5,
          "root_mean_squared_ratio": 1.5
        }
      },
      "perf": "not_key"
    },
    "outputs": null,
    "inputs": [
      {
        "name": "low_precision_marker",
        "type": "tensor",
        "required": true,
        "dtype": "bf16",
        "shape": [
          1
        ],
        "range_values": [
          0,
          0
        ],
        "backward": false
      },
      {
        "name": "fp32_marker",
        "type": "tensor",
        "required": true,
        "dtype": "fp32",
        "shape": [
          1
        ],
        "range_values": [
          0,
          0
        ],
        "backward": false
      },
      {
        "name": "case_spec",
        "type": "attr",
        "required": true,
        "dtype": "non_param",
        "shape": null,
        "range_values": "{\"case_key\":\"fast_gelu_custom_000_vector_fp32\",\"cu_seqlens\":\"\",\"dtype\":\"fp32\",\"explicit_chunk_indices\":false,\"route\":\"ascendc\",\"seed\":20260813,\"shape\":[1024],\"shape_key\":\"vector_fp32\",\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\"}",
        "backward": false
      },
      {
        "name": "soc",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "ascend910b",
        "backward": false
      },
      {
        "name": "route",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "ascendc",
        "backward": false
      },
      {
        "name": "shape_key",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "vector_fp32",
        "backward": false
      },
      {
        "name": "dtype",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "fp32",
        "backward": false
      },
      {
        "name": "cu_seqlens",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "",
        "backward": false
      },
      {
        "name": "explicit_chunk_indices",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": false,
        "backward": false
      }
    ]
  },
  {
    "id": 1,
    "default_seed": 20260814,
    "name": "fast_gelu_custom",
    "aclnn_name": "FastGelu",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_fast_gelu_custom",
    "expected_error_msg": null,
    "backward": false,
    "standard": {
      "acc": {
        "cv_fused_double_benchmark": {
          "max_re_ratio": 5,
          "avg_re_ratio": 1.5,
          "root_mean_squared_ratio": 1.5
        }
      },
      "perf": "not_key"
    },
    "outputs": null,
    "inputs": [
      {
        "name": "low_precision_marker",
        "type": "tensor",
        "required": true,
        "dtype": "bf16",
        "shape": [
          1
        ],
        "range_values": [
          0,
          0
        ],
        "backward": false
      },
      {
        "name": "fp32_marker",
        "type": "tensor",
        "required": true,
        "dtype": "fp32",
        "shape": [
          1
        ],
        "range_values": [
          0,
          0
        ],
        "backward": false
      },
      {
        "name": "case_spec",
        "type": "attr",
        "required": true,
        "dtype": "non_param",
        "shape": null,
        "range_values": "{\"case_key\":\"fast_gelu_custom_001_matrix_bf16\",\"cu_seqlens\":\"\",\"dtype\":\"bf16\",\"explicit_chunk_indices\":false,\"route\":\"ascendc\",\"seed\":20260814,\"shape\":[8,256],\"shape_key\":\"matrix_bf16\",\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\"}",
        "backward": false
      },
      {
        "name": "soc",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "ascend910b",
        "backward": false
      },
      {
        "name": "route",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "ascendc",
        "backward": false
      },
      {
        "name": "shape_key",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "matrix_bf16",
        "backward": false
      },
      {
        "name": "dtype",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "bf16",
        "backward": false
      },
      {
        "name": "cu_seqlens",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "",
        "backward": false
      },
      {
        "name": "explicit_chunk_indices",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": false,
        "backward": false
      }
    ]
  },
  {
    "id": 2,
    "default_seed": 20260815,
    "name": "fast_gelu_custom",
    "aclnn_name": "FastGelu",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_fast_gelu_custom",
    "expected_error_msg": null,
    "backward": false,
    "standard": {
      "acc": {
        "cv_fused_double_benchmark": {
          "max_re_ratio": 5,
          "avg_re_ratio": 1.5,
          "root_mean_squared_ratio": 1.5
        }
      },
      "perf": "not_key"
    },
    "outputs": null,
    "inputs": [
      {
        "name": "low_precision_marker",
        "type": "tensor",
        "required": true,
        "dtype": "bf16",
        "shape": [
          1
        ],
        "range_values": [
          0,
          0
        ],
        "backward": false
      },
      {
        "name": "fp32_marker",
        "type": "tensor",
        "required": true,
        "dtype": "fp32",
        "shape": [
          1
        ],
        "range_values": [
          0,
          0
        ],
        "backward": false
      },
      {
        "name": "case_spec",
        "type": "attr",
        "required": true,
        "dtype": "non_param",
        "shape": null,
        "range_values": "{\"case_key\":\"fast_gelu_custom_002_matrix_fp16\",\"cu_seqlens\":\"\",\"dtype\":\"fp16\",\"explicit_chunk_indices\":false,\"route\":\"ascendc\",\"seed\":20260815,\"shape\":[4,512],\"shape_key\":\"matrix_fp16\",\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\"}",
        "backward": false
      },
      {
        "name": "soc",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "ascend910b",
        "backward": false
      },
      {
        "name": "route",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "ascendc",
        "backward": false
      },
      {
        "name": "shape_key",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "matrix_fp16",
        "backward": false
      },
      {
        "name": "dtype",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "fp16",
        "backward": false
      },
      {
        "name": "cu_seqlens",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "",
        "backward": false
      },
      {
        "name": "explicit_chunk_indices",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": false,
        "backward": false
      }
    ]
  }
]
''')

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_fast_gelu_custom")
    class FastGeluCustomGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return case_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="atk_fast_gelu_custom.generated.json")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(CASES, indent=2) + "\n")
    if args.summary:
        print("fast_gelu_custom: " + str(len(CASES)) + " cases -> " + args.output)


if __name__ == "__main__":
    main()
