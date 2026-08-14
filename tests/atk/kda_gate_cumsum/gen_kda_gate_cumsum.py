"""Deterministic ATK matrix generator for kda_gate_cumsum."""

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
    "name": "kda_gate_cumsum",
    "aclnn_name": "KdaGateCumsum",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_kda_gate_cumsum",
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
        "range_values": "{\"B\":1,\"HV\":8,\"K\":128,\"T\":128,\"case_key\":\"kda_gate_cumsum_000_smoke\",\"chunk_size\":64,\"cu_seqlens\":\"\",\"explicit_chunk_indices\":false,\"g_dtype\":\"fp32\",\"route\":\"ascendc\",\"seed\":20260813,\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\",\"use_gate_in_kernel\":true,\"varlen\":false}",
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
        "name": "B",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 1,
        "backward": false
      },
      {
        "name": "HV",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 8,
        "backward": false
      },
      {
        "name": "T",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 128,
        "backward": false
      },
      {
        "name": "K",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 128,
        "backward": false
      },
      {
        "name": "g_dtype",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "fp32",
        "backward": false
      },
      {
        "name": "chunk_size",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 64,
        "backward": false
      },
      {
        "name": "varlen",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": false,
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
      },
      {
        "name": "use_gate_in_kernel",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": true,
        "backward": false
      }
    ]
  },
  {
    "id": 1,
    "default_seed": 20260814,
    "name": "kda_gate_cumsum",
    "aclnn_name": "KdaGateCumsum",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_kda_gate_cumsum",
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
        "range_values": "{\"B\":1,\"HV\":8,\"K\":128,\"T\":256,\"case_key\":\"kda_gate_cumsum_001_smoke\",\"chunk_size\":64,\"cu_seqlens\":\"0,128,256\",\"explicit_chunk_indices\":true,\"g_dtype\":\"bf16\",\"route\":\"ascendc\",\"seed\":20260814,\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\",\"use_gate_in_kernel\":true,\"varlen\":true}",
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
        "name": "B",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 1,
        "backward": false
      },
      {
        "name": "HV",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 8,
        "backward": false
      },
      {
        "name": "T",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 256,
        "backward": false
      },
      {
        "name": "K",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 128,
        "backward": false
      },
      {
        "name": "g_dtype",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "bf16",
        "backward": false
      },
      {
        "name": "chunk_size",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 64,
        "backward": false
      },
      {
        "name": "varlen",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": true,
        "backward": false
      },
      {
        "name": "cu_seqlens",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "0,128,256",
        "backward": false
      },
      {
        "name": "explicit_chunk_indices",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": true,
        "backward": false
      },
      {
        "name": "use_gate_in_kernel",
        "type": "attr",
        "required": true,
        "dtype": "bool",
        "shape": null,
        "range_values": true,
        "backward": false
      }
    ]
  }
]
''')

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_kda_gate_cumsum")
    class KdaGateCumsumGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return case_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="atk_kda_gate_cumsum.generated.json")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(CASES, indent=2) + "\n")
    if args.summary:
        print("kda_gate_cumsum: " + str(len(CASES)) + " cases -> " + args.output)


if __name__ == "__main__":
    main()
