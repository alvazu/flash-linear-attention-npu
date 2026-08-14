"""Deterministic ATK matrix generator for causal_conv1d_bwd."""

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
    "name": "causal_conv1d_bwd",
    "aclnn_name": "CausalConv1dBwd",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_causal_conv1d_bwd",
    "expected_error_msg": null,
    "backward": true,
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
        "range_values": "{\"B\":2,\"D\":64,\"T\":128,\"W\":4,\"activation\":0,\"case_key\":\"causal_conv1d_bwd_000_smoke\",\"cu_seqlens\":\"\",\"dtype\":\"bf16\",\"explicit_chunk_indices\":false,\"layout\":\"BSND\",\"route\":\"ascendc\",\"seed\":20260813,\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\"}",
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
        "range_values": 2,
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
        "name": "D",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 64,
        "backward": false
      },
      {
        "name": "W",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 4,
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
        "name": "layout",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "BSND",
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
        "name": "activation",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 0,
        "backward": false
      }
    ]
  },
  {
    "id": 1,
    "default_seed": 20260814,
    "name": "causal_conv1d_bwd",
    "aclnn_name": "CausalConv1dBwd",
    "version": "v2.1",
    "api": "pytorch",
    "api_type": "executor_causal_conv1d_bwd",
    "expected_error_msg": null,
    "backward": true,
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
        "range_values": "{\"B\":1,\"D\":128,\"T\":256,\"W\":4,\"activation\":1,\"case_key\":\"causal_conv1d_bwd_001_smoke\",\"cu_seqlens\":\"\",\"dtype\":\"fp16\",\"explicit_chunk_indices\":false,\"layout\":\"BSND\",\"route\":\"ascendc\",\"seed\":20260814,\"soc\":\"ascend910b\",\"tags\":\"accuracy,smoke\"}",
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
        "name": "T",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 256,
        "backward": false
      },
      {
        "name": "D",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 128,
        "backward": false
      },
      {
        "name": "W",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 4,
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
        "name": "layout",
        "type": "attr",
        "required": true,
        "dtype": "string",
        "shape": null,
        "range_values": "BSND",
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
        "name": "activation",
        "type": "attr",
        "required": true,
        "dtype": "int",
        "shape": null,
        "range_values": 1,
        "backward": false
      }
    ]
  }
]
''')

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_causal_conv1d_bwd")
    class CausalConv1dBwdGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            return case_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="atk_causal_conv1d_bwd.generated.json")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(CASES, indent=2) + "\n")
    if args.summary:
        print("causal_conv1d_bwd: " + str(len(CASES)) + " cases -> " + args.output)


if __name__ == "__main__":
    main()
