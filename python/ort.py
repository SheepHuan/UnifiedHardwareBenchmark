import numpy as np

from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema,ONNX_DOMAIN
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple


onnx_op_schemas:  Dict[int, List[OpSchema]] = defaultdict(
            list
            )

# 获取ONNX_DOMAIN中的op_schemas
# onnx_op_schemas = []
for schema in defs.get_all_schemas_with_history():
    if schema.domain == ONNX_DOMAIN:
        # print(f"{schema.domain}, {schema.support_level}, {schema.name}")
        onnx_op_schemas[int(schema.support_level)].append(schema)
print(onnx_op_schemas)

op_dict:  Dict[str, List[int]] = defaultdict(
            list
            )
for support,op_schemas in onnx_op_schemas.items():
    for op_schema in op_schemas:
        op_dict[op_schema.name].append(op_schema.since_version)
        # print(f"{op_schema.name}, {op_schema.since_version}")

print(op_dict)