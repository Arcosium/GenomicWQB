"""Reject known GROUP inputs in numeric positions; allow unknown types."""
from __future__ import annotations

import re

from . import alpha_ast, datafield_palette

BUILTIN_GROUPS = frozenset({'country', 'currency', 'exchange',
                           'sector', 'industry', 'subindustry', 'market'})
_NUMERIC_FIRST = frozenset({
    'rank', 'zscore', 'normalize', 'scale', 'winsorize', 'hump',
    'abs', 'sign', 'log', 'sqrt', 'inverse', 'power', 'signed_power',
    'ts_rank', 'ts_zscore', 'ts_delta', 'ts_mean', 'ts_sum', 'ts_std_dev',
    'ts_av_diff', 'ts_backfill', 'ts_delay', 'ts_decay_linear',
    'ts_arg_max', 'ts_arg_min', 'ts_min', 'ts_max', 'ts_product',
    'group_rank', 'group_zscore', 'group_neutralize', 'group_scale',
    'group_mean', 'group_sum', 'group_backfill', 'group_std_dev',
})
_NUMERIC_PAIR = frozenset({'ts_corr', 'ts_covariance', 'ts_regression',
                          'vector_neut', 'regression_neut'})


def group_fields() -> frozenset:
    return BUILTIN_GROUPS | datafield_palette.group_field_names()


def numeric_input_reason(code: str, groups=None) -> str | None:
    groups = group_fields() if groups is None else groups
    variables = {}
    errors = []

    def infer(node):
        if not node:
            return None
        kind = node.get('type')
        if kind == 'name':
            name = node['value']
            return variables.get(name, 'group' if name in groups else None)
        if kind == 'call':
            op, args = node['name'], node.get('args') or []
            types = [infer(a) for a in args]
            positions = range(min(2, len(args))) if op in _NUMERIC_PAIR else (
                range(min(1, len(args))) if op in _NUMERIC_FIRST else ())
            for i in positions:
                if types[i] == 'group':
                    errors.append(f'group input to numeric operator {op} at argument {i + 1}')
            if op in ('densify', 'bucket', 'group_cartesian_product'):
                return 'group'
            return None
        if kind == 'unary':
            return infer(node.get('operand'))
        if kind == 'group':
            for child in node.get('children', []):
                infer(child)
        return None

    for statement in str(code or '').split(';'):
        match = re.match(r'^\s*([A-Za-z_]\w*)\s*=(?!=)(.*)$', statement, re.S)
        if match:
            variables[match[1].lower()] = infer(alpha_ast.parse(match[2]))
        else:
            infer(alpha_ast.parse(statement))
    return errors[0] if errors else None
