#!/usr/bin/env python3
"""Mini NPU Simulator.

AI 가속기(NPU)의 핵심 연산인 MAC(Multiply-Accumulate)을 반복문으로 직접 구현하고,
입력 패턴이 Cross(십자가)인지 X인지 판정하는 콘솔 프로그램입니다.

- 모드 1: 3×3 필터 2개(A, B)와 패턴을 직접 입력해 판정
- 모드 2: data.json의 필터/패턴을 읽어 일괄 판정 + PASS/FAIL + 성능 분석 + 결과 요약
- 모드 3 (보너스): N×N Cross/X 패턴 자동 생성기 → 판정과 성능 분석에 재활용
- 모드 4 (보너스): 2차원 배열 vs 1차원(평탄화) 배열 MAC 성능 비교

외부 라이브러리 없이 표준 라이브러리(json, time 등)만 사용합니다. Python 3.8 이상.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ============================================================
# 1. 상수 / 정책
# ============================================================

EPSILON = 1e-9          # 점수 비교 허용오차: |a - b| < EPSILON 이면 동점
REPEAT = 10             # 성능 측정 반복 횟수 (과제 최소 기준 10회)
BONUS_REPEAT = 100      # 보너스 2D/1D 비교 반복 횟수 (두 방식에 똑같이 적용, 잡음을 줄이려고 10회보다 크게)
BENCH_SIZES = (3, 5, 13, 25)              # 필수 성능 분석 크기
BONUS_SIZES = (3, 5, 13, 25, 50, 100)     # 보너스 최적화 비교 크기
MAX_GENERATOR_SIZE = 200                  # 생성기에서 받을 수 있는 최대 N

LABEL_CROSS = "Cross"   # 프로그램 내부 표준 라벨
LABEL_X = "X"
UNDECIDED = "UNDECIDED"

# 외부 표기(data.json의 expected 값, filters의 키) → 표준 라벨
# 비교 전에 strip() + lower() 하므로 "Cross", " X " 같은 표기도 같은 라벨이 됩니다.
LABEL_ALIASES: Dict[str, str] = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")   # 예: size_13_2 → N=13, idx=2
FILTER_KEY_RE = re.compile(r"^size_(\d+)$")          # 예: size_13  → N=13

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

SECTION_LINE = "#" + "-" * 39

# 실패 사유 분류 (README의 "데이터/스키마 vs 로직 vs 수치 비교" 진단 기준)
REASON_SCHEMA = "스키마"      # 키 형식, 필터 없음, 크기 불일치, 알 수 없는 라벨 등 데이터 문제
REASON_TIE = "동점"           # |Cross - X| < EPSILON → UNDECIDED (수치 비교 정책)
REASON_MISMATCH = "판정 불일치"  # 판정은 났지만 expected와 다름 (로직 또는 데이터 라벨 문제)


class InputAborted(Exception):
    """사용자가 Ctrl+C 또는 Ctrl+D(EOF)로 입력을 끝냈을 때 사용합니다."""


# ============================================================
# 2. 데이터 구조: n×n 행렬
# ============================================================

class Matrix:
    """n×n 크기의 2차원 패턴/필터를 저장하는 정사각 행렬.

    내부적으로 리스트의 리스트(rows)로 값을 보관하고,
    get/set으로 특정 위치의 값을 읽고 쓸 수 있습니다.
    """

    def __init__(self, size: int, fill: float = 0.0) -> None:
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"행렬 크기는 1 이상의 정수여야 합니다: {size!r}")
        self.size = size
        self._rows: List[List[float]] = [[float(fill)] * size for _ in range(size)]

    @classmethod
    def from_rows(cls, rows: object, expected_size: Optional[int] = None) -> "Matrix":
        """2차원 리스트를 검증한 뒤 Matrix로 만듭니다. 문제가 있으면 ValueError."""
        validate_square_rows(rows, expected_size)
        matrix = cls(len(rows))  # type: ignore[arg-type]
        for r, row in enumerate(rows):  # type: ignore[arg-type]
            for c, value in enumerate(row):
                matrix.set(r, c, value)
        return matrix

    def get(self, row: int, col: int) -> float:
        return self._rows[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        self._rows[row][col] = float(value)

    def rows(self) -> List[List[float]]:
        """MAC 연산용 내부 행 리스트 (복사하지 않으므로 읽기 전용으로 사용)."""
        return self._rows

    def to_flat(self) -> List[float]:
        """행 우선(row-major) 순서로 펼친 길이 N² 1차원 리스트. index = row * N + col."""
        return [value for row in self._rows for value in row]

    def format(self, indent: str = "  ") -> str:
        lines = []
        for r in range(self.size):
            lines.append(indent + " ".join(f"{self.get(r, c):g}" for c in range(self.size)))
        return "\n".join(lines)


def _is_number(value: object) -> bool:
    # bool은 int의 하위 타입이라 True/False가 숫자로 통과하지 않도록 따로 막습니다.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def validate_square_rows(rows: object, expected_size: Optional[int] = None) -> None:
    """rows가 '숫자만 들어 있는 정사각 2차원 리스트'인지 검사합니다."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("2차원 배열(리스트의 리스트)이 아니거나 비어 있습니다")
    n = len(rows)
    if expected_size is not None and n != expected_size:
        raise ValueError(f"행 수 불일치: {expected_size}행이어야 하는데 {n}행입니다")
    for r, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            raise ValueError(f"{r}번째 행이 리스트가 아닙니다")
        if len(row) != n:
            raise ValueError(f"열 수 불일치: {r}번째 행이 {len(row)}열입니다 ({n}×{n} 정사각 행렬이어야 함)")
        for c, value in enumerate(row, start=1):
            if not _is_number(value):
                raise ValueError(f"숫자가 아닌 값: ({r}행, {c}열) = {value!r}")


# ============================================================
# 3. MAC 연산 / 판정 / 시간 측정
# ============================================================

def mac(pattern: Matrix, filt: Matrix) -> float:
    """2차원 MAC: 같은 위치끼리 곱해서(Multiply) 모두 더한다(Accumulate).

    N×N이면 곱셈 N²번 + 덧셈 N²번 → 시간 복잡도 O(N²).
    """
    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, 필터 {filt.size}×{filt.size}")
    n = pattern.size
    p_rows = pattern.rows()
    f_rows = filt.rows()
    total = 0.0
    for i in range(n):
        p_row = p_rows[i]
        f_row = f_rows[i]
        for j in range(n):
            total += p_row[j] * f_row[j]
    return total


def mac_flat(pattern_flat: Sequence[float], filter_flat: Sequence[float]) -> float:
    """1차원(평탄화) MAC (보너스). 인덱스 하나(k)로 N²개를 순서대로 훑습니다.

    2차원 MAC과 같은 순서(행 우선)로 더하므로 결과가 비트 단위까지 같습니다.
    """
    if len(pattern_flat) != len(filter_flat):
        raise ValueError(f"길이 불일치: 패턴 {len(pattern_flat)}, 필터 {len(filter_flat)}")
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


def judge(score_a: float, score_b: float, epsilon: float = EPSILON) -> str:
    """두 점수를 허용오차 기반으로 비교해 'A' / 'B' / UNDECIDED 를 돌려줍니다.

    부동소수점은 0.1 같은 수를 정확히 표현하지 못해 0.9 와 0.8999999999999999 처럼
    '사실상 같은 값'이 다르게 계산될 수 있으므로 == 나 > 를 그대로 쓰지 않습니다.
    """
    if abs(score_a - score_b) < epsilon:
        return UNDECIDED
    return "A" if score_a > score_b else "B"


def format_epsilon(epsilon: float) -> str:
    """화면 표시용: 1e-09 → 1e-9"""
    return f"{epsilon:g}".replace("e-0", "e-")


def measure_ms(func: Callable[..., object], *args: object, repeat: int = REPEAT) -> float:
    """func(*args)를 repeat번 호출한 1회 평균 시간(ms). 입출력은 측정 구간에 넣지 않습니다."""
    if repeat < 1:
        raise ValueError("repeat는 1 이상이어야 합니다")
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


# ============================================================
# 4. 라벨 정규화
# ============================================================

def normalize_label(raw: object) -> Optional[str]:
    """'+', 'cross', 'x' 등 외부 표기를 표준 라벨(Cross/X)로 바꿉니다. 모르면 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ============================================================
# 5. 패턴 생성기 (보너스)
# ============================================================

def _center_indices(n: int) -> Tuple[int, ...]:
    # 홀수 N은 가운데 1줄, 짝수 N은 정확한 가운데가 없으므로 가운데 2줄을 사용
    if n % 2 == 1:
        return (n // 2,)
    return (n // 2 - 1, n // 2)


def make_cross(n: int, value: float = 1.0) -> Matrix:
    """N×N 십자가(Cross) 패턴: 가운데 행과 가운데 열이 value."""
    matrix = Matrix(n)
    centers = _center_indices(n)
    for r in range(n):
        for c in range(n):
            if r in centers or c in centers:
                matrix.set(r, c, value)
    return matrix


def make_x(n: int, value: float = 1.0) -> Matrix:
    """N×N X 패턴: 두 대각선(r == c, r + c == N-1)이 value."""
    matrix = Matrix(n)
    for r in range(n):
        for c in range(n):
            if r == c or r + c == n - 1:
                matrix.set(r, c, value)
    return matrix


def make_pattern(label: str, n: int) -> Matrix:
    return make_cross(n) if label == LABEL_CROSS else make_x(n)


# ============================================================
# 6. data.json 로드 / 스키마 검증 / 평가
# ============================================================

@dataclass
class FilterLoad:
    """filters.size_N 하나를 읽은 결과."""
    key: str
    size: Optional[int]
    filters: Dict[str, Matrix]   # 표준 라벨 → 필터
    error: Optional[str] = None  # 사용할 수 없으면 사유

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class CaseResult:
    """patterns의 케이스 하나를 판정한 결과."""
    case_id: str
    passed: bool
    cross_score: Optional[float] = None
    x_score: Optional[float] = None
    decision: Optional[str] = None
    expected: Optional[str] = None
    category: Optional[str] = None   # 실패 분류 (REASON_*)
    reason: Optional[str] = None     # 실패 사유 (사람이 읽는 문장)


class DataError(Exception):
    """data.json을 아예 사용할 수 없을 때 (파일 없음, JSON 문법 오류, 최상위 구조 오류)."""


def load_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        raise DataError(f"파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError as exc:
        raise DataError(f"JSON 문법 오류 ({exc.lineno}행 {exc.colno}열): {exc.msg}")
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"파일을 읽을 수 없습니다: {exc}")
    if not isinstance(data, dict):
        raise DataError("최상위 구조가 객체({ })가 아닙니다")
    for section in ("filters", "patterns"):
        if not isinstance(data.get(section), dict):
            raise DataError(f"'{section}' 항목이 없거나 객체({{ }})가 아닙니다")
    return data


def load_filters(raw_filters: Dict[str, object]) -> Dict[int, FilterLoad]:
    """filters 섹션을 읽어 {N: FilterLoad}로 돌려줍니다. 문제가 있어도 예외 없이 error에 기록."""
    loaded: Dict[int, FilterLoad] = {}
    for key, group in raw_filters.items():
        match = FILTER_KEY_RE.match(key)
        if not match:
            # 크기를 알 수 없는 키는 패턴과 연결할 수 없으므로 음수 임시 키로 보관(출력용)
            loaded[-(len(loaded) + 1)] = FilterLoad(key, None, {}, f"필터 키 형식 오류 (size_N 형식이어야 함)")
            continue
        n = int(match.group(1))
        load = FilterLoad(key, n, {})
        loaded[n] = load
        if not isinstance(group, dict):
            load.error = "필터 묶음이 객체({ })가 아닙니다"
            continue
        for raw_label, rows in group.items():
            label = normalize_label(raw_label)
            if label is None:
                load.error = f"알 수 없는 필터 라벨 '{raw_label}'"
                break
            if label in load.filters:
                load.error = f"'{raw_label}' 라벨이 중복되었습니다 (정규화 후 {label})"
                break
            try:
                load.filters[label] = Matrix.from_rows(rows, expected_size=n)
            except ValueError as exc:
                load.error = f"{raw_label} 필터: {exc}"
                break
        if load.error is None:
            missing = [lab for lab in (LABEL_CROSS, LABEL_X) if lab not in load.filters]
            if missing:
                load.error = f"필터 누락: {', '.join(missing)}"
    return loaded


def evaluate_case(case_id: str, case: object, filters: Dict[int, FilterLoad],
                  epsilon: float = EPSILON) -> CaseResult:
    """patterns의 한 항목을 검증 → MAC → 판정 → PASS/FAIL 까지 처리합니다. 예외를 밖으로 던지지 않습니다."""
    def fail(category: str, reason: str, **kw: object) -> CaseResult:
        return CaseResult(case_id, False, category=category, reason=reason, **kw)  # type: ignore[arg-type]

    match = PATTERN_KEY_RE.match(case_id)
    if not match:
        return fail(REASON_SCHEMA, "패턴 키 형식 오류 (size_{N}_{idx} 형식이어야 함)")
    n = int(match.group(1))

    if not isinstance(case, dict):
        return fail(REASON_SCHEMA, "패턴 항목이 객체({ })가 아닙니다")
    if "input" not in case or "expected" not in case:
        return fail(REASON_SCHEMA, "input 또는 expected 필드가 없습니다")

    expected = normalize_label(case["expected"])
    if expected is None:
        return fail(REASON_SCHEMA, f"알 수 없는 expected 라벨 {case['expected']!r}")

    load = filters.get(n)
    if load is None:
        return fail(REASON_SCHEMA, f"size_{n} 필터가 없습니다", expected=expected)
    if not load.ok:
        return fail(REASON_SCHEMA, f"size_{n} 필터를 사용할 수 없습니다 ({load.error})", expected=expected)

    try:
        pattern = Matrix.from_rows(case["input"])
    except ValueError as exc:
        return fail(REASON_SCHEMA, f"input 형식 오류: {exc}", expected=expected)
    if pattern.size != n:
        return fail(REASON_SCHEMA,
                    f"크기 불일치: 키는 size_{n}인데 input은 {pattern.size}×{pattern.size}입니다",
                    expected=expected)

    cross_score = mac(pattern, load.filters[LABEL_CROSS])
    x_score = mac(pattern, load.filters[LABEL_X])
    winner = judge(cross_score, x_score, epsilon)
    decision = {"A": LABEL_CROSS, "B": LABEL_X}.get(winner, UNDECIDED)
    result = CaseResult(case_id, decision == expected, cross_score, x_score, decision, expected)
    if decision == UNDECIDED:
        result.category = REASON_TIE
        result.reason = (f"동점(UNDECIDED) 처리 규칙에 따라 FAIL "
                         f"(|Cross-X| = {abs(cross_score - x_score):.3g} < {format_epsilon(epsilon)})")
    elif not result.passed:
        result.category = REASON_MISMATCH
        result.reason = f"판정 {decision} ≠ expected {expected}"
    return result


def evaluate_patterns(raw_patterns: Dict[str, object], filters: Dict[int, FilterLoad],
                      epsilon: float = EPSILON) -> List[CaseResult]:
    return [evaluate_case(case_id, case, filters, epsilon) for case_id, case in raw_patterns.items()]


# ============================================================
# 7. 성능 분석
# ============================================================

@dataclass
class BenchRow:
    size: int
    avg_ms: float
    source: str

    @property
    def ops(self) -> int:
        return self.size * self.size


def benchmark(inputs: Sequence[Tuple[int, Matrix, Matrix, str]], repeat: int = REPEAT) -> List[BenchRow]:
    """(N, 패턴, 필터, 출처) 목록마다 MAC 1회 평균 시간을 잽니다."""
    rows = []
    for size, pattern, filt, source in inputs:
        mac(pattern, filt)  # 첫 호출 준비 비용이 측정에 섞이지 않도록 1회 예열
        rows.append(BenchRow(size, measure_ms(mac, pattern, filt, repeat=repeat), source))
    return rows


def print_bench_table(rows: Sequence[BenchRow], repeat: int = REPEAT) -> None:
    print(pad("크기", 9) + pad("평균 시간(ms)", 15, ">") + pad("연산 횟수(N²)", 16, ">")
          + pad("N²당 시간(ns)", 16, ">") + "  입력")
    print("-" * 70)
    for row in rows:
        per_op_ns = row.avg_ms * 1e6 / row.ops
        print(pad(f"{row.size}×{row.size}", 9) + pad(f"{row.avg_ms:.4f}", 15, ">") + pad(row.ops, 16, ">")
              + pad(f"{per_op_ns:.1f}", 16, ">") + "  " + row.source)
    print(f"(각 크기별 MAC {repeat}회 반복 측정 평균, 입출력 시간 제외)")


@dataclass
class CompareRow:
    size: int
    ms_2d: float
    ms_1d: float
    same_result: bool


def compare_2d_vs_1d(sizes: Sequence[int] = BONUS_SIZES, repeat: int = BONUS_REPEAT) -> List[CompareRow]:
    """보너스: 같은 입력(생성기 Cross 패턴 × Cross 필터), 같은 반복 횟수로 2D와 1D를 비교."""
    rows = []
    for n in sizes:
        pattern, filt = make_cross(n), make_cross(n)
        p_flat, f_flat = pattern.to_flat(), filt.to_flat()   # 평탄화는 측정 전에 한 번만
        mac(pattern, filt)
        mac_flat(p_flat, f_flat)
        ms_2d = measure_ms(mac, pattern, filt, repeat=repeat)
        ms_1d = measure_ms(mac_flat, p_flat, f_flat, repeat=repeat)
        rows.append(CompareRow(n, ms_2d, ms_1d, mac(pattern, filt) == mac_flat(p_flat, f_flat)))
    return rows


# ============================================================
# 8. 콘솔 입력 (모드 1)
# ============================================================

InputFunc = Callable[[str], str]


def safe_input(input_fn: InputFunc, prompt: str = "") -> str:
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        raise InputAborted()


def parse_row(line: str, n: int) -> List[float]:
    """공백으로 구분된 숫자 n개를 float 리스트로 바꿉니다. 형식이 틀리면 ValueError."""
    tokens = line.split()
    if not tokens:
        raise ValueError("빈 줄입니다(행 누락).")
    if len(tokens) != n:
        raise ValueError(f"숫자가 {len(tokens)}개입니다(열 수 불일치).")
    values = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"'{token}'은(는) 숫자가 아닙니다(숫자 파싱 실패).")
        if not math.isfinite(value):
            raise ValueError(f"'{token}'은(는) 유한한 숫자가 아닙니다.")
        values.append(value)
    return values


def read_matrix(title: str, n: int, input_fn: InputFunc = input) -> Matrix:
    """n줄을 한 줄씩 입력받습니다. 틀린 줄은 안내 후 그 줄만 다시 입력받습니다.

    행 수는 '올바른 줄이 n개 모일 때까지' 받으므로 부족/초과가 생기지 않고,
    빈 줄은 행 누락으로 보고 다시 받습니다.
    """
    print(f"{title} ({n}줄 입력, 공백 구분)")
    rows: List[List[float]] = []
    while len(rows) < n:
        line = safe_input(input_fn)
        try:
            rows.append(parse_row(line, n))
        except ValueError as exc:
            print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. ({exc})")
            print(f"→ {len(rows) + 1}번째 줄을 다시 입력하세요.")
    return Matrix.from_rows(rows, expected_size=n)


def read_int(prompt: str, low: int, high: int, input_fn: InputFunc = input) -> int:
    while True:
        text = safe_input(input_fn, prompt).strip()
        try:
            value = int(text)
        except ValueError:
            print(f"숫자를 입력하세요. ({low}~{high})")
            continue
        if low <= value <= high:
            return value
        print(f"{low}~{high} 사이의 숫자를 입력하세요.")


# ============================================================
# 9. 화면 출력 도우미
# ============================================================

def display_width(text: str) -> int:
    """터미널에서 보이는 폭. 한글 등 전각 문자는 2칸으로 셉니다."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: object, width: int, align: str = "<") -> str:
    """표 정렬용: 보이는 폭 기준으로 공백을 채웁니다 (align: '<' 왼쪽, '>' 오른쪽)."""
    text = str(text)
    space = " " * max(0, width - display_width(text))
    return text + space if align == "<" else space + text


def section(title: str) -> None:
    print()
    print(SECTION_LINE)
    print(f"# {title}")
    print(SECTION_LINE)


def print_scores_and_decision(name_a: str, score_a: float, name_b: str, score_b: float,
                              decision_text: str) -> None:
    print(f"{name_a} 점수: {score_a!r}")
    print(f"{name_b} 점수: {score_b!r}")
    print(f"점수 차이 |{name_a}-{name_b}|: {abs(score_a - score_b)!r}")
    print(f"판정: {decision_text}")


def decision_text_ab(winner: str) -> str:
    if winner == UNDECIDED:
        return f"판정 불가 (|A-B| < {format_epsilon(EPSILON)})"
    return winner


# ============================================================
# 10. 모드 1: 사용자 입력 (3×3)
# ============================================================

def run_user_mode(input_fn: InputFunc = input, n: int = 3) -> None:
    section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", n, input_fn)
    print()
    filter_b = read_matrix("필터 B", n, input_fn)
    print()
    print("✓ 필터 저장 완료 — 저장된 값 확인")
    print("필터 A:")
    print(filter_a.format())
    print("필터 B:")
    print(filter_b.format())

    section("[2] 패턴 입력")
    pattern = read_matrix("패턴", n, input_fn)

    report_ab_judgement(pattern, filter_a, filter_b)

    section(f"[4] 성능 분석 ({n}×{n}, 평균/{REPEAT}회)")
    print_bench_table(benchmark([(n, pattern, filter_a, "사용자 입력")]))


def report_ab_judgement(pattern: Matrix, filter_a: Matrix, filter_b: Matrix) -> str:
    """패턴 × 필터 A/B MAC → 판정 결과 출력. 모드 1과 모드 3이 함께 사용합니다."""
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    # A, B 두 번의 MAC으로 판정 1번 → 그 판정 1번에 걸린 평균 시간
    elapsed = measure_ms(lambda: judge(mac(pattern, filter_a), mac(pattern, filter_b)))
    winner = judge(score_a, score_b)

    section("[3] MAC 결과")
    print_scores_and_decision("A", score_a, "B", score_b, decision_text_ab(winner))
    print(f"연산 시간(평균/{REPEAT}회, MAC 2회+판정): {elapsed:.4f} ms")
    return winner


# ============================================================
# 11. 모드 2: data.json 분석
# ============================================================

def run_json_mode(data_path: str = DEFAULT_DATA_PATH) -> Optional[List[CaseResult]]:
    print(f"데이터 파일: {data_path}")
    section("[1] 필터 로드")
    try:
        data = load_json(data_path)
    except DataError as exc:
        print(f"✗ data.json을 사용할 수 없습니다: {exc}")
        print("  → 파일 위치/이름과 JSON 문법을 확인한 뒤 다시 실행하세요. (--data 경로 지정 가능)")
        return None

    filters = load_filters(data["filters"])  # type: ignore[arg-type]
    for load in filters.values():
        if load.ok:
            print(f"✓ {load.key:<8} 필터 로드 완료 ({', '.join(load.filters)})")
        else:
            print(f"✗ {load.key:<8} 필터 로드 실패: {load.error}")

    section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = evaluate_patterns(data["patterns"], filters)  # type: ignore[arg-type]
    for res in results:
        print(f"--- {res.case_id} ---")
        if res.cross_score is None:
            print(f"FAIL [{res.category}] {res.reason}")
            continue
        print(f"Cross 점수: {res.cross_score!r}")
        print(f"X 점수: {res.x_score!r}")
        if res.passed:
            verdict = "PASS"
        elif res.category == REASON_TIE:
            verdict = f"FAIL ({REASON_TIE} 규칙)"
        else:
            verdict = f"FAIL ({res.category})"
        print(f"판정: {res.decision} | expected: {res.expected} | {verdict}")

    section(f"[3] 성능 분석 (평균/{REPEAT}회)")
    print_bench_table(benchmark(json_bench_inputs(data, filters)))

    section("[4] 결과 요약")
    print_summary(results)
    print()
    print('(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)')
    return results


def json_bench_inputs(data: Dict[str, object], filters: Dict[int, FilterLoad]) -> List[Tuple[int, Matrix, Matrix, str]]:
    """성능 분석 입력: data.json에 해당 크기의 필터/패턴이 있으면 그것을, 없으면 생성기 패턴을 씁니다."""
    patterns: Dict[int, Matrix] = {}
    for case_id, case in data["patterns"].items():  # type: ignore[union-attr]
        match = PATTERN_KEY_RE.match(case_id)
        if not match or not isinstance(case, dict):
            continue
        n = int(match.group(1))
        if n in patterns:
            continue
        try:
            patterns[n] = Matrix.from_rows(case.get("input"), expected_size=n)
        except ValueError:
            continue

    inputs = []
    for n in BENCH_SIZES:
        load = filters.get(n)
        if load is not None and load.ok and n in patterns:
            inputs.append((n, patterns[n], load.filters[LABEL_CROSS], "data.json"))
        else:
            inputs.append((n, make_cross(n), make_cross(n), "생성기"))
    return inputs


def print_summary(results: Sequence[CaseResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failed)}개")
    if not failed:
        print("\n실패 케이스: 없음")
        return
    print("\n실패 케이스:")
    for r in failed:
        print(f"- {r.case_id}: [{r.category}] {r.reason}")
    counts: Dict[str, int] = {}
    for r in failed:
        counts[r.category or "기타"] = counts.get(r.category or "기타", 0) + 1
    print("\n실패 원인 분류: " + ", ".join(f"{k} {v}건" for k, v in counts.items()))


# ============================================================
# 12. 모드 3 (보너스): 패턴 생성기
# ============================================================

def run_generator_mode(input_fn: InputFunc = input) -> None:
    section("[1] 크기 입력")
    n = read_int(f"생성할 크기 N (2~{MAX_GENERATOR_SIZE}): ", 2, MAX_GENERATOR_SIZE, input_fn)
    cross, x = make_cross(n), make_x(n)

    section(f"[2] 생성된 필터 ({n}×{n})")
    if n <= 15:
        print("Cross (필터 A):")
        print(cross.format())
        print("X (필터 B):")
        print(x.format())
    else:
        print(f"N이 15보다 커서 그림은 생략합니다. (1의 개수: Cross {sum(cross.to_flat()):g}, X {sum(x.to_flat()):g})")

    # 생성된 Cross/X를 모드 1과 같은 판정 함수(report_ab_judgement)에 그대로 재활용
    for label in (LABEL_CROSS, LABEL_X):
        print(f"\n▶ 입력 패턴 = 생성된 {label}")
        winner = report_ab_judgement(make_pattern(label, n), cross, x)
        decided = {"A": LABEL_CROSS, "B": LABEL_X}.get(winner, UNDECIDED)
        print(f"→ 표준 라벨 판정: {decided} ({'정상' if decided == label else '확인 필요'})")

    sizes = sorted(set(BENCH_SIZES) | {n})
    section(f"[4] 성능 분석 (생성기 패턴, 평균/{REPEAT}회)")
    print_bench_table(benchmark([(s, make_cross(s), make_cross(s), "생성기") for s in sizes]))


# ============================================================
# 13. 모드 4 (보너스): 2차원 vs 1차원 성능 비교
# ============================================================

def run_optimization_mode() -> None:
    section(f"2차원 vs 1차원(평탄화) MAC 비교 (평균/{BONUS_REPEAT}회, 두 방식 동일)")
    rows = compare_2d_vs_1d()
    print(pad("크기", 10) + pad("연산 횟수", 10, ">") + pad("2D(ms)", 11, ">") + pad("1D(ms)", 11, ">")
          + pad("1D/2D", 8, ">") + "  결과 일치")
    print("-" * 62)
    for row in rows:
        ratio = row.ms_1d / row.ms_2d if row.ms_2d > 0 else float("nan")
        print(pad(f"{row.size}×{row.size}", 10) + pad(row.size * row.size, 10, ">")
              + pad(f"{row.ms_2d:.4f}", 11, ">") + pad(f"{row.ms_1d:.4f}", 11, ">")
              + pad(f"{ratio:.2f}", 8, ">") + "  " + ("✓" if row.same_result else "✗"))
    print("입력: 생성기 Cross 패턴 × Cross 필터 (두 방식 동일), 평탄화 비용은 측정에서 제외")
    print("1D/2D < 1 이면 1차원 방식이 더 빠름")


# ============================================================
# 14. 메인 메뉴
# ============================================================

MENU = """
[모드 선택]
1. 사용자 입력 (3x3)
2. data.json 분석
3. 패턴 생성기 (보너스)
4. 최적화 비교: 2차원 vs 1차원 (보너스)
0. 종료"""


def main(argv: Optional[Sequence[str]] = None, input_fn: InputFunc = input) -> int:
    parser = argparse.ArgumentParser(description="Mini NPU Simulator (MAC 연산 기반 Cross/X 판정)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="data.json 경로 (기본: main.py와 같은 폴더)")
    args = parser.parse_args(argv)

    print("=== Mini NPU Simulator ===")
    try:
        while True:
            print(MENU)
            choice = read_int("선택: ", 0, 4, input_fn)
            if choice == 0:
                print("프로그램을 종료합니다.")
                return 0
            if choice == 1:
                run_user_mode(input_fn)
            elif choice == 2:
                run_json_mode(args.data)
            elif choice == 3:
                run_generator_mode(input_fn)
            else:
                run_optimization_mode()
    except InputAborted:
        print("\n입력이 종료되어 프로그램을 마칩니다.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
