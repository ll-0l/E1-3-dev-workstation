"""Mini NPU Simulator 단위 테스트 (표준 라이브러리 unittest만 사용).

실행: python3 -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
CROSS3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


def fake_input(lines):
    """input() 대신 쓸 함수. 미리 준비한 줄을 차례로 돌려주고, 다 쓰면 EOFError."""
    it = iter(lines)

    def _input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return _input


def quiet(func, *args, **kwargs):
    """출력을 숨기고 (결과, 출력문자열)을 돌려줍니다."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


class MatrixTest(unittest.TestCase):
    def test_get_set(self):
        m = main.Matrix(3)
        m.set(1, 2, 7)
        self.assertEqual(m.get(1, 2), 7.0)
        self.assertEqual(m.get(0, 0), 0.0)

    def test_supported_sizes(self):
        for n in (3, 5, 13, 25):
            m = main.Matrix(n)
            m.set(n - 1, n - 1, 1.5)
            self.assertEqual(m.size, n)
            self.assertEqual(len(m.to_flat()), n * n)
            self.assertEqual(m.to_flat()[-1], 1.5)

    def test_from_rows_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            main.Matrix.from_rows([[1, 2], [3]])            # 열 수 불일치
        with self.assertRaises(ValueError):
            main.Matrix.from_rows(CROSS3, expected_size=5)  # 행 수 불일치
        with self.assertRaises(ValueError):
            main.Matrix.from_rows([[1, "a"], [0, 1]])       # 숫자 아님
        with self.assertRaises(ValueError):
            main.Matrix.from_rows([[True, 0], [0, 1]])      # bool은 숫자로 보지 않음
        with self.assertRaises(ValueError):
            main.Matrix.from_rows([])

    def test_fast_validation_keeps_exact_errors(self):
        # 행 단위 빠른 검사가 놓치면 안 되는 값들: NaN, inf, bool, 문자열
        for bad, where in (([[0, float("nan")], [0, 1]], "(1행, 2열)"),
                           ([[0, 1], [float("inf"), 1]], "(2행, 1열)"),
                           ([[1, 0], [0, False]], "(2행, 2열)")):
            with self.assertRaises(ValueError) as ctx:
                main.Matrix.from_rows(bad)
            self.assertIn(where, str(ctx.exception))
        # 각각은 유한한데 더하면 넘치는 값은 정상으로 받아야 함
        self.assertEqual(main.Matrix.from_rows([[1e308, 1e308], [0, 1]]).get(0, 1), 1e308)

    def test_from_rows_copies(self):
        rows = [[1, 2], [3, 4]]
        m = main.Matrix.from_rows(rows)
        rows[0][0] = 99
        self.assertEqual(m.get(0, 0), 1.0)

    def test_flat_is_row_major(self):
        m = main.Matrix.from_rows([[1, 2], [3, 4]])
        self.assertEqual(m.to_flat(), [1.0, 2.0, 3.0, 4.0])


class MacTest(unittest.TestCase):
    def test_example_scores(self):
        cross, x = main.Matrix.from_rows(CROSS3), main.Matrix.from_rows(X3)
        self.assertEqual(main.mac(cross, cross), 5.0)
        self.assertEqual(main.mac(cross, x), 1.0)
        self.assertEqual(main.mac(x, x), 5.0)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            main.mac(main.Matrix(3), main.Matrix(5))

    def test_flat_matches_2d_exactly(self):
        for n in (3, 5, 13, 25, 50):
            p, f = main.make_cross(n), main.make_x(n)
            self.assertEqual(main.mac(p, f), main.mac_flat(p.to_flat(), f.to_flat()))

    def test_float_error_is_real(self):
        # 0.1을 9번 더하면 0.9가 아니라 0.8999999999999999 → epsilon이 필요한 이유
        p = main.Matrix.from_rows([[1] * 3 for _ in range(3)])
        f = main.Matrix.from_rows([[0.1] * 3 for _ in range(3)])
        score = main.mac(p, f)
        self.assertNotEqual(score, 0.9)
        self.assertEqual(main.judge(score, 0.9), main.UNDECIDED)


class LargeScaleTest(unittest.TestCase):
    def test_sparse_matches_dense_exactly(self):
        with open(os.path.join(ROOT, "data.json"), encoding="utf-8") as fp:
            import json
            data = json.load(fp)
        for n in (5, 13, 25):
            for label in ("cross", "x"):
                filt = main.Matrix.from_rows(data["filters"][f"size_{n}"][label])
                entries = main.nonzero_entries(filt)
                for idx in (1, 2):
                    pattern = main.Matrix.from_rows(data["patterns"][f"size_{n}_{idx}"]["input"])
                    self.assertEqual(main.mac(pattern, filt), main.mac_sparse(pattern, entries))

    def test_nonzero_count_of_cross(self):
        self.assertEqual(len(main.nonzero_entries(main.make_cross(101))), 2 * 101 - 1)

    def test_relative_epsilon_catches_large_ties(self):
        n = 300
        score = main.mac(main.Matrix(n, 1.0), main.Matrix(n, 0.1))   # 수학적으로 9000
        self.assertNotEqual(main.judge(score, n * n / 10), main.UNDECIDED)   # 절대 1e-9로는 놓침
        self.assertEqual(main.judge(score, n * n / 10, rel_epsilon=main.REL_EPSILON), main.UNDECIDED)
        self.assertEqual(main.judge(9000.0, 9001.0, rel_epsilon=main.REL_EPSILON), "B")  # 진짜 차이는 유지

    def test_experiment_small(self):
        rows = main.large_scale_experiment(sizes=(10, 30), repeat=2)
        for r in rows:
            self.assertTrue(r.same_result)
            self.assertEqual(r.rel_tie, main.UNDECIDED)
            self.assertGreater(r.mem_list_mb, r.mem_array_mb)


class JudgeTest(unittest.TestCase):
    def test_winner(self):
        self.assertEqual(main.judge(5.0, 1.0), "A")
        self.assertEqual(main.judge(1.0, 5.0), "B")

    def test_epsilon_tie(self):
        self.assertEqual(main.judge(0.9, 0.8999999999999999), main.UNDECIDED)
        self.assertEqual(main.judge(1.0, 1.0), main.UNDECIDED)
        self.assertEqual(main.judge(1.0, 1.0 + 1e-10), main.UNDECIDED)
        self.assertEqual(main.judge(1.0, 1.0 + 1e-8), "B")   # epsilon보다 크면 판정


class LabelTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(main.normalize_label("+"), "Cross")
        self.assertEqual(main.normalize_label("cross"), "Cross")
        self.assertEqual(main.normalize_label(" Cross "), "Cross")
        self.assertEqual(main.normalize_label("x"), "X")
        self.assertEqual(main.normalize_label("X"), "X")
        self.assertIsNone(main.normalize_label("o"))
        self.assertIsNone(main.normalize_label(None))
        self.assertIsNone(main.normalize_label(1))


class GeneratorTest(unittest.TestCase):
    def test_3x3_matches_mission_example(self):
        self.assertEqual(main.make_cross(3).rows(), [[float(v) for v in r] for r in CROSS3])
        self.assertEqual(main.make_x(3).rows(), [[float(v) for v in r] for r in X3])

    def test_generated_patterns_are_classified(self):
        for n in (3, 4, 5, 13, 25, 50):
            cross, x = main.make_cross(n), main.make_x(n)
            self.assertEqual(main.judge(main.mac(cross, cross), main.mac(cross, x)), "A", n)
            self.assertEqual(main.judge(main.mac(x, cross), main.mac(x, x)), "B", n)


class InputTest(unittest.TestCase):
    def test_parse_row(self):
        self.assertEqual(main.parse_row(" 0 1  0.5 ", 3), [0.0, 1.0, 0.5])
        for bad in ("0 1", "0 1 0 1", "a b c", "", "nan 1 0", "inf 0 0"):
            with self.assertRaises(ValueError, msg=bad):
                main.parse_row(bad, 3)

    def test_read_matrix_retries_bad_lines(self):
        lines = ["0 1 0", "1 1", "1 x 1", "", "1 1 1", "0 1 0"]
        matrix, out = quiet(main.read_matrix, "패턴", 3, fake_input(lines))
        self.assertEqual(matrix.rows(), [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(out.count("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요."), 3)

    def test_read_int_retries(self):
        value, out = quiet(main.read_int, "선택: ", 0, 4, fake_input(["a", "9", "2"]))
        self.assertEqual(value, 2)

    def test_eof_raises_input_aborted(self):
        with self.assertRaises(main.InputAborted):
            quiet(main.read_matrix, "필터 A", 3, fake_input(["0 1 0"]))


class DatasetTest(unittest.TestCase):
    def test_real_data_json(self):
        results, out = quiet(main.run_json_mode, main.DEFAULT_DATA_PATH)
        self.assertEqual(len(results), 6)
        status = {r.case_id: r.passed for r in results}
        self.assertEqual(status, {
            "size_5_1": False, "size_5_2": True,
            "size_13_1": True, "size_13_2": False,
            "size_25_1": False, "size_25_2": True,
        })
        for r in results:
            if not r.passed:
                self.assertEqual(r.category, main.REASON_TIE)   # 실패 3건 모두 동점 규칙
                self.assertEqual(r.decision, main.UNDECIDED)
        self.assertIn("총 테스트: 6개", out)
        self.assertIn("통과: 3개", out)
        self.assertIn("실패: 3개", out)

    def test_bad_data_does_not_crash(self):
        path = os.path.join(FIXTURES, "bad_data.json")
        results, out = quiet(main.run_json_mode, path)
        by_id = {r.case_id: r for r in results}
        self.assertEqual(len(results), 10)
        self.assertTrue(by_id["size_5_1"].passed)
        self.assertEqual(by_id["size_5_5"].category, main.REASON_MISMATCH)
        for case_id in ("size_5_2", "size_5_3", "size_5_4", "size_5_6",
                        "size_3_1", "size_7_1", "size_9_1", "cross_5_1"):
            self.assertFalse(by_id[case_id].passed, case_id)
            self.assertEqual(by_id[case_id].category, main.REASON_SCHEMA, case_id)
        self.assertIn("크기 불일치", by_id["size_5_2"].reason)
        self.assertIn("실패: 9개", out)

    def test_unusable_files(self):
        for path in (os.path.join(FIXTURES, "broken.json"), os.path.join(FIXTURES, "없는파일.json")):
            results, out = quiet(main.run_json_mode, path)
            self.assertIsNone(results)
            self.assertIn("사용할 수 없습니다", out)


class FlowTest(unittest.TestCase):
    def test_mode1_end_to_end(self):
        lines = ["1",
                 "0 1 0", "1 1 1", "0 1 0",     # 필터 A (Cross)
                 "1 0 1", "0 1 0", "1 0 1",     # 필터 B (X)
                 "1 0 1", "0 1 0", "1 0 1",     # 패턴 (X)
                 "0"]
        code, out = quiet(main.main, [], fake_input(lines))
        self.assertEqual(code, 0)
        self.assertIn("A 점수: 1.0", out)
        self.assertIn("B 점수: 5.0", out)
        self.assertIn("판정: B", out)
        self.assertIn("연산 시간", out)
        self.assertIn("성능 분석 (3×3", out)

    def test_mode1_tie(self):
        # A = 0.1을 9번 더함 = 0.8999999999999999, B = 가운데 0.9 하나 = 0.9 → 판정 불가
        lines = ["1",
                 "0.1 0.1 0.1", "0.1 0.1 0.1", "0.1 0.1 0.1",
                 "0 0 0", "0 0.9 0", "0 0 0",
                 "1 1 1", "1 1 1", "1 1 1", "0"]
        _, out = quiet(main.main, [], fake_input(lines))
        self.assertIn("A 점수: 0.8999999999999999", out)
        self.assertIn("B 점수: 0.9", out)
        self.assertIn("판정: 판정 불가 (|A-B| < 1e-9)", out)

    def test_bonus_modes_and_eof(self):
        code, out = quiet(main.main, [], fake_input(["3", "7", "4", "6"]))   # 생성기 → 최적화 → 범위 밖 → EOF
        self.assertEqual(code, 0)
        self.assertIn("표준 라벨 판정: Cross (정상)", out)
        self.assertIn("표준 라벨 판정: X (정상)", out)
        self.assertIn("100×100", out)
        self.assertNotIn("✗", out)
        self.assertIn("0~5 사이의 숫자를 입력하세요.", out)
        self.assertIn("입력이 종료되어", out)

    def test_benchmark_table(self):
        rows = main.benchmark([(n, main.make_cross(n), main.make_cross(n), "생성기") for n in main.BENCH_SIZES])
        self.assertEqual([r.ops for r in rows], [9, 25, 169, 625])
        self.assertTrue(all(r.avg_ms > 0 for r in rows))


if __name__ == "__main__":
    unittest.main()
