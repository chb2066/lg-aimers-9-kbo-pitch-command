"""E1.1 - 외부 노트북(`train.ipynb`)의 61피처 재현.

구성: 공식 raw 47 + 투수 current-season 10 + 타자 current-season 7.

★핵심 설계★
외부 노트북은 train과 test에서 서로 다른 경로를 쓴다:
  - train: `_season_delta_by_entity` - 각 (season, entity)의 첫 행 값을 빼서
           "이번 시즌 지금까지"를 만든다(행별 인과적).
  - test : `build_train_end_entity_baseline` - 학습 마지막 행의 누적값을
           고정 상수로 얼려 두고 그걸 뺀다(다른 test 행 미참조 -> 합법).

walk-forward 검증에서 **validation fold는 test 경로를 써야 한다.**
val_season 시작 시점의 기준값 = 학습 끝 시점의 누적값이라는 구조가
실제 배포(2025 예측)와 정확히 같기 때문. validation 안에서 첫 행을
찾는 건 test 내부 행 참조에 해당하므로 실전 재현이 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TGT = "control_success"
SMOOTH_ALPHA = 50.0

OFFICIAL_BASE_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "base_state", "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

PITCHER_CS = [
    "season_pitcher_n", "season_pitcher_success_count",
    "season_pitcher_success_rate", "season_pitcher_success_sm50",
    "season_pitcher_reverse_count", "season_pitcher_reverse_rate",
    "season_pitcher_reverse_sm50", "season_pitcher_middle_count",
    "season_pitcher_middle_rate", "season_pitcher_middle_sm50",
]
BATTER_CS = [
    "season_batter_n", "season_batter_success_count",
    "season_batter_success_rate", "season_batter_middle_count",
    "season_batter_middle_rate", "season_batter_success_sm50",
    "season_batter_middle_sm50",
]
CURRENT_SEASON_FEATURES = PITCHER_CS + BATTER_CS
FEATURE_COLS = OFFICIAL_BASE_FEATURES + CURRENT_SEASON_FEATURES

CATEGORICAL_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
]


def _count_from_rate(df, n_col, rate_col):
    n = pd.to_numeric(df[n_col], errors="coerce").fillna(0).to_numpy("f8")
    rate = pd.to_numeric(df[rate_col], errors="coerce").to_numpy("f8")
    count = np.zeros(len(df), "f8")
    v = np.isfinite(rate)
    count[v] = np.rint(n[v] * rate[v])
    return n, count


def _safe_rate(count, n):
    r = np.full(len(count), np.nan, "f8")
    v = n > 0
    r[v] = count[v] / n[v]
    return r


def _smoothed(season_count, season_n, asof_rate, alpha=None):
    """(season_count + a*prior) / (season_n + a) - 우리 K90과 같은 공식.
    외부는 a=50, prior=통산 asof rate.

    ★alpha 기본값은 호출 시점에 모듈 전역에서 읽는다.★
    (정의 시점 바인딩이면 `FE.SMOOTH_ALPHA = x` 로 바꿔도 안 먹힌다
     2026-08-27 E15 에서 실제로 이 버그로 세 스윕이 전부 동일값이 나왔다.)
    """
    if alpha is None:
        alpha = SMOOTH_ALPHA
    prior = pd.to_numeric(asof_rate, errors="coerce").to_numpy("f8")
    out = np.full(len(season_count), np.nan, "f8")
    v = np.isfinite(prior)
    out[v] = (season_count[v] + alpha * prior[v]) / (season_n[v] + alpha)
    return out


def _season_delta(df, values, entity_col):
    """각 (season, entity)의 첫 행 값을 빼서 '이번 시즌 지금까지'로."""
    keys = pd.MultiIndex.from_arrays(
        [df["season"].to_numpy(), df[entity_col].to_numpy()])
    s = pd.Series(values, index=df.index)
    start = s.groupby(keys, sort=False).transform("first").to_numpy("f8")
    return np.maximum(values - start, 0.0)


def build_end_baseline(df, entity_col, n_col, rate_cols):
    """학습 마지막 행 기준 누적 상태를 얼린다(=다음 시즌 시작점)."""
    last = df.groupby(entity_col, sort=False).tail(1)
    out = {}
    for row in last.itertuples(index=False):
        n_before = float(getattr(row, n_col))
        rec = {"n": int(round(n_before)) + 1}
        for name, rc in rate_cols.items():
            rate = getattr(row, rc)
            if pd.isna(rate) or n_before <= 0:
                c = 0
            else:
                c = int(round(n_before * float(rate)))
            if name == "success":
                c += int(getattr(row, TGT))
            rec[name] = c
        out[getattr(row, entity_col)] = rec
    return out


def _base_arr(keys, baseline, name):
    return keys.map(lambda k: baseline.get(k, {}).get(name, 0)).to_numpy("f8")


def _apply_pitcher(out, season_n, season_suc, season_rev, season_mid):
    out["season_pitcher_n"] = season_n
    out["season_pitcher_success_count"] = season_suc
    out["season_pitcher_success_rate"] = _safe_rate(season_suc, season_n)
    out["season_pitcher_reverse_count"] = season_rev
    out["season_pitcher_reverse_rate"] = _safe_rate(season_rev, season_n)
    out["season_pitcher_middle_count"] = season_mid
    out["season_pitcher_middle_rate"] = _safe_rate(season_mid, season_n)
    out["season_pitcher_success_sm50"] = _smoothed(
        season_suc, season_n, out["asof_pitcher_success_rate"])
    out["season_pitcher_reverse_sm50"] = _smoothed(
        season_rev, season_n, out["asof_pitcher_reverse_rate"])
    out["season_pitcher_middle_sm50"] = _smoothed(
        season_mid, season_n, out["asof_pitcher_middle_rate"])


def _apply_batter(out, season_n, season_suc, season_mid):
    out["season_batter_n"] = season_n
    out["season_batter_success_count"] = season_suc
    out["season_batter_success_rate"] = _safe_rate(season_suc, season_n)
    out["season_batter_middle_count"] = season_mid
    out["season_batter_middle_rate"] = _safe_rate(season_mid, season_n)
    out["season_batter_success_sm50"] = _smoothed(
        season_suc, season_n, out["asof_batter_success_rate"])
    out["season_batter_middle_sm50"] = _smoothed(
        season_mid, season_n, out["asof_batter_middle_rate"])


def add_train_cs(df):
    """학습 행: 시즌 첫 행 기준 delta (행별 인과적)."""
    out = df.copy()
    p_n, p_suc = _count_from_rate(out, "asof_pitcher_n",
                                  "asof_pitcher_success_rate")
    _, p_rev = _count_from_rate(out, "asof_pitcher_n",
                                "asof_pitcher_reverse_rate")
    _, p_mid = _count_from_rate(out, "asof_pitcher_n",
                                "asof_pitcher_middle_rate")
    _apply_pitcher(out,
                   _season_delta(out, p_n, "pitcher_id"),
                   _season_delta(out, p_suc, "pitcher_id"),
                   _season_delta(out, p_rev, "pitcher_id"),
                   _season_delta(out, p_mid, "pitcher_id"))

    b_n, b_suc = _count_from_rate(out, "asof_batter_n",
                                  "asof_batter_success_rate")
    _, b_mid = _count_from_rate(out, "asof_batter_n",
                                "asof_batter_middle_rate")
    _apply_batter(out,
                  _season_delta(out, b_n, "batter_id"),
                  _season_delta(out, b_suc, "batter_id"),
                  _season_delta(out, b_mid, "batter_id"))
    return out


def add_test_cs(df, p_base, b_base):
    """검증/테스트 행: 학습 끝 시점 고정 상수 기준 (실전과 동일 경로)."""
    out = df.copy()
    p_n, p_suc = _count_from_rate(out, "asof_pitcher_n",
                                  "asof_pitcher_success_rate")
    _, p_rev = _count_from_rate(out, "asof_pitcher_n",
                                "asof_pitcher_reverse_rate")
    _, p_mid = _count_from_rate(out, "asof_pitcher_n",
                                "asof_pitcher_middle_rate")
    pk = out["pitcher_id"]
    s_n = np.maximum(p_n - _base_arr(pk, p_base, "n"), 0.0)
    s_suc = np.maximum(p_suc - _base_arr(pk, p_base, "success"), 0.0)
    s_rev = np.minimum(
        np.maximum(p_rev - _base_arr(pk, p_base, "reverse"), 0.0), s_n)
    s_mid = np.minimum(
        np.maximum(p_mid - _base_arr(pk, p_base, "middle"), 0.0), s_n)
    _apply_pitcher(out, s_n, s_suc, s_rev, s_mid)

    b_n, b_suc = _count_from_rate(out, "asof_batter_n",
                                  "asof_batter_success_rate")
    _, b_mid = _count_from_rate(out, "asof_batter_n",
                                "asof_batter_middle_rate")
    bk = out["batter_id"]
    sb_n = np.maximum(b_n - _base_arr(bk, b_base, "n"), 0.0)
    sb_suc = np.maximum(b_suc - _base_arr(bk, b_base, "success"), 0.0)
    sb_mid = np.minimum(
        np.maximum(b_mid - _base_arr(bk, b_base, "middle"), 0.0), sb_n)
    _apply_batter(out, sb_n, sb_suc, sb_mid)
    return out


def builder(tr, va, feature_cols=None, cat_cols=None):
    """wf_harness용 builder. tr만 보고 상수를 만든다."""
    cols = feature_cols or FEATURE_COLS
    cats = cat_cols if cat_cols is not None else CATEGORICAL_COLS
    tr2 = add_train_cs(tr)
    p_base = build_end_baseline(
        tr, "pitcher_id", "asof_pitcher_n",
        {"success": "asof_pitcher_success_rate",
         "reverse": "asof_pitcher_reverse_rate",
         "middle": "asof_pitcher_middle_rate"})
    b_base = build_end_baseline(
        tr, "batter_id", "asof_batter_n",
        {"success": "asof_batter_success_rate",
         "middle": "asof_batter_middle_rate"})
    va2 = add_test_cs(va, p_base, b_base)
    cats = [c for c in cats if c in cols]
    return tr2[cols], va2[cols], cats
