"""앵커 방식 current-season 피처.

핵심 아이디어
-------------
통산 누적에서 **시즌 시작 시점 상태**를 빼면 시즌 누적만 남는다.
그런데 데이터에는 카운트가 없고 비율뿐이라 역산이 필요하다.

    S = round(asof_n x asof_rate)        # 통산 성공 횟수

리셋되지 않는 누적이라 이 항등식이 성립한다.
(2026-08-28 실측: 1,474,300행 검사, 불일치 **0행**, 정수 오차 0.00e+00)

앵커
----
그 선수가 **가장 최근의 더 이른 시즌**을 끝냈을 때 상태 (n0, S0).
fit 단계에서 미리 만들어 동결한다.

    idx = train.groupby([id_col, "season"])[n_col].idxmax()   # 시즌별 마지막 행
    end[cols] = end.groupby(id_col)[cols].shift(1)            # 한 시즌 밀기

`shift(1)` 이 핵심이다 — (선수, 2024) 키를 조회하면 **2023 종료 시점**이 나온다.
2024 정보가 새지 않는다.

적용
----
    span        = n_t - n0                      # 올해 던진 공 수
    season_rate = (S_t - S0) / span             # 올해 성공률
    career_minus_season = asof_rate - season_rate

apply 는 (id, season) reindex 조회 하나뿐이다.
groupby / shift / rolling 없음 → **한 행만 넘겨도 전체 프레임과 같은 값**.

기존 feat_F 와 다른 두 곳
-------------------------
| | feat_F | anchored |
|---|---|---|
| 조회 키 | (pitcher_id, season - 1) | 가장 최근의 더 이른 시즌 |
| 게이트 | cnt > 50 | 없음 |

1. `season - 1` 은 한 시즌 거른 선수(부상/군복무)가 NaN 이 된다.
   "가장 최근의 더 이른 시즌" 은 2년 전이든 3년 전이든 찾아낸다.
2. 게이트를 없애 시즌 초반 행도 값을 받는다. 표본이 작아 분산은 크지만
   트리가 `season_n` 을 같이 보고 신뢰도를 알아서 가른다.

실측 (2024, 253,507행): 결측 74,259 -> **81**. 커버리지 **91배**.
겹치는 구간의 값은 동일하다 — 다른 수식이 아니라 같은 수식의 커버리지 확장.

출력 16컬럼
-----------
pitcher 5사건(success/middle/ball/strike/reverse): season_n 1 + rate 5 + cms 5 = 11
batter  2사건(success/middle)                    : season_n 1 + rate 2 + cms 2 = 5
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PITCHER_EVENTS = ["success", "middle", "ball", "strike", "reverse"]
BATTER_EVENTS = ["success", "middle"]


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce").to_numpy("f8")


def _counts(d, who, events):
    """비율에서 통산 성공 횟수를 역산한다. S = round(n x rate)."""
    n = _num(d, f"asof_{who}_n")
    out = {"n": n}
    for e in events:
        col = f"asof_{who}_{e}_rate"
        out[e] = np.round(n * _num(d, col)) if col in d.columns else np.full(len(d), np.nan)
    return out


def build_anchor(df_fit, who, events):
    """(id, season) -> **가장 최근의 더 이른 시즌** 종료 시점 (n0, S0...).

    fit 데이터만 보고 만들어 동결한다. shift(1) 로 한 시즌 밀기 때문에
    (선수, T) 를 조회하면 T 이전 마지막 시즌의 종료 상태가 나온다.
    """
    c = _counts(df_fit, who, events)
    t = pd.DataFrame({"id": df_fit[f"{who}_id"].to_numpy(),
                      "season": df_fit["season"].to_numpy(),
                      "n": c["n"]})
    for e in events:
        t[e] = c[e]

    # 시즌별 마지막 행 = 그 시즌 종료 시점 누적
    idx = t.groupby(["id", "season"])["n"].idxmax()
    end = t.loc[idx].sort_values(["id", "season"]).set_index(["id", "season"])
    cols = ["n"] + events

    # ★시즌 격자로 채운다★
    # 그냥 shift(1) 만 하면 학습에 없는 시즌(=평가 시즌) 키가 통째로 미스나고
    # n0=0 으로 폴백해 '올해 성적'이 아니라 '통산 성적'이 나온다.
    # (2026-08-28 실측: 겹치는 구간 상관 0.598 로 어긋남 — 이 버그였다.)
    ids = end.index.get_level_values(0).unique()
    s_lo = int(end.index.get_level_values(1).min())
    s_hi = int(end.index.get_level_values(1).max()) + 2   # 평가 시즌까지 포함
    grid = pd.MultiIndex.from_product([ids, range(s_lo, s_hi + 1)],
                                      names=["id", "season"])
    E = end[cols].reindex(grid)
    # ffill = 안 뛴 시즌은 직전 종료 상태를 물려받는다(부상/군복무 대응)
    E = E.groupby(level=0).ffill()
    # shift(1) = 그 시즌 '시작' 상태 = 가장 최근의 더 이른 시즌 종료 상태
    E = E.groupby(level=0).shift(1)
    return E


def apply_anchored(d, anchors, min_span=1.0):
    """행별 조회 하나로 시즌 누적을 복원한다. groupby/shift/rolling 없음."""
    out = pd.DataFrame(index=d.index)
    for who, events in (("pitcher", PITCHER_EVENTS), ("batter", BATTER_EVENTS)):
        A = anchors[who]
        c = _counts(d, who, events)
        key = pd.MultiIndex.from_arrays([d[f"{who}_id"], d["season"]])
        prev = A.reindex(key)

        n0 = prev["n"].to_numpy("f8")
        n0 = np.where(np.isfinite(n0), n0, 0.0)     # 데뷔 시즌은 0 에서 시작
        span = c["n"] - n0
        span = np.where(span >= min_span, span, np.nan)
        out[f"season_{who}_n"] = np.where(np.isfinite(span), span, 0.0)

        for e in events:
            s0 = prev[e].to_numpy("f8")
            s0 = np.where(np.isfinite(s0), s0, 0.0)
            rate = (c[e] - s0) / span
            out[f"season_{who}_{e}_rate"] = rate
            career = _num(d, f"asof_{who}_{e}_rate")
            out[f"cms_{who}_{e}"] = career - rate
    return out


def build_anchors(df_fit):
    return {"pitcher": build_anchor(df_fit, "pitcher", PITCHER_EVENTS),
            "batter": build_anchor(df_fit, "batter", BATTER_EVENTS)}


ANCHORED_COLS = (["season_pitcher_n"]
                 + [f"season_pitcher_{e}_rate" for e in PITCHER_EVENTS]
                 + [f"cms_pitcher_{e}" for e in PITCHER_EVENTS]
                 + ["season_batter_n"]
                 + [f"season_batter_{e}_rate" for e in BATTER_EVENTS]
                 + [f"cms_batter_{e}" for e in BATTER_EVENTS])
