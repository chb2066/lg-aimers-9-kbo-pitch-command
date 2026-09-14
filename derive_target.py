"""평균 보정 상수 TARGET_R 을 train.csv 만으로 계산한다.

실행
    python derive_target.py

방법
    시즌 성공률을 예측하는 규칙 13가지를 만들고, 2021~2024년 각 시즌을 그 이전
    시즌들만으로 예측해 규칙마다 평균 절대오차를 구한다. 오차 제곱의 역수로
    가중평균해 2025년 값을 낸다. 결과는 0.48051 이고 train.py 의 TARGET_R 에 쓴다.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.environ.get("KBO_DATA_DIR", os.path.join(ROOT, "data")), "train.csv")


def main():
    df = pd.read_csv(TRAIN, usecols=["season", "game_month", "pitcher_id",
                                     "control_success"])
    yrs = sorted(df.season.unique())
    print("=" * 74)
    print("입력: data/train.csv  (%d행, 시즌 %s)" % (len(df), yrs))
    print("=" * 74)

    # ── 재료 1. 시즌 평균 ────────────────────────────────────────
    M = df.groupby("season").control_success.mean().to_dict()
    print("\n[재료 1] 시즌별 제구 성공률")
    for s in yrs:
        n = int((df.season == s).sum())
        print("    %d   %.5f   (n=%,d)".replace(",", "") % (s, M[s], n))

    # ── 재료 2. 시즌 말 k개월 평균 ───────────────────────────────
    T = {}
    for k in (1, 2, 3, 4):
        T[k] = {}
        for s in yrs:
            q = df[df.season == s]
            mos = sorted(q.game_month.unique())[-k:]
            T[k][s] = q[q.game_month.isin(mos)].control_success.mean()
    print("\n[재료 2] 시즌 말 k개월 평균")
    print("    %-6s" % "시즌" + "".join("%12s" % ("말%d개월" % k) for k in (1, 2, 3, 4))
          + "%12s" % "시즌평균")
    for s in yrs:
        print("    %-6d" % s + "".join("%12.5f" % T[k][s] for k in (1, 2, 3, 4))
              + "%12.5f" % M[s])

    # ── 재료 3. 같은 투수 안에서의 시즌간 변화 ────────────────────
    def within(a, b):
        q = df[df.season.isin([a, b])]
        g = q.pivot_table(index="pitcher_id", columns="season",
                          values="control_success", aggfunc=["mean", "size"])
        n = g["size"]
        ok = (n[a] >= 200) & (n[b] >= 200)
        return float((g["mean"][ok][b] - g["mean"][ok][a]).mean()), int(ok.sum())

    W = {}
    print("\n[재료 3] 같은 투수 안에서의 변화 (양 시즌 200구 이상인 투수만)")
    for a, b in zip(yrs[:-1], yrs[1:]):
        W[(a, b)], nn = within(a, b)
        print("    %d -> %d   %+.5f   (투수 %d명)" % (a, b, W[(a, b)], nn))

    # ── 후보 규칙 13개 ──────────────────────────────────────────
    def diffs(h):
        return np.diff([M[y] for y in h])

    def wchg(h):
        return np.mean([W[(a, b)] for a, b in zip(h[:-1], h[1:])])

    RULES = {
        "시즌평균":            lambda h: M[h[-1]],
        "시즌말1개월":          lambda h: T[1][h[-1]],
        "시즌말2개월":          lambda h: T[2][h[-1]],
        "시즌말3개월":          lambda h: T[3][h[-1]],
        "시즌말4개월":          lambda h: T[4][h[-1]],
        "평균/말2 반반":        lambda h: .5 * M[h[-1]] + .5 * T[2][h[-1]],
        "평균/말3 반반":        lambda h: .5 * M[h[-1]] + .5 * T[3][h[-1]],
        "평균/말4 반반":        lambda h: .5 * M[h[-1]] + .5 * T[4][h[-1]],
        "말3 + 0.5x차분":      lambda h: T[3][h[-1]] + .5 * diffs(h).mean(),
        "평균 + 0.5x차분":      lambda h: M[h[-1]] + .5 * diffs(h).mean(),
        "평균 + 같은투수변화":    lambda h: M[h[-1]] + wchg(h),
        "말3 + 같은투수변화":    lambda h: T[3][h[-1]] + wchg(h),
        "평균/말3 + 0.3x차분":  lambda h: .5 * M[h[-1]] + .5 * T[3][h[-1]] + .3 * diffs(h).mean(),
    }

    # ── 백테스트 ────────────────────────────────────────────────
    tests = yrs[2:]                      # 규칙에 최소 2시즌 이력이 필요
    print("\n" + "=" * 74)
    print("[백테스트] 각 목표 시즌은 **그 이전 시즌만** 보고 예측한다 (누출 없음)")
    print("=" * 74)
    print("    %-22s" % "규칙" + "".join("%10d" % t for t in tests)
          + "%11s%12s" % ("MAE", "2025 예측"))
    res = []
    for nm, f in RULES.items():
        err = []
        for t in tests:
            h = [y for y in yrs if y < t]
            err.append(f(h) - M[t])
        mae = float(np.mean(np.abs(err)))
        p25 = float(f(yrs))
        res.append((nm, mae, p25))
        print("    %-22s" % nm + "".join("%+10.4f" % e for e in err)
              + "%11.5f%12.5f" % (mae, p25))

    # ── 역분산 가중 결합 ────────────────────────────────────────
    e = np.array([r[1] for r in res])
    v = np.array([r[2] for r in res])
    w = 1.0 / e ** 2
    target = float((w * v).sum() / w.sum())
    print("\n" + "=" * 74)
    print("[결합] 역분산(1/MAE^2) 가중평균")
    print("=" * 74)
    print("    %-22s%11s%12s%11s" % ("규칙", "MAE", "2025 예측", "가중치"))
    for (nm, mae, p25), ww in sorted(zip(res, w), key=lambda z: -z[1]):
        print("    %-22s%11.5f%12.5f%11.0f" % (nm, mae, p25, ww))
    sd = float(np.sqrt((w * (v - target) ** 2).sum() / w.sum()))
    print("\n    가중평균                             %.5f" % target)
    print("    규칙 간 가중표준편차                    %.5f" % sd)
    print("    (추정 불확실성의 대리값. 기대손실 4e5 x sd^2 = %.1f 점)" % (4e5 * sd * sd))

    print("\n" + "=" * 74)
    print("  TARGET_R = %.5f" % target)
    print("  PROXY_OFF = 0.0")
    print("=" * 74)
    print("""
  PROXY_OFF 는 0 이다. 프록시(직전 시즌 행을 목표 시즌으로 재라벨한 것)와 실제
  목표 시즌 사이의 예측평균 차이를 배포와 같은 구조(학습 <= N-1)로 네 시즌에서
  측정했더니 -0.0055 / +0.0072 / -0.0006 / -0.0126 로 **부호가 일정하지 않았고**
  평균 -0.0029 가 표준오차 0.0036 안이라 0 과 구분되지 않았다. 그래서 0 으로 둔다.

  이 값이 최종 제출본에서 쓰이는 곳
    train.py 가 만드는 bundle.pkl 의 calib_shift
    프록시에서 예측 평균이 TARGET_R + PROXY_OFF 가 되도록 이분탐색으로 로짓
    상수 하나를 푼 값이며, bundle 의 target_provenance 에 유래가 기록돼 있다.

  리더보드 미사용 확인
    이 스크립트는 data/train.csv 만 읽는다. 대회 제출 점수를 입력으로 쓰지 않는다.
""")


if __name__ == "__main__":
    sys.exit(main())
