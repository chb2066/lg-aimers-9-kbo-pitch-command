"""범용 추론 스크립트 — 번들이 선언한 피처/모델만 켜서 돌린다.

세 제출본(A/B/C)이 같은 코드를 쓴다. 무엇을 켤지는 bundle.pkl 의
`feats` 와 `models` 가 정한다.

규정 준수
---------
평가 행은 **완전히 독립**으로 처리된다.
  current-season  학습 끝 누적을 얼려두고 그 행 자신의 asof_* 에서 뺀다
  조건부 프로파일   2025 이전으로 동결한 표를 (키, 상황) 으로 조회만
  앵커            (선수, 시즌) 앵커표를 조회만
  실패유형/매치업/모드교호  그 행 자신의 컬럼끼리 연산
  보정            학습 때 얼린 상수 하나를 로짓에 더하기만
test.csv 전체를 훑는 집계(groupby/rolling/누적/분포)는 어디에도 없다.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feat_anchored as FA   # noqa: E402
import feat_ext61 as FE      # noqa: E402
import kbofeat as KF         # noqa: E402

ID_COL, TGT = "row_id", "control_success"
FORBIDDEN = {"plate_x", "plate_z", "pitch_result", "call", "result",
             "tagged_pitch_type", "auto_pitch_type", "pitch_type_group",
             "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
             "extension", "rel_height", "rel_side", "zone_speed",
             "control_success"}


def _logit(p):
    p = np.clip(np.asarray(p, "f8"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce").to_numpy("f8")


def add_failtype(d, X):
    """실패유형 x 상황. 상황마다 위험한 실패 유형이 다르다 —
    3볼이면 한가운데, 2스트라이크면 크게 벗어남."""
    mid = _num(d, "asof_pitcher_middle_rate")
    rev = _num(d, "asof_pitcher_reverse_rate")
    ball = _num(d, "asof_pitcher_ball_rate")
    bb, ss = _num(d, "balls_before"), _num(d, "strikes_before")
    X["ft_mid_x_3b"] = mid * (bb >= 3)
    X["ft_mid_x_balls"] = mid * bb
    X["ft_rev_x_2s"] = rev * (ss >= 2)
    X["ft_rev_x_strikes"] = rev * ss
    X["ft_ball_x_3b"] = ball * (bb >= 3)
    X["ft_mid_minus_rev"] = mid - rev
    X["ft_mid_x_2out"] = mid * (_num(d, "outs_before") >= 2)
    X["ft_rev_x_runner"] = rev * (_num(d, "num_runners_on") > 0)
    return X


def add_matchup(d, X):
    """투수-타자 상호작용. 지금은 두 축이 따로 들어가 있다."""
    ps = _num(d, "asof_pitcher_success_rate")
    bs = _num(d, "asof_batter_success_rate")
    pm = _num(d, "asof_pitcher_middle_rate")
    bm = _num(d, "asof_batter_middle_rate")
    same = (d["pitcher_hand"].astype(str).to_numpy()
            == d["batter_hand"].astype(str).to_numpy()).astype("f8")
    X["mu_same_hand"] = same
    X["mu_p_x_b"] = ps * bs
    X["mu_p_minus_b"] = ps - bs
    X["mu_mid_x"] = pm * bm
    X["mu_p_x_same"] = ps * same
    X["mu_b_x_same"] = bs * same
    return X


def add_era(d, X, new_era=2023):
    """★신체제 플래그★ game_type F 는 2023 에 성공률 부호가 뒤집혔다
    (2022 0.7087 -> 2023 0.4729, 기록방식 변경 추정).
    옛 F 를 버리면 2022 가 무너지므로, 버리지 말고 **체제를 알려준다**.
    2025 는 신체제이므로 1 로 들어간다. 그 행 자신의 season/game_type 만 사용."""
    isF = d["game_type"].astype(str).eq("F").to_numpy().astype("f8")
    new = (pd.to_numeric(d["season"], errors="coerce").to_numpy("f8") >= new_era).astype("f8")
    X["gt_new_era"] = new
    X["gt_F_new"] = isF * new
    X["gt_F_old"] = isF * (1 - new)
    return X


def recode_game_type(d, X, new_era=2023):
    """★game_type 3범주 재코딩★ R / F_old / F_new.
    F 의 의미가 2023 에 바뀌었다(같은 투수 안 F−R: 2022 +0.248 -> 2023 +0.020).
    범주가 바뀐 문제는 범주로 표현한다 — 이진 플래그보다 2023 에서 360점 낫다.
    그 행 자신의 season / game_type 만 사용."""
    isF = d["game_type"].astype(str).eq("F").to_numpy()
    new = pd.to_numeric(d["season"], errors="coerce").to_numpy("f8") >= new_era
    X["game_type"] = np.where(isF, np.where(new, "F_new", "F_old"), "R")
    return X


def add_team13(d, X):
    """★팀13 체제 지시자★ train.csv 라벨 통계에서 확인한 상수만 쓴다.

    팀13 관여행(전체 28.6%)의 성공률 격차가 두 번 꺾인다.
      2019 -0.0142  2020 -0.0324 | 2021 +0.0582  2022 +0.0581 | 2023 +0.0246  2024 +0.0210
    2023 안에서는 4월만 음수(-0.0133)이고 5월부터 양수(+0.0176 -> +0.0393).
    팀13 은 F 비율 38.1% 로 다른 팀(3~10%)의 4~12배인 이상 팀이다.

    전환 시점(2023-05)과 F 조건은 학습 데이터 통계에서 확인해 **상수로 고정**했다.
    계산에 쓰는 것은 그 행 자신의 pitcher_team_id / batter_team_id / season /
    game_month / game_type 뿐이다 — 다른 평가 행이나 전체 집계를 보지 않는다.
    """
    X = X.copy()
    pt = pd.to_numeric(d["pitcher_team_id"], errors="coerce").to_numpy("f8")
    bt = pd.to_numeric(d["batter_team_id"], errors="coerce").to_numpy("f8")
    s = pd.to_numeric(d["season"], errors="coerce").to_numpy("f8")
    mo = pd.to_numeric(d["game_month"], errors="coerce").to_numpy("f8")
    isF = d["game_type"].astype(str).eq("F").to_numpy()
    new = (s > 2023) | ((s == 2023) & ((mo >= 5) | isF))
    t13 = ((pt == 13) | (bt == 13)).astype("f8")
    X["t13"] = t13
    X["t13_new"] = t13 * new
    return X


def build_frame(d, b):
    f = b["feats"]
    FE.SMOOTH_ALPHA = b["alpha"]
    cs = FE.add_test_cs(d, b["p_base"], b["b_base"])
    X = cs[FE.FEATURE_COLS].copy()
    if f.get("cond"):
        # 번들이 조합을 직접 실어 오면 그것만 쓴다(재정의안: 축을 골라 쓴다).
        # 없으면 kbofeat 의 기본 COND_SPECS.
        if b.get("cond_specs") is not None:
            KF.COND_SPECS = b["cond_specs"]
        X = KF.cond_apply_frozen_all(X, KF.cond_add_keys(d), b["cond_frozen"])
    if f.get("mode"):
        X = KF.add_mode_inter(X, d)
    if f.get("anchor"):
        Z = FA.apply_anchored(d, b["anchors"])
        for c in Z.columns:
            X["an_" + c] = Z[c].to_numpy("f8")
    if f.get("failtype"):
        X = add_failtype(d, X)
    if f.get("matchup"):
        X = add_matchup(d, X)
    if f.get("era"):
        X = add_era(d, X)
    if f.get("recode_gt"):
        X = recode_game_type(d, X)
    if f.get("team13"):
        X = add_team13(d, X)
    return X[b["cols"]]


def predict(test, b, verbose=False):
    X = build_frame(test, b)
    A = X.copy()
    for c in b["cats"]:
        A[c] = A[c].where(A[c].notna(), "__NA__").astype(str)
    for c in A.columns:
        if c not in b["cats"]:
            A[c] = pd.to_numeric(A[c], errors="coerce")

    # 모델을 손실 종류별로 묶어 **그룹 안에서는 평균, 그룹 간에는 가중합**.
    #   MultiClass 는 실패 유형(한가운데/의도반대/크게벗어남)까지 학습한 모델이고
    #   클래스 0 이 제구 성공이다. 회귀와 틀리는 방식이 달라 섞으면 이득이 난다.
    #   가중치는 bundle 의 `blend_w`(손실 -> 가중) 가 정한다. 없으면 균등.
    grp = {}
    for fn, loss in b["models"]:
        path = os.path.join(HERE, "model", fn)
        if loss == "MultiClass":
            m = CatBoostClassifier(); m.load_model(path)
            p = m.predict_proba(A)[:, 0]          # 클래스 0 = 제구 성공
        elif loss == "Logloss":
            m = CatBoostClassifier(); m.load_model(path)
            p = m.predict_proba(A)[:, 1]
        else:
            m = CatBoostRegressor(); m.load_model(path)
            p = m.predict(A, task_type="CPU")
        g = grp.setdefault(loss, [0.0, 0])
        g[0] = g[0] + _logit(np.clip(p, 1e-6, 1 - 1e-6))
        g[1] += 1
    bw = b.get("blend_w") or {k: 1.0 for k in grp}
    tot = sum(float(bw.get(k, 0.0)) for k in grp)
    if tot <= 0:
        raise RuntimeError("[번들] blend_w 합이 0")
    acc = sum(float(bw.get(k, 0.0)) * (v[0] / v[1]) for k, v in grp.items())
    acc, n = acc, tot

    # ------------------------------------------------------------------
    # 출력 클리핑 — 다중분류가 드물게 과도하게 확신한다.
    #   실측(2024 fold): 예측 0.0~0.2 인 81행의 실제 성공률이 0.41 이었다.
    #   행 수는 0.03% 지만 Brier 손실이 커서 LB 로 ~9점이다.
    #   [0.30, 0.70] 은 3 fold 모두 양수(+10.3/+6.5/+7.9). 상한을 0.65 로
    #   더 조이면 2022 가 -89.9 로 무너진다 — 그 시즌엔 0.65 이상이 진짜 예측이다.
    # 그 행 자신의 값만 자르므로 행 독립은 유지된다.
    # ------------------------------------------------------------------
    cl = b.get("clip")
    if cl:
        lo, hi = float(cl[0]), float(cl[1])
        pp = np.clip(1 / (1 + np.exp(-(acc / n))), lo, hi)
        acc, n = np.log(pp / (1 - pp)), 1.0
    # ------------------------------------------------------------------
    # 잔차 보정 — 학습 시즌 OOF 잔차를 **시즌별 중심화**해 얼린 표를 조회만 한다.
    # 전역 레벨은 뒤의 목표평균 이동이 잡고, 여기서는 세그먼트 편차만 더한다.
    # game_type F 는 2023 에 체제가 뒤집혔으므로 R 에만 적용한다.
    # 조회는 그 행 자신의 키로만 -> 행 독립.
    # ------------------------------------------------------------------
    rc = b.get("residual") or {}
    if rc.get("use"):
        pp = 1 / (1 + np.exp(-(acc / n)))
        bh = test["batter_hand"].where(test["batter_hand"].notna(), "__NA__").astype(str)
        b_ = pd.to_numeric(test["balls_before"], errors="coerce").astype("Int64").astype(str)
        s_ = pd.to_numeric(test["strikes_before"], errors="coerce").astype("Int64").astype(str)
        cs = b_ + "-" + s_
        pid = test["pitcher_id"].astype(str)
        bsk = test["base_state"].where(test["base_state"].notna(), "__NA__").astype(str)
        c = np.array([rc["ph"].get(f"{int(pp_)}|{h}", 0.0)
                      for pp_, h in zip(test["pitcher_id"], bh)], dtype=float)
        c = c + cs.map(rc["ct"]).fillna(0.0).to_numpy(dtype=float)
        c = c + (pid + "|" + cs).map(rc["pc"]).fillna(0.0).to_numpy(dtype=float)
        c = c + bsk.map(rc["bs"]).fillna(0.0).to_numpy(dtype=float)
        rmask = test["game_type"].astype(str).eq(rc.get("apply_game_type", "R")).to_numpy()
        pp = np.clip(pp + np.where(rmask, float(rc["shrink"]) * c, 0.0), 1e-6, 1 - 1e-6)
        acc = np.log(pp / (1 - pp)) * n

    p = 1 / (1 + np.exp(-(acc / n + float(b["calib_shift"]))))
    # ★test 전체를 훑는 연산은 로그 출력이라도 두지 않는다★
    #   예측에 되먹임되지 않더라도 규정 검토 시 오해 소지가 된다.
    if verbose:
        print(f" models={n}", flush=True)
    return np.clip(p, 1e-6, 1 - 1e-6)


def find_data():
    for name in ("data", "open", "."):
        d = os.path.join(HERE, name)
        if (os.path.exists(os.path.join(d, "test.csv"))
                and os.path.exists(os.path.join(d, "sample_submission.csv"))):
            return d
    raise FileNotFoundError("./data 에서 test.csv / sample_submission.csv 를 못 찾음")


def main():
    dd = find_data()
    test = pd.read_csv(os.path.join(dd, "test.csv"))
    sample = pd.read_csv(os.path.join(dd, "sample_submission.csv"))
    b = joblib.load(os.path.join(HERE, "model", "bundle.pkl"))

    bad = set(b["cols"]) & FORBIDDEN
    if bad:
        raise RuntimeError(f"[규정] 금지 컬럼 사용: {sorted(bad)}")
    if TGT in test.columns:
        raise RuntimeError("[규정] test 에 정답 컬럼이 있습니다")

    p = predict(test, b, verbose=True)
    if not np.isfinite(p).all():
        raise ValueError("예측에 NaN/Inf")
    sub = sample[[ID_COL]].copy()
    sub[TGT] = p
    out = os.path.join(HERE, "output")
    os.makedirs(out, exist_ok=True)
    sub.to_csv(os.path.join(out, "submission.csv"), index=False)
    print(f"Saved: output/submission.csv rows={len(sub)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
