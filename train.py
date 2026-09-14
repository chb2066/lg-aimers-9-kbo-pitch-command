"""리더보드 1146.91점 제출본을 처음부터 다시 만드는 학습 스크립트.

실행
    python train.py           전체 학습 (12스레드 기준 약 3시간)
    python train.py --fast    회귀·분류 모델을 하나씩만 학습한다.
                              output/ 에 이전 실행의 잔차 보정표가 있으면 재사용한다.

필요한 파일
    data/train.csv            대회 학습 데이터
    submission/               제출 zip 에 들어가는 추론 코드. 그대로 복사해 담는다.

단계
    1. 시간순 검증 예측. 2022, 2023, 2024 시즌을 각각 그 이전 시즌으로 학습해 예측한다.
    2. 검증 예측의 잔차로 잔차 보정표를 만든다. 정규리그 경기만 쓴다.
    3. 전체 데이터로 최종 모델을 학습한다. 회귀 3개와 5클래스 분류 2개.
    4. 대역 데이터로 평균 보정 상수를 풀고, 행 독립성을 검사한 뒤 output/submit.zip 을 만든다.

평가 데이터의 각 행은 독립적으로 처리된다. 이번 시즌 기록의 기준값, 경기유형 재코딩,
팀13 지시자, 잔차 보정표는 모두 학습 데이터로 만들어 저장해 두고, 각 행이 자기 값으로만
조회한다.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import warnings
import zipfile

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "submission"))

import joblib                                                  # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
from catboost import CatBoostClassifier, CatBoostRegressor     # noqa: E402

import feat_ext61 as FE                                        # noqa: E402
import kbofeat as KF                                           # noqa: E402

DATA = os.environ.get("KBO_DATA_DIR", os.path.join(ROOT, "data"))
SRC = os.path.join(ROOT, "submission")
OUT = os.path.join(ROOT, "output")
STAGE = os.path.join(OUT, "package")
PROXY = os.path.join(OUT, "proxy_test.csv")

# ── 상수 ────────────────────────────────────────────────────────────
ALPHA = 50.0                 # current-season 평활
NEW_ERA = 2023               # F 체제 전환 시점
W_MC = 0.75                  # 다중분류 블렌드 가중
CLIP = (0.38, 0.70)          # 출력 클리핑
RES_K, RES_SHRINK = 3200.0, 1.00
RES_KEYS = [["pitcher_id", "bh"], ["count_state"], ["p_cnt"], ["bs"]]
OOF_SEASONS = (2022, 2023, 2024)
REG_SEEDS, MC_SEEDS = (42, 202, 777), (42, 202)
TARGET_R, PROXY_OFF = 0.48051, 0.0   # derive_target.py 로 계산한 값
CB = dict(depth=6, iterations=900, learning_rate=0.03, l2_leaf_reg=3.0,
          allow_writing_files=False, verbose=False, thread_count=-1)


def _logit(p):
    p = np.clip(np.asarray(p, "f8"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ── 피처 ────────────────────────────────────────────────────────────
def recode_game_type(d, X):
    """R / F_old / F_new. 그 행 자신의 season, game_type 만."""
    isF = d["game_type"].astype(str).eq("F").to_numpy()
    new = pd.to_numeric(d["season"], errors="coerce").to_numpy("f8") >= NEW_ERA
    X = X.copy()
    X["game_type"] = np.where(isF, np.where(new, "F_new", "F_old"), "R")
    return X


def add_team13(d, X):
    """팀13 체제 지시자. 그 행 자신의 team_id / season / game_month / game_type 만.
    submission/script.py 의 add_team13 과 같은 식이어야 한다."""
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


def baselines(df):
    """학습 끝 시점 누적 — current-season 계산의 동결 기준."""
    return (FE.build_end_baseline(
                df, "pitcher_id", "asof_pitcher_n",
                {"success": "asof_pitcher_success_rate",
                 "reverse": "asof_pitcher_reverse_rate",
                 "middle": "asof_pitcher_middle_rate"}),
            FE.build_end_baseline(
                df, "batter_id", "asof_batter_n",
                {"success": "asof_batter_success_rate",
                 "middle": "asof_batter_middle_rate"}))


def make_frame(d, pb=None, bb=None, train_mode=True):
    FE.SMOOTH_ALPHA = ALPHA
    X = (FE.add_train_cs(d) if train_mode
         else FE.add_test_cs(d, pb, bb))[FE.FEATURE_COLS].copy()
    return add_team13(d, recode_game_type(d, X))


def to_cb(X, cats):
    A = X.copy()
    for c in cats:
        A[c] = A[c].where(A[c].notna(), "__NA__").astype(str)
    for c in A.columns:
        if c not in cats:
            A[c] = pd.to_numeric(A[c], errors="coerce")
    return A


# ── 다중분류 라벨 ────────────────────────────────────────────────────
def modes_label(d):
    """suc/mid/rev/far 복원 -> 5클래스. 복원 실패는 -1(가중 0 으로 처리).
       train.csv 내부에서 asof_pitcher_n x rate 의 연속행 차분으로 만든다."""
    r = KF.recover_modes(d)
    r = r.loc[d.index] if r.index.equals(d.index) else r.set_index(d.index)
    ok = r["ok"].to_numpy()
    S = r["suc"].fillna(0).to_numpy().astype(bool)
    M = r["mid"].fillna(0).to_numpy().astype(bool)
    V = r["rev"].fillna(0).to_numpy().astype(bool)
    F = r["far"].fillna(0).to_numpy().astype(bool)
    lab = np.full(len(d), -1, dtype="int64")
    lab[ok & S] = 0
    lab[ok & ~S & M & ~V] = 1
    lab[ok & ~S & ~M & V] = 2
    lab[ok & ~S & M & V] = 3
    lab[ok & ~S & F] = 4
    lab[ok & (lab < 0)] = 4
    return lab


# ── 잔차 ────────────────────────────────────────────────────────────
def res_context(d):
    o = d.copy()
    b = pd.to_numeric(o.balls_before, errors="coerce").astype("Int64").astype(str)
    s = pd.to_numeric(o.strikes_before, errors="coerce").astype("Int64").astype(str)
    o["count_state"] = b + "-" + s
    o["bh"] = o.batter_hand.where(o.batter_hand.notna(), "__NA__").astype(str)
    o["bs"] = o.base_state.where(o.base_state.notna(), "__NA__").astype(str)
    o["p_cnt"] = o.pitcher_id.astype(str) + "|" + o["count_state"]
    return o


def fit_residual_maps(src):
    """시즌별로 중심화한다. 전역 레벨은 목표평균이 잡고 세그먼트 편차만 남긴다."""
    src = src.copy()
    src["r"] = src["r"] - src.groupby("season")["r"].transform("mean")
    out = {}
    for keys in RES_KEYS:
        a = src.groupby(keys, observed=True)["r"].agg(["sum", "count"]).reset_index()
        a["c"] = a["sum"] / (a["count"] + RES_K)
        if len(keys) == 1:
            out[keys[0]] = {str(r[keys[0]]): float(r.c) for _, r in a.iterrows()}
        else:
            out["ph"] = {f"{int(r[keys[0]])}|{r[keys[1]]}": float(r.c)
                         for _, r in a.iterrows()}
    return dict(use=True, k=RES_K, shrink=RES_SHRINK, apply_game_type="R",
                ph=out["ph"], ct=out["count_state"], pc=out["p_cnt"], bs=out["bs"])


def ensure_proxy(df):
    """프록시 = 직전 시즌(2024) 행을 평가 시즌(2025)으로 재라벨한 것.

    목표평균 보정 상수를 **학습 시점에** 풀기 위한 대역 데이터다. 실제 test.csv
    는 5행짜리 샘플이라 평균을 잴 수 없고, 추론 때 test 통계를 재면 규정 위반이다.
    정답 컬럼은 제거하므로 라벨 정보가 들어가지 않는다.
    """
    if os.path.exists(PROXY):
        return
    os.makedirs(os.path.dirname(PROXY), exist_ok=True)
    last = int(df.season.max())
    q = df[df.season == last].reset_index(drop=True).copy()
    q = q.drop(columns=["control_success"])
    q["season"] = last + 1
    q["row_id"] = ["TEST_%06d" % (i + 1) for i in range(len(q))]
    q.to_csv(PROXY, index=False)
    print(f"  프록시 생성: {PROXY}  ({len(q):,}행, season {last} -> {last+1})",
          flush=True)


def blend(pr_list, pm_list):
    """로짓공간 가중평균. 그룹 안에서는 평균, 그룹 간에는 W_MC 가중."""
    lr = np.mean([_logit(p) for p in pr_list], axis=0)
    lm = np.mean([_logit(p) for p in pm_list], axis=0)
    return np.clip(1 / (1 + np.exp(-((1 - W_MC) * lr + W_MC * lm))), *CLIP)


# ── 파이프라인 ──────────────────────────────────────────────────────
def main():
    t0 = time.time()
    fast = "--fast" in sys.argv
    reg_seeds = (42,) if fast else REG_SEEDS
    mc_seeds = (42,) if fast else MC_SEEDS
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    y = df["control_success"].to_numpy("f8")
    print(f"train {df.shape}   모드: {'빠른 재현(--fast)' if fast else '전체'}",
          flush=True)
    ensure_proxy(df)

    # [1] walk-forward OOF — 잔차맵 재료. 각 fold 는 그 시즌 이전만 학습한다.
    #     --fast 는 이 단계를 건너뛰고 기존 제출본의 잔차맵을 재사용한다.
    RES = None
    if fast:
        prev = os.path.join(STAGE, "model", "bundle.pkl")
        if os.path.exists(prev):
            RES = joblib.load(prev)["residual"]
            print(f"\n[1/4] OOF 건너뜀 — 기존 잔차맵 재사용 "
                  f"(K={RES['k']:.0f} sh={RES['shrink']:.2f})", flush=True)
        else:
            print("\n[1/4] 기존 잔차맵 없음 -> OOF 를 계산한다", flush=True)
            fast = False
    if RES is None:
        print("\n[1/4] walk-forward OOF", flush=True)
    oof = np.full(len(df), np.nan)
    for s in (() if RES is not None else OOF_SEASONS):
        t1 = time.time()
        m_tr, m_va = (df.season < s).to_numpy(), (df.season == s).to_numpy()
        tr, va = df[m_tr], df[m_va]
        pb, bb = baselines(tr)
        Xt = make_frame(tr, train_mode=True)
        Xv = make_frame(va, pb, bb, train_mode=False)[Xt.columns]   # 추론 경로와 같은 방식
        cats = [c for c in FE.CATEGORICAL_COLS if c in Xt.columns]
        A, B = to_cb(Xt, cats), to_cb(Xv, cats)
        pr = CatBoostRegressor(loss_function="RMSE", random_seed=42, **CB)
        pr.fit(A, y[m_tr], cat_features=cats)
        p_reg = np.clip(pr.predict(B), 1e-6, 1 - 1e-6)
        lab = modes_label(tr)
        w = (lab >= 0).astype("f8")          # 복원 실패 행은 빼지 않고 가중 0
        pc = CatBoostClassifier(loss_function="MultiClass", classes_count=5,
                                random_seed=42, **CB)
        pc.fit(A, np.where(lab < 0, 0, lab), cat_features=cats, sample_weight=w)
        p_mc = np.clip(pc.predict_proba(B)[:, 0], 1e-6, 1 - 1e-6)
        oof[m_va] = blend([p_reg], [p_mc])
        print(f"  {s} 완료 ({time.time()-t1:.0f}s)", flush=True)

    # [2] 잔차맵 — game_type R 만(F 는 2023 에 체제가 뒤집혀 오염된다)
    print("\n[2/4] 잔차맵", flush=True)
    mm = np.isfinite(oof) & df.game_type.astype(str).eq("R").to_numpy()
    src = res_context(df[mm])
    src["r"] = y[mm] - oof[mm]
    RES = fit_residual_maps(src)
    print("  " + " / ".join(f"{k}:{len(RES[k])}" for k in ("ph", "ct", "pc", "bs")),
          flush=True)

    # [3] 최종 모델 — 전체 데이터
    print("\n[3/4] 최종 학습", flush=True)
    PB, BB = baselines(df)
    X = make_frame(df, train_mode=True)
    cols = list(X.columns)
    cats = [c for c in FE.CATEGORICAL_COLS if c in cols]
    print(f"  피처 {len(cols)}개 (신규 {[c for c in cols if c not in FE.FEATURE_COLS]})",
          flush=True)
    A = to_cb(X, cats)

    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(os.path.join(STAGE, "model"))
    for f in ("script.py", "feat_ext61.py", "kbofeat.py", "feat_anchored.py",
              "requirements.txt"):
        shutil.copy(os.path.join(SRC, f), os.path.join(STAGE, f))

    files = []
    for sd in reg_seeds:
        t1 = time.time()
        m = CatBoostRegressor(loss_function="RMSE", random_seed=sd, **CB)
        m.fit(A, y, cat_features=cats)
        m.save_model(os.path.join(STAGE, "model", f"reg_s{sd}.cbm"))
        files.append((f"reg_s{sd}.cbm", "RMSE"))
        print(f"    reg_s{sd} ({time.time()-t1:.0f}s)", flush=True)
    lab = modes_label(df)
    w = (lab >= 0).astype("f8")
    print(f"  다중분류 라벨 유효 {int(w.sum()):,}/{len(w):,}  "
          f"클래스 분포 {np.bincount(lab[w > 0].astype(int), minlength=5)}", flush=True)
    for sd in mc_seeds:
        t1 = time.time()
        m = CatBoostClassifier(loss_function="MultiClass", classes_count=5,
                               random_seed=sd, **CB)
        m.fit(A, np.where(lab < 0, 0, lab), cat_features=cats, sample_weight=w)
        m.save_model(os.path.join(STAGE, "model", f"mc_s{sd}.cbm"))
        files.append((f"mc_s{sd}.cbm", "MultiClass"))
        print(f"    mc_s{sd} ({time.time()-t1:.0f}s)", flush=True)

    joblib.dump(dict(cols=cols, cats=cats,
                     feats=dict(recode_gt=True, team13=True),
                     cond_specs=None, cond_frozen=None, anchors=None,
                     alpha=ALPHA, models=files,
                     blend_w={"RMSE": 1 - W_MC, "MultiClass": W_MC},
                     clip=CLIP, p_base=PB, b_base=BB,
                     residual=RES, calib_shift=0.0,
                     target_provenance=dict(TARGET_R=TARGET_R, PROXY_OFF=PROXY_OFF,
                                            source="train.csv only")),
                os.path.join(STAGE, "model", "bundle.pkl"), compress=3)

    # [4] 목표평균 보정 + 감사 + 패키징
    print("\n[4/4] 보정 + 감사", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mod_fin", os.path.join(STAGE, "script.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    prox = pd.read_csv(PROXY)
    bd = joblib.load(os.path.join(STAGE, "model", "bundle.pkl"))
    lg = _logit(mod.predict(prox, bd))
    goal, lo, hi = TARGET_R + PROXY_OFF, -12.0, 12.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lg + mid)))).mean() < goal:
            lo = mid
        else:
            hi = mid
    sh = (lo + hi) / 2
    bd["calib_shift"] = float(sh)
    joblib.dump(bd, os.path.join(STAGE, "model", "bundle.pkl"), compress=3)
    bd = joblib.load(os.path.join(STAGE, "model", "bundle.pkl"))
    p = mod.predict(prox, bd)
    print(f"  shift {sh:+.5f} -> 착지 {p.mean():.5f} (목표 {goal:.5f})  "
          f"범위 {p.min():.4f}~{p.max():.4f}", flush=True)

    # 행 독립성 — 부분집합·순서·이웃을 바꿔도 같은 값이어야 한다
    rng = np.random.default_rng(0)
    smp = prox.iloc[rng.choice(len(prox), 1500, replace=False)].reset_index(drop=True)
    base = dict(zip(smp.row_id, mod.predict(smp, bd)))
    shf = smp.sample(frac=1.0, random_state=7).reset_index(drop=True)
    r5 = dict(zip(shf.row_id, mod.predict(shf, bd)))
    r13 = {}
    for i in range(3):
        q = smp.iloc[i::3].reset_index(drop=True)
        r13.update(zip(q.row_id, mod.predict(q, bd)))
    one = smp.iloc[[0]].reset_index(drop=True)
    nb = pd.concat([one, prox.nsmallest(700, "asof_pitcher_success_rate")]
                   ).reset_index(drop=True)
    t13n = pd.concat([one, prox[(prox.pitcher_team_id == 13)
                                | (prox.batter_team_id == 13)].head(700)]
                     ).reset_index(drop=True)
    d = [abs(base[one.row_id.iloc[0]] - mod.predict(one, bd)[0]),
         max(abs(base[k] - r5[k]) for k in base),
         max(abs(base[k] - r13[k]) for k in base),
         abs(base[one.row_id.iloc[0]] - mod.predict(nb, bd)[0]),
         abs(base[one.row_id.iloc[0]] - mod.predict(t13n, bd)[0])]
    ok = max(d) < 1e-9
    print(f"  행독립 R1 {d[0]:.1e} / R5 {d[1]:.1e} / R13 {d[2]:.1e} / "
          f"이웃 {d[3]:.1e} / 팀13이웃 {d[4]:.1e}  {'통과' if ok else '실패'}",
          flush=True)
    if not ok:
        raise RuntimeError("행 독립성 감사 실패 — 패키징 중단")
    bad = set(cols) & mod.FORBIDDEN
    if bad:
        raise RuntimeError(f"금지 컬럼 {sorted(bad)}")
    print("  금지 컬럼 없음", flush=True)

    out = os.path.join(OUT, "submit.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(STAGE):
            for f in fs:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, STAGE).replace(os.sep, "/")
                if rel.startswith("output/") or "__pycache__" in rel:
                    continue
                z.write(fp, rel)
    print(f"\n-> {out}  {os.path.getsize(out)/1e6:.1f} MB  "
          f"(총 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
