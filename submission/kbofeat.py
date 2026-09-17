"""feat_F 피처 생성 - 학습과 추론이 같은 코드를 쓴다.

핵심 아이디어(레짐 조정)
  asof_pitcher_success_rate 는 통산 누적이라 ABS 도입 전후가 한 숫자에 섞여 있다.
  리그 평균이 2019년 0.565 -> 2024년 0.486 으로 변했으므로, 경력 대부분이
  옛 레짐인 베테랑은 값이 부풀려지고 신인은 그렇지 않다. 같은 0.55 라도
  실제 실력이 다르다는 뜻이다.
  각 투수의 시즌별 투구수를 학습 데이터에서 복원해 '그 경력에 해당하는 리그
  기대치'를 구하고 빼주면 레짐 혼입이 제거된다.
"""
import numpy as np
import pandas as pd

TGT = "control_success"


def feat_D(d, base):
    X = d[base].copy()
    X["count_id"] = d.balls_before * 3 + d.strikes_before
    X["count_diff"] = d.balls_before - d.strikes_before
    X["two_strike"] = (d.strikes_before == 2).astype(np.int8)
    X["three_ball"] = (d.balls_before == 3).astype(np.int8)
    for k in (1, 3, 5):
        X[f"form{k}"] = (d[f"asof_pitcher_prev{k}_game_success_rate"]
                         - d.asof_pitcher_success_rate)
        X[f"formmid{k}"] = (d[f"asof_pitcher_prev{k}_game_middle_rate"]
                            - d.asof_pitcher_middle_rate)
    X["pb_diff"] = d.asof_pitcher_success_rate - d.asof_batter_success_rate
    X["pb_mid_diff"] = d.asof_pitcher_middle_rate - d.asof_batter_middle_rate
    X["same_hand"] = (d.pitcher_hand == d.batter_hand).astype(np.int8)
    X["log_pn"] = np.log1p(d.asof_pitcher_n)
    X["log_bn"] = np.log1p(d.asof_batter_n)
    n, m = d.asof_pitcher_n, 500.0
    X["shrunk_rate"] = ((d.asof_pitcher_success_rate.fillna(0.5) * n
                         + 0.5 * m) / (n + m))
    return X


def build_last(df_fit, who="pitcher"):
    """(id, season) 별 그 시즌 마지막 시점의 누적 상태.

    다음 시즌의 '시작 상태' 로 쓴다. asof 값만 참조하므로 라벨 미사용.
    """
    g = df_fit.sort_values(f"asof_{who}_n").groupby([f"{who}_id", "season"])
    agg = {"n1": (f"asof_{who}_n", "last"),
           "s1": (f"asof_{who}_success_rate", "last"),
           "m1": (f"asof_{who}_middle_rate", "last")}
    if who == "pitcher":
        agg["r1"] = ("asof_pitcher_reverse_rate", "last")
    return g.agg(**agg)


def build_baseline_table(df_fit, last_p):
    """(pitcher_id, season) -> 그 시점 누적 경력의 리그 기대 성공률."""
    league = df_fit.groupby("season")[TGT].mean()
    t = last_p.reset_index().sort_values(["pitcher_id", "season"])
    t["prev_n1"] = t.groupby("pitcher_id")["n1"].shift(1).fillna(0.0)
    t["cnt"] = (t["n1"] - t["prev_n1"]).clip(lower=0)
    t["lg"] = t["season"].map(league).astype("f8")
    t["w"] = t["cnt"] * t["lg"]
    t["cum_w"] = t.groupby("pitcher_id")["w"].cumsum()
    t["cum_n"] = t.groupby("pitcher_id")["cnt"].cumsum()
    t["baseline"] = t["cum_w"] / t["cum_n"].replace(0, np.nan)
    return (t.set_index(["pitcher_id", "season"])[["baseline", "cum_n"]],
            league)


def feat_F(d, base, last_p, last_b, btab, league):
    """feat_D + 실패유형 분해 + 시즌 캐리오버 + 레짐 조정."""
    X = feat_D(d, base)
    s = d.asof_pitcher_success_rate
    mid = d.asof_pitcher_middle_rate
    rev = d.asof_pitcher_reverse_rate
    ball = d.asof_pitcher_ball_rate
    strike = d.asof_pitcher_strike_rate

    # --- 실패유형 분해 ---
    # 제구 실패는 (한가운데 / 크게 벗어남 / 포수 요구 반대) 3종인데
    # '크게 벗어남' 은 주어지지 않는다. 나머지로 복원한다.
    X["faroff"] = 1 - s - mid - rev
    X["fail_mix_rev"] = rev / (1 - s + 1e-9)
    X["fail_mix_mid"] = mid / (1 - s + 1e-9)
    X["fail_mix_far"] = X["faroff"] / (1 - s + 1e-9)
    X["rev_minus_mid"] = rev - mid
    X["ball_minus_strike"] = ball - strike
    X["inplay_resid"] = 1 - ball - strike

    # --- 직전 시즌 최종 상태 (2025 행이면 2024 최종) ---
    pk = pd.MultiIndex.from_arrays([d.pitcher_id, d.season - 1])
    lp = last_p.reindex(pk)
    n0 = lp["n1"].to_numpy(dtype="f8")
    s0 = lp["s1"].to_numpy(dtype="f8")
    r0 = lp["r1"].to_numpy(dtype="f8")
    m0 = lp["m1"].to_numpy(dtype="f8")
    X["prev_season_succ"] = s0
    X["prev_season_rev"] = r0
    X["carryover"] = s0 - s.to_numpy()
    X["carryover_rev"] = r0 - rev.to_numpy()

    # --- 이번 시즌 성적 = (통산 - 직전시즌말) 차분 ---
    n = d.asof_pitcher_n.to_numpy(dtype="f8")
    cnt = n - n0
    ok = cnt > 50
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.where(ok, cnt, np.nan)
        X["season_succ"] = (n * s.to_numpy() - n0 * s0) / den
        X["season_rev"] = (n * rev.to_numpy() - n0 * r0) / den
        X["season_mid"] = (n * mid.to_numpy() - n0 * m0) / den
    X["season_n"] = np.where(ok, cnt, np.nan)
    X["season_vs_career"] = X["season_succ"] - s.to_numpy()
    X["season_vs_prev"] = X["season_succ"] - s0

    # --- 레짐 조정 (핵심) ---
    bl = btab.reindex(pk)["baseline"].to_numpy(dtype="f8")
    X["career_league_baseline"] = bl
    X["skill_adj"] = s.to_numpy() - bl
    X["skill_adj_shrunk"] = X["skill_adj"] * n / (n + 500.0)
    cur = d.season.map(league).astype("f8").to_numpy()
    cur = np.where(np.isfinite(cur), cur, float(league.iloc[-1]))
    X["league_now"] = cur
    X["baseline_minus_now"] = bl - cur
    X["skill_adj_vs_now"] = X["skill_adj"] + cur
    X["season_adj"] = X["season_succ"] - cur

    bk = pd.MultiIndex.from_arrays([d.batter_id, d.season - 1])
    lb = last_b.reindex(bk)
    X["prev_season_bsucc"] = lb["s1"].to_numpy(dtype="f8")
    X["b_carryover"] = X["prev_season_bsucc"] - d.asof_batter_success_rate.to_numpy()
    return X


# 상황 전용 피처 (선수 이력 전면 제외). v5 블렌드와 상관 0.48 로 가장 낮다.
SITU_COLS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "count_id", "count_diff", "two_strike", "three_ball", "same_hand",
]


def add_mode_inter(X, d):
    """실패유형 x 상황 상호작용.

    타깃은 실패 3종(한가운데 / 크게 벗어남 / 포수 요구 반대)의 여집합이고,
    상황에 따라 어느 유형이 위험한지가 다르다.
      3볼   -> 반드시 스트라이크를 넣어야 하니 '한가운데' 가 위험
      2스트라이크 -> 유인구를 던지니 '크게 벗어남' 이 위험
    성공률 0.52 인 두 투수라도 실패 방식이 다르면 같은 상황에서 위험도가
    정반대다. success_rate 하나로는 구분이 불가능하고 유형별로 쪼개야 한다.
    (대조군인 success_rate x 상황 은 -2.8, 이쪽은 +11.4)
    """
    b = d.balls_before.to_numpy("f8")
    s = d.strikes_before.to_numpy("f8")
    mid = d.asof_pitcher_middle_rate.to_numpy("f8")
    rev = d.asof_pitcher_reverse_rate.to_numpy("f8")
    ball = d.asof_pitcher_ball_rate.to_numpy("f8")
    suc = d.asof_pitcher_success_rate.to_numpy("f8")
    far = 1 - suc - mid - rev
    press = b - s                          # +면 볼이 몰림(투수 불리)
    must = (b >= 3).astype("f8")           # 반드시 스트라이크
    waste = (s >= 2).astype("f8")          # 유인구 가능
    X = X.copy()
    X["press"] = press
    for nm, v in (("mid", mid), ("rev", rev), ("ball", ball), ("far", far)):
        X[f"x_{nm}_press"] = v * press
        X[f"x_{nm}_must"] = v * must
        X[f"x_{nm}_waste"] = v * waste
    X["risk_now"] = np.where(must > 0, mid,
                             np.where(waste > 0, far, mid * .5 + far * .5))
    X["risk_ratio"] = X["risk_now"] / (1 - suc + 1e-9)
    X["mid_minus_far"] = mid - far
    X["x_midfar_press"] = (mid - far) * press
    return X


# ======================================================================
# v8 추가: 조건부 투수 프로파일 (최신성 가중)
#
# 대회가 준 asof_pitcher_*_rate 는 전부 무조건부 통산 비율이다.
# n x rate 의 연속행 차분으로 각 투구의 실패 유형을 복원할 수 있고
# (train 내부에서만, DACON 이 명시 허용), 그로부터 '이 투수가 어떤
# 상황에서 어떻게 무너지는가' 를 만든다.
#
# 누출 차단: 각 행의 조건부 값은 '그 행의 시즌보다 이전 시즌' 에서만
# 집계한다. 2025 테스트 행은 2019~2024 전체를 최신성 가중으로 사용한다.
# 조회 키가 (투수, 그 행 자신의 상황) 이므로 평가 행 간 참조가 없다.
# ======================================================================
MODES_C = ["suc", "mid", "rev", "far"]
TYPES_C = ["fb", "br", "os"]
AXES_C = ["cnt", "two", "thr", "bh", "run"]
COND_K = 150.0
COND_LAM = 0.6
# (조회키, 상황축, 접두사, 수축강도) - 2024 홀드아웃에서 확정된 조합
#   투수만 962.1 / +투수팀 970.6 / +타자팀 980.7 / +투수손x타자팀 991.0
#   + form1(직전등판 폼) 992.1 -> v7 믹스와 결합하면 1000.3 (w=0.70)
# form1 은 h2/h1 이월 비율을 0.423 -> 0.454 로 올린다(이월 내성 개선).
# reverse 는 '포수 요구의 반대' 이므로 팀 배터리·사인 체계가 직접 관여한다.
# 키를 더 늘리면(팀쌍 등) 오히려 깎인다 - 셀당 표본이 충분한 조합만 쓴다.
COND_SPECS = [
    ("pitcher_id", AXES_C + ["form1"], "c_", 150.0),
    ("pitcher_team_id", AXES_C, "pt_", 400.0),
    ("batter_team_id", AXES_C, "bt_", 400.0),
    ("hteam", ["cnt", "two", "thr", "bh"], "ht_", 400.0),
]


def recover_modes(df):
    """연속행 차분으로 각 투구의 실패 유형 라벨을 복원한다 (train 전용)."""
    d = df.copy()
    d["_rid"] = d.row_id.str.slice(6).astype("int64")
    d = d.sort_values(["pitcher_id", "asof_pitcher_n"])
    n = d.asof_pitcher_n.to_numpy("f8")
    pid = d.pitcher_id.to_numpy()
    cs = n * d.asof_pitcher_success_rate.to_numpy("f8")
    cm = n * d.asof_pitcher_middle_rate.to_numpy("f8")
    cr = n * d.asof_pitcher_reverse_rate.to_numpy("f8")
    ok = np.r_[(pid[1:] == pid[:-1]) & (n[1:] - n[:-1] == 1), False]
    S = np.r_[cs[1:] - cs[:-1], np.nan] > 0.5
    M = np.r_[cm[1:] - cm[:-1], np.nan] > 0.5
    R = np.r_[cr[1:] - cr[:-1], np.nan] > 0.5
    d["ok"] = ok
    d["suc"] = np.where(ok, S, np.nan)
    d["mid"] = np.where(ok, M, np.nan)
    d["rev"] = np.where(ok, R, np.nan)
    d["far"] = np.where(ok, (~S) & (~M) & (~R), np.nan)
    # 구종도 같은 차분으로 복원된다 (pitchmix_n == asof_pitcher_n 확인됨).
    # 추론에서 '현재 투구의 구종' 은 금지지만, 투수의 과거 구종 성향과
    # 구종별 실패율은 과거 이력이므로 허용된다.
    mn = d.asof_pitcher_pitchmix_n.to_numpy("f8")
    for nm, col in (("fb", "fastball"), ("br", "breaking"), ("os", "offspeed")):
        c = mn * d[f"asof_pitcher_{col}_rate"].to_numpy("f8")
        d[nm] = np.where(ok, np.r_[c[1:] - c[:-1], np.nan] > 0.5, np.nan)
    return d.sort_values("_rid").drop(columns="_rid")


def cond_add_keys(d):
    """조회용 파생 키. 그 행 자신의 컬럼만 조합한다."""
    d = d.copy()
    d["hteam"] = (d.pitcher_hand.astype("int64") * 100
                  + d.batter_team_id.astype("int64"))
    return d


def cond_ctx(d, kind):
    b = d.balls_before.to_numpy("i8")
    s = d.strikes_before.to_numpy("i8")
    if kind == "cnt":
        return np.where(b > s, 2, np.where(s > b, 0, 1))
    if kind == "two":
        return (s >= 2).astype("i8")
    if kind == "thr":
        return (b >= 3).astype("i8")
    if kind == "bh":
        return d.batter_hand.to_numpy("i8")
    if kind == "run":
        return (d.num_runners_on.to_numpy() > 0).astype("i8")
    if kind == "form1":
        # 직전 등판이 통산 대비 부진/보통/호조. 그 행 자신의 컬럼만 쓴다.
        v = (d.asof_pitcher_prev1_game_success_rate.to_numpy("f8")
             - d.asof_pitcher_success_rate.to_numpy("f8"))
        return np.where(np.isnan(v), 1,
                        np.where(v < -0.05, 0, np.where(v > 0.05, 2, 1)))
    if kind == "cnt12":
        # ★볼-스트라이크 12칸★ 0-0 부터 3-2 까지. cnt/two/thr 3축을 대체한다.
        # 3축은 0-2 와 1-2, 3-0 과 3-1 을 구분하지 못했다.
        # 그 행 자신의 balls_before / strikes_before 만 쓴다.
        return (np.clip(d.balls_before.to_numpy("i8"), 0, 3) * 3
                + np.clip(d.strikes_before.to_numpy("i8"), 0, 2)).astype("i8")
    if kind == "run3":
        # 없음 / 1루만 / 득점권(2루 이상). 1루는 슬라이드스텝이라는 물리 효과,
        # 득점권은 사인 복잡화 + 압박이라는 다른 메커니즘이다.
        r1 = d.runner_on_1b.to_numpy("f8") > 0
        sp = ((d.runner_on_2b.to_numpy("f8") > 0)
              | (d.runner_on_3b.to_numpy("f8") > 0))
        return np.where(sp, 2, np.where(r1, 1, 0)).astype("i8")
    raise ValueError(kind)


def _wmat(cols, lam, out_cols=None):
    """W[t, s] = lam^(t-s) for s < t else 0.

    out_cols 를 주면 (출력시즌 x 입력시즌) 직사각 행렬이 된다. 학습 데이터에
    없는 시즌(예: 2025) 의 슬라이스를 만들기 위한 것 - 그 시즌보다 **이전**
    시즌만 s<t 조건으로 들어가므로 누출은 여전히 없다.
    """
    t = (cols if out_cols is None else out_cols).astype("f8").reshape(-1, 1)
    s = cols.astype("f8").reshape(1, -1)
    return np.where(t - s > 0, lam ** (t - s), 0.0)


def cond_tables(src, who, axes, lam=COND_LAM, extra_cols=()):
    """(조회키, 상황) 별 최신성 가중 누적. src 는 recover_modes 를 거친 학습 데이터.

    시즌 t 의 값 = sum_{s<t} lam^(t-s) * (그 시즌 집계).
    s<t 이므로 그 시즌 자신의 라벨은 절대 들어가지 않는다.

    ★extra_cols★ 학습에 없는 시즌(예: 2025)의 슬라이스를 추가로 만든다.
      이게 없으면 freeze(2025) 가 cols.max()=2024 로 떨어지고, 2024 컬럼은
      정의상 s<2024 라서 **직전 시즌 한 해가 통째로 버려진다**(가중치 1.00 짜리).
      2026-08-30 실측으로 확인된 결함. 검증·배포 양쪽에 대칭으로 존재했다.
    """
    f = src[src.ok].copy()
    g = f.groupby([who, "season"])[MODES_C].agg(["sum", "count"])
    cnt = g[(MODES_C[0], "count")].unstack(fill_value=0)
    in_cols = cnt.columns.to_numpy()
    cols = np.array(sorted(set(int(c) for c in in_cols)
                           | set(int(c) for c in extra_cols)))
    W = _wmat(in_cols, lam, out_cols=cols)
    cv0 = cnt.to_numpy("f8") @ W.T
    T = {"_cols": cols, "_own": {}}
    for m in MODES_C:
        P_ = g[(m, "sum")].unstack(fill_value=0).reindex(columns=in_cols,
                                                         fill_value=0)
        pv = P_.to_numpy("f8") @ W.T
        T["_own"][m] = pd.DataFrame(pv / np.where(cv0 > 0, cv0, np.nan),
                                    index=cnt.index, columns=cols)
    for kind in axes:
        f["_k"] = cond_ctx(f, kind)
        gg = f.groupby([who, "_k", "season"])[MODES_C].agg(["sum", "count"])
        tab = {}
        for m in MODES_C:
            S_ = gg[(m, "sum")].unstack(fill_value=0).reindex(
                columns=in_cols, fill_value=0)
            C_ = gg[(m, "count")].unstack(fill_value=0).reindex(
                columns=in_cols, fill_value=0)
            tab[m] = (pd.DataFrame(S_.to_numpy("f8") @ W.T, index=S_.index,
                                   columns=cols).stack(dropna=False),
                      pd.DataFrame(C_.to_numpy("f8") @ W.T, index=C_.index,
                                   columns=cols).stack(dropna=False))
        T[kind] = tab
    return T


def cond_apply(X, d, T, who, axes, pre, K):
    """학습용 - 각 행의 시즌에 해당하는 슬라이스를 쓴다."""
    X = X.copy()
    key = d[who].to_numpy()
    ssn = d.season.to_numpy()
    cols = T["_cols"]
    col = np.where(np.isin(ssn, cols), ssn, cols.max())
    own = {}
    for m in MODES_C:
        st = T["_own"][m].stack(dropna=False)
        own[m] = st.reindex(
            pd.MultiIndex.from_arrays([key, col])).to_numpy("f8")
        X[f"{pre}own_{m}"] = own[m]
    for kind in axes:
        kv = cond_ctx(d, kind)
        idx = pd.MultiIndex.from_arrays([key, kv, col])
        for m in MODES_C:
            sv_, cv_ = T[kind][m]
            sv = sv_.reindex(idx).to_numpy("f8")
            cv = cv_.reindex(idx).to_numpy("f8")
            pr = own[m]
            X[f"{pre}{kind}_{m}"] = ((np.nan_to_num(sv) + K * pr) /
                                     (np.nan_to_num(cv) + K)) - pr
    return X


def cond_freeze(T, axes, target_season):
    """추론용 - 목표 시즌 슬라이스만 뽑아 소형 표로 동결한다.

    평가 시 모든 행이 같은 시즌(2025)이므로 한 장이면 충분하다.
    조회는 (그 행의 키, 그 행 자신의 상황) 만 쓴다.
    """
    cols = T["_cols"]
    col = target_season if target_season in cols else cols.max()
    F = {"season_used": int(col), "own": {}, "cell": {}}
    for m in MODES_C:
        F["own"][m] = T["_own"][m][col].astype("f4")
    for kind in axes:
        F["cell"][kind] = {
            m: (T[kind][m][0].xs(col, level=-1).astype("f4"),
                T[kind][m][1].xs(col, level=-1).astype("f4"))
            for m in MODES_C}
    return F


def cond_apply_frozen(X, d, F, who, axes, pre, K):
    """추론용 - 동결표로 학습과 동일한 피처를 만든다."""
    X = X.copy()
    key = d[who].to_numpy()
    own = {}
    for m in MODES_C:
        own[m] = F["own"][m].reindex(key).to_numpy("f8")
        X[f"{pre}own_{m}"] = own[m]
    for kind in axes:
        kv = cond_ctx(d, kind)
        idx = pd.MultiIndex.from_arrays([key, kv])
        for m in MODES_C:
            sv_, cv_ = F["cell"][kind][m]
            sv = sv_.reindex(idx).to_numpy("f8")
            cv = cv_.reindex(idx).to_numpy("f8")
            pr = own[m]
            X[f"{pre}{kind}_{m}"] = ((np.nan_to_num(sv) + K * pr) /
                                     (np.nan_to_num(cv) + K)) - pr
    return X


def cond_build_all(src, lam=COND_LAM, extra_cols=()):
    """COND_SPECS 전부에 대해 표를 만든다.

    extra_cols 에 예측 대상 시즌을 넣어야 그 시즌 슬라이스가 직전 시즌까지
    포함한다. 안 넣으면 freeze 가 cols.max() 로 떨어져 한 시즌을 잃는다.
    """
    return {who: cond_tables(src, who, ax, lam, extra_cols)
            for who, ax, _, _ in COND_SPECS}


def cond_apply_all(X, d, tabs):
    for who, ax, pre, K in COND_SPECS:
        X = cond_apply(X, d, tabs[who], who, ax, pre, K)
    return X


def cond_freeze_all(tabs, target_season):
    return {who: cond_freeze(tabs[who], ax, target_season)
            for who, ax, _, _ in COND_SPECS}


def cond_apply_frozen_all(X, d, froz):
    for who, ax, pre, K in COND_SPECS:
        X = cond_apply_frozen(X, d, froz[who], who, ax, pre, K)
    return X


# ----------------------------------------------------------------------
# v9 추가: 구종 성향 피처
#   mix(구종 | 투수, 카운트)  이 투수가 이 카운트에서 무엇을 던지는가
#   suc(구종 | 투수)         이 투수가 구종별로 어떻게 제구를 놓치는가
#   기대성공 = sum_t mix(t) x suc(t)
# 현재 투구의 구종은 쓰지 않는다. 전부 투수 단위 과거 이력이다.
# ----------------------------------------------------------------------
TYPE_K = 150.0


def type_tables(src, lam=COND_LAM):
    f = src[src.ok].copy()
    g = f.groupby(["pitcher_id", "season"])["suc"].agg(["sum", "count"])
    cn = g["count"].unstack(fill_value=0)
    cols = cn.columns.to_numpy()
    W = _wmat(cols, lam)
    cv = cn.to_numpy("f8") @ W.T
    T = {"_cols": cols,
         "own": pd.DataFrame(g["sum"].unstack(fill_value=0).to_numpy("f8") @ W.T
                             / np.where(cv > 0, cv, np.nan),
                             index=cn.index, columns=cols),
         "suc": {}, "mix": {}}
    f["_k"] = cond_ctx(f, "cnt")
    for t in TYPES_C:
        ft = f[f[t].astype("f8") > 0.5]
        gt = ft.groupby(["pitcher_id", "season"])["suc"].agg(["sum", "count"])
        S_ = gt["sum"].unstack(fill_value=0).reindex(columns=cols, fill_value=0)
        C_ = gt["count"].unstack(fill_value=0).reindex(columns=cols,
                                                       fill_value=0)
        T["suc"][t] = (pd.DataFrame(S_.to_numpy("f8") @ W.T, index=S_.index,
                                    columns=cols),
                       pd.DataFrame(C_.to_numpy("f8") @ W.T, index=C_.index,
                                    columns=cols))
        gm = f.groupby(["pitcher_id", "_k", "season"])[t].agg(["sum", "count"])
        Sm = gm["sum"].unstack(fill_value=0).reindex(columns=cols, fill_value=0)
        Cm = gm["count"].unstack(fill_value=0).reindex(columns=cols,
                                                       fill_value=0)
        T["mix"][t] = (pd.DataFrame(Sm.to_numpy("f8") @ W.T, index=Sm.index,
                                    columns=cols).stack(dropna=False),
                       pd.DataFrame(Cm.to_numpy("f8") @ W.T, index=Cm.index,
                                    columns=cols).stack(dropna=False))
    return T


def _type_core(X, d, own_s, suc_tabs, mix_tabs, K=TYPE_K):
    X = X.copy()
    X_exp = np.zeros(len(d))
    wsum = np.zeros(len(d))
    for t in TYPES_C:
        st = suc_tabs[t]
        sc = (np.nan_to_num(st[0]) + K * np.nan_to_num(own_s, nan=0.5)) / (
            np.nan_to_num(st[1]) + K)
        X[f"ty_suc_{t}"] = sc - own_s
        mx = (np.nan_to_num(mix_tabs[t][0]) + K * 0.33) / (
            np.nan_to_num(mix_tabs[t][1]) + K)
        X[f"ty_mix_{t}"] = mx
        X_exp += mx * np.nan_to_num(sc, nan=0.5)
        wsum += mx
    X["ty_exp_suc"] = X_exp / np.maximum(wsum, 1e-9)
    X["ty_exp_vs_own"] = X["ty_exp_suc"] - own_s
    return X


def type_apply(X, d, T, K=TYPE_K):
    pid = d.pitcher_id.to_numpy()
    ssn = d.season.to_numpy()
    cols = T["_cols"]
    col = np.where(np.isin(ssn, cols), ssn, cols.max())
    ip = pd.MultiIndex.from_arrays([pid, col])
    own_s = T["own"].stack(dropna=False).reindex(ip).to_numpy("f8")
    kv = cond_ctx(d, "cnt")
    im = pd.MultiIndex.from_arrays([pid, kv, col])
    suc = {t: (T["suc"][t][0].stack(dropna=False).reindex(ip).to_numpy("f8"),
               T["suc"][t][1].stack(dropna=False).reindex(ip).to_numpy("f8"))
           for t in TYPES_C}
    mix = {t: (T["mix"][t][0].reindex(im).to_numpy("f8"),
               T["mix"][t][1].reindex(im).to_numpy("f8")) for t in TYPES_C}
    return _type_core(X, d, own_s, suc, mix, K)


def type_freeze(T, target_season):
    cols = T["_cols"]
    col = target_season if target_season in cols else cols.max()
    return {"season_used": int(col),
            "own": T["own"][col].astype("f4"),
            "suc": {t: (T["suc"][t][0][col].astype("f4"),
                        T["suc"][t][1][col].astype("f4")) for t in TYPES_C},
            "mix": {t: (T["mix"][t][0].xs(col, level=-1).astype("f4"),
                        T["mix"][t][1].xs(col, level=-1).astype("f4"))
                    for t in TYPES_C}}


def type_apply_frozen(X, d, F, K=TYPE_K):
    pid = d.pitcher_id.to_numpy()
    own_s = F["own"].reindex(pid).to_numpy("f8")
    kv = cond_ctx(d, "cnt")
    im = pd.MultiIndex.from_arrays([pid, kv])
    suc = {t: (F["suc"][t][0].reindex(pid).to_numpy("f8"),
               F["suc"][t][1].reindex(pid).to_numpy("f8")) for t in TYPES_C}
    mix = {t: (F["mix"][t][0].reindex(im).to_numpy("f8"),
               F["mix"][t][1].reindex(im).to_numpy("f8")) for t in TYPES_C}
    return _type_core(X, d, own_s, suc, mix, K)
