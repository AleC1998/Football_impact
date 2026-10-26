"""Paper's home/TV model, week-block bootstrap and frozen-baseline tests.

Edit the example input filenames in model_core.py and below. One outcome is
fitted per invocation. --bootstrap-only skips permutation tests; --build-only
checks the design. Original fitting calculations are retained.
Source: Model/T_home_tv_model.py and Model/T_home_tv_main_results.py.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
import contextlib
import io
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.special import xlogy

import model_core as M
from model_core import write_audit, NuisanceProjector

# ---------------------------------------------------------------- settings --
# Here load the proper fixture TV-status file (not supplied).
# Columns: date, home, away, televised (True/False/unknown), kickoff (HH:MM or blank).
TV_LISTINGS_PATH = Path(os.environ.get("T_TV_LISTINGS_FILE", str(M.DATA / "example_broadcasts.csv")))

# Match the supervisor's analysis: retain unknown-TV fixtures as non-televised.
# Raw listing labels remain unknown; this modelling policy is saved in metadata.
# Explicit sensitivity overrides "exclude" / "televised" remain available.
TV_UNKNOWN_AS = os.environ.get("T_TV_UNKNOWN_AS", "not_televised")

# Home exposure H for a home match of club k:
#   "W"          W_k        stadium catchment only (default)
#   "S_times_W"  S_k * W_k  home supporters inside the catchment; S and W peak in
#                           the same LTLA, so the product is extremely heavy-tailed
#                           (max/mean ~550x) and 1 + D_s*H > 0 then forbids almost
#                           any negative D_s
#   "S"          S_k        home supporters wherever they live
HOME_EXPOSURE = "W"

# Scale S by the club's allocation of the assumed TV audience pool?
# Viewing scales with how many supporters a club has; attendance is capped by
# stadium capacity, so by default the home term is left unweighted.
_tv_weight_setting = os.environ.get('T_TV_SUPPORTER_WEIGHTS', '1')
if _tv_weight_setting not in {'0', '1'}:
    raise ValueError('T_TV_SUPPORTER_WEIGHTS must be 0 or 1')
TV_USE_SUPPORTER_WEIGHTS = _tv_weight_setting == '1'
HOME_USE_SUPPORTER_WEIGHTS = False

# True: Model C baseline with a date fixed effect (identification of C_m and
# D_s comes only from the cross-LTLA contrast within each date).
# False: Model A baseline (no date effect; national date-level variation can
# then leak into the football terms).
USE_DATE_EFFECT = True

# Each random draw applies BOTH a TV-label permutation among fixtures and a
# home-label permutation among home-eligible team-events. Confidence regions invert
# a locally linearized efficient-score test (football_permutation.py), with
# spline/weekday/date nuisance directions projected out. Neutral fixtures have
# no home assignment. These are approximate simultaneous, not profile, CIs.
N_PERMUTATIONS = int(os.environ.get("T_N_PERMUTATIONS", 999))
PERMUTE_TV = True
PERMUTE_HOME = True
PERMUTATION_SEED = 11

# Groups within which labels are shuffled (TV labels among fixtures, home labels
# among team-events):
#   "date"           same date
#   "date_and_slot"  same date AND same kickoff slot (<14:00, 14:00-16:59, 17:00+);
#                    fixtures without a kickoff time (cup ties) are not shuffled
#   "week"           same ISO week (Monday-Sunday)
#   "dow_month"      same day of the week within the same calendar month
# The date effect delta_t absorbs whatever is common to a date, so groups wider
# than a date remain valid under the model (see T_home_tv_permutation_groups.py).
# TV: within-date mixing is poor (few dates have both televised and
# non-televised fixtures); "week" moves ~29% of TV labels per draw against ~18%
# within date or ~19% within dow_month, and "date_and_slot" moves almost none.
# Home: about half the team-events on any date are home, so every grouping mixes
# well; "date" is the most conservative choice and loses almost nothing.
TV_PERMUTATION_WITHIN = os.environ.get("T_TV_PERMUTATION_WITHIN", "week")
HOME_PERMUTATION_WITHIN = os.environ.get("T_HOME_PERMUTATION_WITHIN", "date")
PERMUTATION_GROUPINGS = ("date", "date_and_slot", "week", "dow_month")

# Keep fixtures with at least one supported club (see T_estimation_joint_models.py).
M.SUPPORTED_FIXTURE_RULE = os.environ.get("T_FIXTURE_RULE", "any")

# Week-block wild bootstrap (score-based, no refitting): each ISO week (Monday-
# Sunday, all LTLAs together) is one block. Draws flip the sign of each week's
# score contribution at random (Rademacher weights).
BOOTSTRAP_DRAWS = int(os.environ.get("T_BOOTSTRAP_DRAWS", 9999))

# Weights used by the permutation tests and the bootstrap (not by the fit itself):
#   "poisson"  variance proportional to mu (default; weight of an LTLA-day grows with mu)
#   "power"    variance proportional to mu^p, which down-weights the largest LTLAs
#              when p > 1. p is estimated from the joint fit's residuals ("auto":
#              slope of log mean squared residual on log mu across bins of mu) or
#              set by T_VARIANCE_POWER. The football coefficients are then updated
#              by one scoring step with these weights, starting from the Poisson fit.
VARIANCE_WEIGHTING = os.environ.get("T_VARIANCE_WEIGHTING", "poisson")
VARIANCE_POWER = os.environ.get("T_VARIANCE_POWER", "auto")
VARIANCE_BINS = 20
BOOTSTRAP_SEED = 23

MAX_OUTER_ITER = int(os.environ.get("T_MAX_OUTER_ITER", M.MAX_BLOCK_ITER_C))
OUTER_REL_TOL = M.BLOCK_REL_TOL
COEF_CHANGE_TOL = 1e-5
MIN_MULTIPLIER = 1e-8


# ------------------------------------------------------------------ inputs --
def kickoff_slot(kickoff):
    if pd.isna(kickoff) or str(kickoff) in ("", "None", "nan"):
        return None
    hour = int(str(kickoff).split(":")[0])
    return "lunch" if hour < 14 else ("afternoon" if hour < 17 else "evening")


def load_tv_lookup():
    """
    (date, standardised club) -> True / False / None for a 2021/22 fixture,
    plus the fixture id and kickoff slot for the same key.
    """
    if not TV_LISTINGS_PATH.exists():
        raise FileNotFoundError(f"{TV_LISTINGS_PATH} missing; load the proper TV-status table at TV_LISTINGS_PATH.")
    tv = pd.read_csv(TV_LISTINGS_PATH)
    if M.RESTRICT_TO_SUPPORTED_FIXTURES:
        tv = tv[M.supported_fixture_mask(tv.home, tv.away, M.supported_teams())]
    status = tv["televised"].astype(str).map({"True": True, "False": False})  # unknown -> NaN
    lookup, fixture_id, slot = {}, {}, {}
    for i, (d, h, a, s, k) in enumerate(zip(tv["date"], tv["home"], tv["away"], status, tv["kickoff"])):
        date = pd.Timestamp(d)
        value = None if pd.isna(s) else bool(s)
        for club in (M.standardise_team_name(h), M.standardise_team_name(a)):
            if (date, club) in lookup:
                raise ValueError(f"Ambiguous TV fixture for {club} on {date}")
            lookup[(date, club)] = value
            fixture_id[(date, club)] = i
            slot[(date, club)] = kickoff_slot(k)
    return lookup, fixture_id, slot


def resolve_tv(value):
    if TV_UNKNOWN_AS not in {"exclude", "not_televised", "televised"}:
        raise ValueError("T_TV_UNKNOWN_AS must be exclude, not_televised, or televised")
    if value is None:
        if TV_UNKNOWN_AS == "exclude":
            return None
        return TV_UNKNOWN_AS == "televised"
    return value


def build_exposures(df, unique_dates, data, s_tv, s_home, w_lookup, seasons_by_team, events, tv_lookup, fixture_id,
                    slot_lookup=None):
    """
    Returns
        T, H           row-level exposure arrays (observed TV labels and home sides)
        fixtures       dict fixture -> {
                           "date", "tv",
                           "t_rows", "t_vals"   T contribution of each club, used
                                                if the fixture is televised
                           "h_parts"            one entry per club: {"home", "rows",
                                                "vals"} = the H contribution that club
                                                WOULD make if it were the home side
                                                (vals None if it has no exposure)
                           "has_home"           False for neutral venues
                       }
        diagnostics    counts
    The per-club potential contributions are what the permutation test reshuffles.
    """
    n = len(df)
    nd = len(unique_dates)
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}
    order = np.argsort(data.date_idx, kind="stable")
    rows_by_date = np.split(order, np.cumsum(np.bincount(data.date_idx, minlength=nd))[:-1])
    lad_idx = data.lad_idx
    first, last = pd.Timestamp(unique_dates.min()), pd.Timestamp(unique_dates.max())

    def window_rows(start, end):
        start, end = max(pd.Timestamp(start), first), min(pd.Timestamp(end), last)
        if start > end:
            return np.array([], dtype=int)
        idx = [date_pos[d] for d in pd.date_range(start, end, freq="D") if d in date_pos]
        return np.concatenate([rows_by_date[i] for i in idx]) if idx else np.array([], dtype=int)

    def home_vector(club, season_key):
        if HOME_EXPOSURE == "S":
            return s_home.get(club)
        w_vec, _, _ = M.resolve_W_vector(club, int(season_key), w_lookup, seasons_by_team)
        if w_vec is None:
            return None
        if HOME_EXPOSURE == "W":
            return w_vec
        return None if club not in s_home else s_home[club] * w_vec

    fixtures = {}
    diag = dict(team_events=0, outside_window=0, no_S=0, not_in_tv_listing=0, tv_unknown=0,
                televised_team_events=0, home_events_used=0, home_events_missing_exposure=0,
                away_not_televised_dropped=0, unknown_excluded=0)

    for k, e in events.reset_index(drop=True).iterrows():
        diag["team_events"] += 1
        club = M.standardise_team_name(e["team"])
        date = pd.Timestamp(e["match_date"])
        if max(pd.Timestamp(e["t_m_end"]), pd.Timestamp(e["t_h_end"])) < first or min(pd.Timestamp(e["t_m_start"]), pd.Timestamp(e["t_h_start"])) > last:
            diag["outside_window"] += 1
            continue

        key = (date, club)
        if key in tv_lookup:
            raw_tv = tv_lookup[key]
            fid = fixture_id[key]
        else:
            raw_tv, fid = None, f"unlisted_{k}"
            diag["not_in_tv_listing"] += 1
        if raw_tv is None:
            diag["tv_unknown"] += 1
        televised = resolve_tv(raw_tv)
        if televised is None:
            diag["unknown_excluded"] += 1
            continue

        is_home = bool(e["is_home"])
        fx = fixtures.setdefault(fid, {"date": date, "tv": televised, "t_rows": [], "t_vals": [], "t_clubs": [],
                                       "h_parts": [], "has_home": False,
                                       "slot": slot_lookup.get(key) if slot_lookup else None})
        fx["has_home"] |= is_home

        # televised term: potential contribution of this club to its fixture
        if club in s_tv:
            rows = window_rows(e["t_m_start"], e["t_m_end"])
            fx["t_rows"].append(rows)
            fx["t_vals"].append(s_tv[club][lad_idx[rows]])
            fx["t_clubs"].append(club)
            if televised:
                diag["televised_team_events"] += 1
        else:
            diag["no_S"] += 1

        # home term: potential contribution of this club if it is the home side
        vec = home_vector(club, e["season_key"])
        rows = window_rows(e["t_h_start"], e["t_h_end"])
        fx["h_parts"].append({"home": is_home, "rows": rows, "club": club, "season": int(e["season_key"]),
                              "vec": vec, "vals": None if vec is None else vec[lad_idx[rows]]})
        if is_home:
            diag["home_events_used" if vec is not None else "home_events_missing_exposure"] += 1
        elif not televised:
            diag["away_not_televised_dropped"] += 1

    T = assemble_T(n, fixtures, observed_tv_flags(fixtures))
    H = assemble_H(n, fixtures, observed_home_labels(fixtures))
    return T, H, fixtures, diag


def group_key(date, slot, fid, within):
    """Key of the permutation group a fixture (or team-event) falls in."""
    date = pd.Timestamp(date)
    if within == "date":
        return date
    if within == "date_and_slot":
        return (date, slot) if slot is not None else ("unshuffled", fid)
    if within == "week":
        iso = date.isocalendar()
        return (iso[0], iso[1])
    if within == "dow_month":
        return (date.year, date.month, date.dayofweek)
    raise ValueError(f"unknown permutation grouping {within!r}; use one of {PERMUTATION_GROUPINGS}")


def tv_permutation_groups(fixtures, within=None):
    """Fixture groups within which TV labels are exchangeable."""
    within = TV_PERMUTATION_WITHIN if within is None else within
    groups = {}
    for fid, fx in fixtures.items():
        groups.setdefault(group_key(fx["date"], fx["slot"], fid, within), []).append(fid)
    return groups


def home_units(fixtures):
    """Team-events that can carry the home label: non-neutral venue, club has W."""
    return [(fid, i) for fid, fx in fixtures.items() if fx["has_home"]
            for i, p in enumerate(fx["h_parts"]) if p["vals"] is not None]


def home_permutation_groups(fixtures, within=None):
    """Team-event groups within which home labels are exchangeable."""
    within = HOME_PERMUTATION_WITHIN if within is None else within
    groups = {}
    for fid, i in home_units(fixtures):
        fx = fixtures[fid]
        groups.setdefault(group_key(fx["date"], fx["slot"], (fid, i), within), []).append((fid, i))
    return groups


def observed_tv_flags(fixtures):
    return {fid: fx["tv"] for fid, fx in fixtures.items()}


def observed_home_labels(fixtures):
    """(fixture, club index) -> True if that club played at home."""
    return {(fid, i): bool(fixtures[fid]["h_parts"][i]["home"]) for fid, i in home_units(fixtures)}


def assemble_T(n, fixtures, tv_flags):
    T = np.zeros(n)
    for fid, fx in fixtures.items():
        if tv_flags[fid]:
            for rows, vals in zip(fx["t_rows"], fx["t_vals"]):
                np.add.at(T, rows, vals)
    return T


def assemble_H(n, fixtures, home_labels):
    H = np.zeros(n)
    for (fid, i), is_home in home_labels.items():
        if is_home:
            part = fixtures[fid]["h_parts"][i]
            np.add.at(H, part["rows"], part["vals"])
    return H


def combined_assignment(fixtures, tv_groups, home_groups, rng, tv_obs=None, home_obs=None):
    """One draw ALWAYS applies both shuffles: TV labels among fixtures and home
    labels among team-events, each within its groups. Label counts per group,
    dates and fixture participants are fixed."""
    tv_obs = observed_tv_flags(fixtures) if tv_obs is None else tv_obs
    home_obs = observed_home_labels(fixtures) if home_obs is None else home_obs
    flags, labels = dict(tv_obs), dict(home_obs)
    for fids in tv_groups.values():
        flags.update(zip(fids, rng.permutation([tv_obs[f] for f in fids])))
    for units in home_groups.values():
        shuffled = rng.permutation([home_obs[u] for u in units])
        labels.update(zip(units, (bool(v) for v in shuffled)))
    return flags, labels


def permutation_mixing(fixtures, tv_groups, home_groups):
    """How much a draw can change: groups with both labels, label share that moves,
    and the expected correlation between permuted and observed label vectors."""
    def summary(groups, label_of):
        n_mixed, moved, within_var, labels = 0, 0.0, 0.0, []
        for units in groups.values():
            x = np.array([label_of(u) for u in units], dtype=float)
            labels.extend(x)
            k, m = x.sum(), len(x)
            if 0 < k < m:
                n_mixed += 1
            moved += k * (m - k) / m
            within_var += k * (m - k) / m
        labels = np.array(labels)
        total_var = labels.sum() * (len(labels) - labels.sum()) / max(len(labels), 1)
        return dict(groups=len(groups), groups_mixed=n_mixed,
                    share_true_moved=moved / max(labels.sum(), 1),
                    label_overlap=1 - within_var / total_var if total_var > 0 else 1.0)
    return (summary(tv_groups, lambda f: fixtures[f]["tv"]),
            summary(home_groups, lambda u: fixtures[u[0]]["h_parts"][u[1]]["home"]))


def scale_of(v):
    nz = np.abs(v[np.abs(v) > 0])
    return 1.0 if len(nz) == 0 else float(np.median(nz))


def week_block_wild_bootstrap(x, r, block, estimate, n_draws, seed, alpha=0.05):
    """
    Wild bootstrap of the two football coefficients with weeks as blocks.

    x (n x 2) and r (n) are the design columns and Pearson residuals of the joint
    fit, both with the baseline (spline, weekday, date) directions projected out,
    so x' r is the efficient score. With U_w the score of week w and
    I = x' x, each draw is
        theta* = estimate + I^-1 sum_w e_w U_w,    e_w = +-1 with probability 1/2.
    Correlation of any kind WITHIN a week (across LTLAs and days) and
    overdispersion are carried by U_w; weeks are assumed independent.
    Returns percentile intervals, two-sided p-values for zero, the week-clustered
    robust SE and the draws.
    """
    x, r = np.asarray(x, float), np.asarray(r, float)
    _, block = np.unique(block, return_inverse=True)
    n_blocks = block.max() + 1
    U = np.column_stack([np.bincount(block, weights=x[:, k]*r, minlength=n_blocks) for k in range(x.shape[1])])
    info_inv = np.linalg.inv(x.T @ x)
    rng = np.random.default_rng(seed)
    e = rng.choice([-1.0, 1.0], size=(n_draws, n_blocks))
    draws = np.asarray(estimate, float) + (e @ U) @ info_inv.T
    lo, hi = np.quantile(draws, [alpha/2, 1-alpha/2], axis=0)
    dev = np.abs(draws - np.asarray(estimate))
    p_zero = (1 + (dev >= np.abs(np.asarray(estimate))).sum(axis=0)) / (n_draws + 1)
    meat = U.T @ U * n_blocks / max(n_blocks - 1, 1)
    se_cluster = np.sqrt(np.diag(info_inv @ meat @ info_inv))
    return dict(ci=np.column_stack([lo, hi]), p_zero=p_zero, se_cluster=se_cluster,
                n_blocks=int(n_blocks), draws=draws)


def estimate_variance_power(y, mu, n_bins=VARIANCE_BINS):
    """Slope of log E[(y-mu)^2] on log mu over quantile bins of mu, clipped to [1, 2].
    Returns (p, table of bins)."""
    frame = pd.DataFrame({"mu": mu, "r2": (y - mu)**2})
    frame["bin"] = pd.qcut(frame.mu, n_bins, duplicates="drop")
    table = frame.groupby("bin", observed=True).agg(mu=("mu", "mean"), var=("r2", "mean"), n=("mu", "size"))
    table = table[(table.mu > 0) & (table["var"] > 0)]
    slope = np.polyfit(np.log(table.mu), np.log(table["var"]), 1, w=np.sqrt(table.n))[0]
    table["dispersion"] = table["var"] / table.mu
    return float(np.clip(slope, 1.0, 2.0)), table


# ----------------------------------------------------------------- fitting --
def exposure_vertices(T_s, H_s):
    """A linear multiplier reaches its minimum at a convex-hull vertex."""
    points = np.unique(np.column_stack([T_s, H_s]), axis=0)
    if len(points) <= 2:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        # Collinear exposure designs: retain endpoints along their main direction.
        _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        z = points @ vt[0]
        return points[[z.argmin(), z.argmax()]]


def nll_given_tau(data, tau, a, b, T_s, H_s):
    F = 1.0 + a * T_s + b * H_s
    mu = data.bg + tau * F
    if not np.isfinite(mu).all() or (mu <= 0).any():
        return np.inf, mu, F
    mu = np.maximum(mu, M.EPS)
    return float(np.sum(mu - data.y * np.log(mu))), mu, F


def fit_football_given_tau(data, tau, x0, T_s, H_s):
    """Signed coefficients with a JOINT nonnegative-component constraint."""
    vertices = exposure_vertices(T_s, H_s)
    if not np.any(vertices):
        return np.zeros(2)
    normalizer = max(float(data.y.sum()), len(data.y), 1.)
    def obj(p):
        val, mu, F = nll_given_tau(data, tau, *p, T_s, H_s)
        if not np.isfinite(val):
            return np.inf, np.zeros(2)
        g = (1.0 - data.y / mu) * tau * ((data.bg + tau*F) > M.EPS)
        return val / normalizer, np.array([g @ T_s, g @ H_s]) / normalizer
    constraints = {"type": "ineq", "fun": lambda p: 1 + vertices @ p - MIN_MULTIPLIER,
                   "jac": lambda p: vertices}
    bounds = [(None, None) if np.any(x) else (0., 0.) for x in [T_s, H_s]]
    best = None
    messages = []
    for initial in [x0, (0., 0.), (.05, .05)]:
        start = np.array(initial, dtype=float)
        if np.min(1 + vertices @ start) < MIN_MULTIPLIER:
            start[:] = 0
        res = minimize(obj, start, jac=True, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": M.MAXITER_GLOBAL, "ftol": 1e-12})
        messages.append(str(res.message))
        if res.success and np.isfinite(res.fun) and np.min(1 + vertices @ res.x) >= MIN_MULTIPLIER - 1e-10:
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        raise RuntimeError("Football optimizer failed: " + "; ".join(messages))
    return best.x


def nuisance_objective(params, data, a, b):
    """All baseline parameters jointly; avoids slow LTLA/date block cycling."""
    nb = data.n_lad*data.Ks
    nw = data.n_lad*6
    beta = params[:nb].reshape(data.n_lad,data.Ks)
    wfree = params[nb:nb+nw].reshape(data.n_lad,6)
    # A cubic B-spline has at most four nonzero entries per observation.
    # Cache the block-diagonal spline design instead of repeatedly materialising
    # every observation x every spline coefficient in the optimizer hot path.
    if not hasattr(data, '_nuisance_spline_design'):
        rows,columns=np.nonzero(data.B_s)
        data._nuisance_spline_design=csr_matrix(
            (data.B_s[rows,columns],(rows,data.lad_idx[rows]*data.Ks+columns)),
            shape=(data.n,nb))
    spline_design=data._nuisance_spline_design
    eta = spline_design @ beta.ravel()
    eta += M.constrained_weekday_effects(wfree)[data.lad_idx,data.dow]
    if data.Q_delta is not None:
        eta += (data.Q_delta@params[nb+nw:])[data.date_idx]
    tau = np.exp(np.clip(eta,*M.ETA_BOUNDS))
    F = 1+a*data.A_scaled+b*data.H_scaled
    raw_mu = data.bg+tau*F
    mu = np.maximum(raw_mu,M.EPS)
    # Half-deviance differs from the NLL by a data-only constant and avoids
    # convergence criteria being dominated by a large irrelevant constant.
    value = float(np.sum(mu-data.y+xlogy(data.y,data.y/mu)))
    penalty,grad_penalty = M.roughness_2d(beta,M.S_SMOOTH_PENALTY)
    g = (1-data.y/mu)*tau*F*((eta>M.ETA_BOUNDS[0])&(eta<M.ETA_BOUNDS[1]))*(raw_mu>M.EPS)
    gb = (spline_design.T @ g).reshape(data.n_lad,data.Ks)+grad_penalty
    gw7 = np.bincount(7*data.lad_idx+data.dow,weights=g,minlength=7*data.n_lad).reshape(data.n_lad,7)
    grad = [gb.ravel(),(gw7[:,:6]-gw7[:,6,None]).ravel()]
    if data.Q_delta is not None:
        grad.append(data.Q_delta.T@np.bincount(data.date_idx,weights=g,minlength=len(data.unique_dates)))
    return value+penalty,np.concatenate(grad)



def fit_nuisance_joint(data,beta,wfree,a,b,multistart=False):
    pieces = [beta.ravel(),wfree.ravel()]
    if data.Q_delta is not None:
        pieces.append(np.zeros(data.q_delta) if data.gamma is None else data.gamma)
    initial = np.concatenate(pieces)
    starts = [initial]
    if multistart:
        for shift in [-2.,2.]:
            alternative = initial.copy()
            alternative[:data.n_lad*data.Ks] += shift
            starts.append(alternative)
    solutions = []
    for number,initial in enumerate(starts,1):
        iteration=[0]
        def progress(_):
            iteration[0]+=1
            if iteration[0]%250==0:
                print(f'    nuisance start {number}/{len(starts)}: iteration {iteration[0]}',flush=True)
        result=minimize(nuisance_objective,initial,args=(data,a,b),jac=True,callback=progress,
                        method='L-BFGS-B',options={'maxiter':int(os.environ.get('T_MAX_NUISANCE_ITER',5000)),
                                                 'ftol':1e-12,'gtol':1e-6,'maxls':50,'maxcor':30})
        solutions.append(result)
        if multistart:
            print(f'    nuisance start {number}/{len(starts)} finished: success={result.success}, iterations={result.nit}',flush=True)
    successful = [r for r in solutions if r.success and np.isfinite(r.fun)]
    result = min(successful or solutions,key=lambda r:r.fun)
    nb,nw = data.n_lad*data.Ks,data.n_lad*6
    beta = result.x[:nb].reshape(data.n_lad,data.Ks)
    wfree = result.x[nb:nb+nw].reshape(data.n_lad,6)
    if data.Q_delta is not None:
        data.set_delta(result.x[nb+nw:])
    return beta,wfree,result



def fit_model(data, football, start=None, coef0=(0.0, 0.0), label=""):
    if start is None:
        beta, wfree = M.initial_A(data)
        gamma = np.zeros(data.q_delta) if data.Q_delta is not None else None
    else:
        beta, wfree, gamma = start["beta"].copy(), start["wfree"].copy(), start["gamma"]
    data.set_delta(gamma)
    a, b = coef0
    T_s, H_s = data.A_scaled, data.H_scaled

    old, converged, failures = np.inf, False, 0
    history = []
    for it in range(1, MAX_OUTER_ITER + 1):
        old_coef = np.array([a,b])
        if football:
            tau = M.evaluate_A_arrays(data, beta, wfree, 0.0, 0.0)["tau"]
            a, b = fit_football_given_tau(data, tau, (a, b), T_s, H_s)
        beta,wfree,nuisance_result = fit_nuisance_joint(data,beta,wfree,a,b,multistart=(it==1 and not football))
        failures = int(not nuisance_result.success)
        delta_ok = bool(nuisance_result.success)
        if football:
            tau = M.evaluate_A_arrays(data, beta, wfree, 0.0, 0.0)["tau"]
            a, b = fit_football_given_tau(data, tau, (a, b), T_s, H_s)
        obj = M.penalized_objective_A(data, beta, wfree, a, b)
        rel = M.relative_change(old, obj)
        coef_change = float(np.max(np.abs(np.array([a,b])-old_coef)))
        history.append(dict(iteration=it, objective=obj, relative_change=rel,
                            coefficient_change=coef_change, nuisance_failures=failures, nuisance_success=delta_ok,
                            nuisance_message=str(nuisance_result.message),nuisance_iterations=int(nuisance_result.nit)))
        if it <= 3 or it % 10 == 0 or rel < OUTER_REL_TOL:
            extra = f"  a={a:.5g} b={b:.5g}" if football else ""
            print(f"  [{label}] iter {it:03d} objective={obj:.6f} rel={rel:.2e}{extra}", flush=True)
        if failures or not np.isfinite(obj):
            print(f"  [{label}] baseline optimizer failed: {nuisance_result.message}",flush=True)
            break
        if (not football or (rel < OUTER_REL_TOL and coef_change < COEF_CHANGE_TOL)) and not failures and delta_ok:
            converged = True
            break
        old = obj

    ev = M.evaluate_A_arrays(data, beta, wfree, a, b)
    result = dict(beta=beta, wfree=wfree, gamma=None if data.gamma is None else data.gamma.copy(),
                  a=a, b=b, ev=ev, converged=converged, iterations=it, lad_failures=failures,
                  date_success=delta_ok, history=history)
    return result


def fit_signature(data):
    h = hashlib.sha256()
    for array in [data.y,data.bg,data.lad_idx,data.date_idx,data.B_s_by_date,
                  data.dow,data.unique_dates.asi8,data.A_scaled,data.H_scaled]:
        h.update(np.ascontiguousarray(array).tobytes())
    if data.Q_delta is not None:
        h.update(np.ascontiguousarray(data.Q_delta).tobytes())
    h.update(str((data.q_delta,M.S_SMOOTH_PENALTY,MIN_MULTIPLIER,M.ETA_BOUNDS,
                  M.TV_SUPPORTER_SCALE,M.TV_AUDIENCE_TOTAL,TV_USE_SUPPORTER_WEIGHTS)).encode())
    return h.hexdigest()


def load_baseline_reference(data, reference):
    """Reuse a zero-football fit only after verifying its full saved signature.

    Football columns are irrelevant when both coefficients are zero. Substitute
    the reference columns for signature validation, retaining CURRENT outcomes,
    dates, LAD indices, nuisance basis and model constants. Restore current
    football columns before returning; never reuse the joint football fit.
    """
    design=pd.read_csv(reference/'T_home_tv_design_terms.csv',float_precision='round_trip')
    history=pd.read_csv(reference/'T_home_tv_baseline_convergence.csv')
    if history.empty or not bool(history.iloc[-1].nuisance_success) or history.iloc[-1].nuisance_failures:
        raise ValueError('Reference background fit did not converge')
    if len(design)!=data.n:
        raise ValueError('Reference background panel length mismatch')
    with np.load(reference/'T_home_tv_baseline_checkpoint.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['coef'],[0.,0.]):
            raise ValueError('Reference is not a zero-football baseline')
        old_T,old_H=data.A_scaled,data.H_scaled
        try:
            T=design.T_televised.to_numpy(); H=design.H_home.to_numpy()
            data.A_scaled=T/scale_of(T); data.H_scaled=H/scale_of(H)
            if saved['signature'].item()!=fit_signature(data):
                raise ValueError('Reference baseline outcome/design/settings mismatch')
        finally:
            data.A_scaled,data.H_scaled=old_T,old_H
        beta=saved['beta'].copy(); wfree=saved['wfree'].copy()
        gamma=saved['gamma'].copy() if data.Q_delta is not None else None
    data.set_delta(gamma)
    return dict(beta=beta,wfree=wfree,gamma=gamma,a=0.,b=0.,
                ev=M.evaluate_A_arrays(data,beta,wfree,0.,0.),converged=True,
                iterations=int(history.iloc[-1].iteration),lad_failures=0,date_success=True,
                history=history.to_dict('records'))


def checked_fit(data, football, label, start=None, resume=False):
    checkpoint = M.RESULTS_DIR / f"T_home_tv_{label}_checkpoint.npz"
    coef = (0.,0.)
    reference = os.environ.get('T_BASELINE_REFERENCE_DIR')
    if not football and not resume and reference:
        result = load_baseline_reference(data, Path(reference))
        np.savez_compressed(checkpoint, beta=result['beta'],wfree=result['wfree'],
                            gamma=np.array([]) if result['gamma'] is None else result['gamma'],
                            coef=[0.,0.],signature=fit_signature(data))
        pd.DataFrame(result['history']).to_csv(M.RESULTS_DIR / f'T_home_tv_{label}_convergence.csv',index=False)
        (M.RESULTS_DIR/'baseline_reference.json').write_text(json.dumps(dict(
            reference=str(reference),validation='Reference exposure substituted only for full checkpoint signature check; current outcome and nuisance design must match.'),indent=2))
        print(f'  Reused compatible converged background-only fit from {reference}',flush=True)
        return result
    if resume and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if saved['signature'].item() != fit_signature(data):
                raise ValueError("Checkpoint input/design/settings mismatch; use a new results directory.")
            start = dict(beta=saved['beta'],wfree=saved['wfree'],
                         gamma=saved['gamma'] if data.Q_delta is not None else None)
            coef = tuple(saved['coef'])
            if 'converged' in saved and bool(saved['converged'].item()):
                data.set_delta(start['gamma'])
                ev = M.evaluate_A_arrays(data, start['beta'], start['wfree'], *coef)
                if not np.isfinite(ev['mu']).all() or (ev['mu'] <= 0).any():
                    raise ValueError('Invalid converged checkpoint predictions')
                history = pd.read_csv(M.RESULTS_DIR / f'T_home_tv_{label}_convergence.csv')
                print(f'  Reusing converged {label} checkpoint', flush=True)
                return dict(**start, a=coef[0], b=coef[1], ev=ev, converged=True,
                            iterations=int(history.iloc[-1].iteration), lad_failures=0, date_success=True,
                            history=history.to_dict('records'))
    result = fit_model(data, football=football, start=start, coef0=coef, label=label)
    np.savez_compressed(checkpoint, beta=result['beta'],wfree=result['wfree'],
                        gamma=np.array([]) if result['gamma'] is None else result['gamma'],
                        coef=[result['a'],result['b']],signature=fit_signature(data),converged=result['converged'])
    pd.DataFrame(result['history']).to_csv(M.RESULTS_DIR / f"T_home_tv_{label}_convergence.csv",index=False)
    if not result['converged']:
        raise RuntimeError(f"{label} did not converge. Inference stopped. Checkpoint saved; increase "
                           "T_MAX_NUISANCE_ITER for a failed inner optimizer or T_MAX_OUTER_ITER for outer cycling, "
                           "and rerun with --resume using the same results directory.")
    return result



# --------------------------------------------------------------- inference --
def pearson_dispersion(y, mu, n_params):
    return float(np.sum((y - mu) ** 2 / np.maximum(mu, M.EPS)) / max(len(y) - n_params, 1))


# -------------------------------------------------------------------- main --
def export_bootstrap_fit(df, joint, tau0, T, H, T_scale, H_scale, boot_row, diag, ll0, ll1, phi, weights, n_teams):
    """Keep the main outputs available when only bootstrap CIs are requested."""
    row = dict(input=str(M.INFECTION_PATH), s_file=str(M.S_PATH), w_file=str(M.W_PATH),
               date_effect=USE_DATE_EFFECT, home_exposure=HOME_EXPOSURE,
               tv_unknown_as=TV_UNKNOWN_AS, fixture_rule=M.SUPPORTED_FIXTURE_RULE,
               tv_supporter_weights_applied=weights is not None and TV_USE_SUPPORTER_WEIGHTS,
               tv_supporter_scale=M.TV_SUPPORTER_SCALE,
               tv_audience_total=M.TV_AUDIENCE_TOTAL if TV_USE_SUPPORTER_WEIGHTS and M.TV_SUPPORTER_SCALE=='assumed_audience' else None,
               n_supported_teams=n_teams, C_m_tv=joint['a']/T_scale, D_s_home=joint['b']/H_scale,
               baseline_converged=True, joint_converged=True, ci_method='week_block_wild_bootstrap',
               ci_simultaneous=False, n_permutations=0,
               tv_permutation_within=TV_PERMUTATION_WITHIN, home_permutation_within=HOME_PERMUTATION_WITHIN,
               pearson_phi_diagnostic=phi, loglik_baseline=ll0, loglik_joint=ll1,
               **{f'n_{k}':v for k,v in diag.items()})
    row.update(boot_row)
    for prefix in ['C_m_tv', 'D_s_home']:
        for bound in ['low', 'high']:
            row[f'{prefix}_ci_{bound}_pct'] = boot_row[f'{prefix}_boot_ci_{bound}_pct']
    pd.DataFrame([row]).to_csv(M.RESULTS_DIR/'T_home_tv_summary.csv',index=False)
    daily = df[[M.LOCATION_COL,M.DATE_COL,M.INFECTIONS_COL,M.BACKGROUND_COL]].copy()
    daily['tau_baseline_no_football'], daily['tau_joint'] = tau0, joint['ev']['tau']
    daily['T_televised'], daily['H_home'] = T, H
    daily['football_multiplier'], daily['mu_joint'] = joint['ev']['football_multiplier'], joint['ev']['mu']
    daily.to_csv(M.RESULTS_DIR/'T_home_tv_daily.csv',index=False)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true", help="build design terms and exit")
    parser.add_argument("--resume", action="store_true", help="continue compatible saved fitting checkpoints")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="fit (use with --resume to reuse saved fits), run the week-block bootstrap, skip permutations")
    args = parser.parse_args()
    if VARIANCE_WEIGHTING != "poisson":
        raise ValueError("This reconciled pipeline uses Poisson fits; the experimental power-weighted one-step output is not supported.")
    if not args.build_only:
        # A failed rerun must not leave a previous summary looking current.
        previous = M.RESULTS_DIR / "previous_outputs" / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        for name in ['T_home_tv_summary.csv','T_home_tv_daily.csv','T_home_tv_inference.json',
                     'T_home_tv_permutation_confidence_grid.csv','T_home_tv_permutation_scores.npz',
                     'T_home_tv_permutation_null.csv']:
            path = M.RESULTS_DIR/name
            if path.exists():
                previous.mkdir(parents=True,exist_ok=True)
                path.rename(previous/name)

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        df, lads, lad_to_idx, unique_dates = M.load_infections()
        B_s, _, _ = M.bspline_basis(unique_dates, spacing_days=M.S_KNOT_SPACING_DAYS, degree=M.SPLINE_DEGREE)
        _, s_raw = M.load_S(lads, lad_to_idx)
        weights = M.load_supporter_weights(sorted(s_raw))
        s_weighted = M.apply_supporter_weights(dict(s_raw), weights)
        _, w_lookup, seasons_by_team = M.load_W(lads, lad_to_idx)
        events = M.load_events()
    print(quiet.getvalue(), end="")
    if TV_USE_SUPPORTER_WEIGHTS and weights is None:
        raise ValueError("Requested national fanbase weighting could not be applied.")
    s_tv = s_weighted if TV_USE_SUPPORTER_WEIGHTS else s_raw
    s_home = s_weighted if HOME_USE_SUPPORTER_WEIGHTS else s_raw
    tv_lookup, fixture_id, slot_lookup = load_tv_lookup()
    audit = pd.read_csv(TV_LISTINGS_PATH)
    audit_dates = pd.to_datetime(audit['date'])
    audit['supported_pair'] = M.supported_fixture_mask(audit.home, audit.away, set(s_raw))
    audit['window_overlap'] = ((audit_dates+pd.to_timedelta(max(M.MATCH_WINDOW_END_DAYS,M.HOME_WINDOW_END_DAYS),unit='D') >= unique_dates.min()) &
                               (audit_dates+pd.to_timedelta(min(M.MATCH_WINDOW_START_DAYS,M.HOME_WINDOW_START_DAYS),unit='D') <= unique_dates.max()))
    audit['tv_known'] = audit.televised.astype(str).isin(['True','False'])
    audit['included'] = audit.supported_pair & audit.window_overlap & (audit.tv_known | (TV_UNKNOWN_AS != 'exclude'))
    audit.to_csv(M.RESULTS_DIR/'T_home_tv_fixture_audit.csv',index=False)
    inputs = [M.INFECTION_PATH,M.S_PATH,M.W_PATH,TV_LISTINGS_PATH,M.SUPPORTER_PATH,*M.EVENT_PATHS]
    configuration = dict(club_map_mode=os.environ.get("T_CLUB_MAP_MODE", "geography_only"), input_label=os.environ.get('T_INPUT_LABEL',M.INFECTION_PATH.name),
        source_input=os.environ.get('T_SOURCE_INPUT',str(M.INFECTION_PATH)),supported_teams=sorted(s_raw),fixture_rule=M.SUPPORTED_FIXTURE_RULE,
        tv_unknown_policy=TV_UNKNOWN_AS,supporter_weights=weights,
        home_exposure=HOME_EXPOSURE,date_effect=USE_DATE_EFFECT,permutations=N_PERMUTATIONS,
        supporter_weight_column=M.SUPPORTER_WEIGHT_COLUMN,
        **M.supporter_scaling_metadata(weights, TV_USE_SUPPORTER_WEIGHTS),
        tv_use_supporter_weights=TV_USE_SUPPORTER_WEIGHTS,
        baseline_method="supervisor_20260921", bootstrap_draws=BOOTSTRAP_DRAWS,
        geography="April 2020 LAD; mid-2020 population",
        permutation_grouping=TV_PERMUTATION_WITHIN,
        home_permutation_grouping=HOME_PERMUTATION_WITHIN,permutation_seed=PERMUTATION_SEED,
        date_min=str(unique_dates.min().date()),date_max=str(unique_dates.max().date()),
        min_multiplier=MIN_MULTIPLIER,eta_numerical_bounds=M.ETA_BOUNDS,
        inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs})
    (M.RESULTS_DIR/'T_home_tv_configuration.json').write_text(json.dumps(configuration,indent=2))
    if weights is not None and TV_USE_SUPPORTER_WEIGHTS:
        write_audit(M.RESULTS_DIR, M.S_PATH, weights, M.supporter_scaling_metadata(weights))

    print(f"input: {M.INFECTION_PATH}   window {unique_dates.min().date()}..{unique_dates.max().date()}"
          f"   rows {len(df):,}   results -> {M.RESULTS_DIR}/")
    print(f"settings: TV_UNKNOWN_AS={TV_UNKNOWN_AS}  HOME_EXPOSURE={HOME_EXPOSURE}  "
          f"TV weights={TV_USE_SUPPORTER_WEIGHTS}  home weights={HOME_USE_SUPPORTER_WEIGHTS}  "
          f"date effect={USE_DATE_EFFECT}  fixture rule={M.SUPPORTED_FIXTURE_RULE}  "
          f"permutation within: TV={TV_PERMUTATION_WITHIN} home={HOME_PERMUTATION_WITHIN}  "
          f"inference weights={VARIANCE_WEIGHTING}  "
          f"draws={N_PERMUTATIONS}")

    Q = M.delta_basis(B_s, unique_dates) if USE_DATE_EFFECT else None
    shell = M.ModelData(df, lads, unique_dates, B_s)
    T, H, fixtures, diag = build_exposures(df, unique_dates, shell, s_tv, s_home, w_lookup,
                                           seasons_by_team, events, tv_lookup, fixture_id, slot_lookup)
    T_scale, H_scale = scale_of(T), scale_of(H)

    print("\ndesign:")
    for k, v in diag.items():
        print(f"  {k:28s} {v}")
    by_date = {}
    for fx in fixtures.values():
        if any(len(r) for r in fx["t_rows"]):
            by_date.setdefault(fx["date"], []).append(fx["tv"])
    tv_groups, home_groups = tv_permutation_groups(fixtures), home_permutation_groups(fixtures)
    tv_mix, home_mix = permutation_mixing(fixtures, tv_groups, home_groups)
    mixed, home_mixed = tv_mix["groups_mixed"], home_mix["groups_mixed"]
    print(f"  match dates with S exposure    {len(by_date)}")
    print(f"  dates with >=1 televised       {sum(1 for f in by_date.values() if any(f))}")
    print(f"  TV permutation (within {TV_PERMUTATION_WITHIN}): {mixed}/{tv_mix['groups']} groups mix TV and "
          f"non-TV; {100*tv_mix['share_true_moved']:.0f}% of TV labels move per draw on average; "
          f"label overlap {tv_mix['label_overlap']:.2f}")
    print(f"  home permutation (within {HOME_PERMUTATION_WITHIN}): {len(home_units(fixtures))} team-events, "
          f"{home_mixed}/{home_mix['groups']} groups mix home and away; "
          f"{100*home_mix['share_true_moved']:.0f}% of home labels move; label overlap {home_mix['label_overlap']:.2f}")
    print(f"  LTLA-days T>0: {(T > 0).sum():,}   H>0: {(H > 0).sum():,}   "
          f"corr(T,H) = {np.corrcoef(T, H)[0, 1]:.3f}")

    design = df[[M.LOCATION_COL, M.DATE_COL]].copy()
    design["T_televised"] = T
    design["H_home"] = H
    design.to_csv(M.RESULTS_DIR / "T_home_tv_design_terms.csv", index=False)
    if args.build_only:
        print(f"\nwrote {M.RESULTS_DIR / 'T_home_tv_design_terms.csv'} (build only)")
        return False

    if not (T > 0).any() or not (H > 0).any():
        raise ValueError("Both TV and home exposure are required for this two-term fit")
    data = M.ModelData(df, lads, unique_dates, B_s, A_scaled=T / T_scale, H_scaled=H / H_scale,
                       A_scale=T_scale, H_scale=H_scale, Q_delta=Q)
    T_s, H_s = data.A_scaled, data.H_scaled

    print("\nfitting baseline (no football)...", flush=True)
    base = checked_fit(data, football=False, label="baseline", resume=args.resume)
    tau0 = base["ev"]["tau"].copy()
    ll0 = M.poisson_loglik(data.y, base["ev"]["mu"])

    print("\nfitting joint model (baseline + C_m*T + D_s*H)...", flush=True)
    joint = checked_fit(data, football=True, start=base, label="joint", resume=args.resume)
    tau1, mu1 = joint["ev"]["tau"], joint["ev"]["mu"]
    ll1 = M.poisson_loglik(data.y, mu1)
    a, b = joint["a"], joint["b"]

    n_params = data.n_lad * (data.Ks + 6) + (data.q_delta if Q is not None else 0) + 2
    phi = pearson_dispersion(data.y, mu1, n_params)

    # No likelihood/profile CIs are substituted for permutation inference.
    # The two shuffled designs are passed together to score-test inversion.
    if not (PERMUTE_TV and PERMUTE_HOME):
        raise ValueError("This version requires BOTH TV and home/away permutations together.")
    if N_PERMUTATIONS < 19:
        raise ValueError("Use at least 19 permutations; 999 or more recommended for 95% intervals.")
    F = joint['ev']['football_multiplier']
    eta = joint['ev']['smooth_log'] + joint['ev']['weekday_log'] + joint['ev']['delta_log']
    if not np.isfinite(mu1).all() or (mu1 <= 0).any() or (F < MIN_MULTIPLIER-1e-10).any():
        raise ValueError("Invalid fitted means/multipliers: inference stopped.")
    clipped = (eta <= M.ETA_BOUNDS[0]) | (eta >= M.ETA_BOUNDS[1])
    if VARIANCE_WEIGHTING == "poisson":
        variance_power = 1.0
    elif VARIANCE_WEIGHTING == "power":
        if VARIANCE_POWER == "auto":
            variance_power, vtable = estimate_variance_power(data.y, mu1)
            print(f"\nvariance power estimated from {len(vtable)} bins of fitted mean:")
            print(vtable[["mu", "dispersion"]].round(2).to_string())
        else:
            variance_power = float(VARIANCE_POWER)
    else:
        raise ValueError("T_VARIANCE_WEIGHTING must be 'poisson' or 'power'")
    print(f"\ninference weights: variance proportional to mu^{variance_power:.3f}", flush=True)
    # root of the working variance (up to a constant, which cancels)
    root_mu = mu1**(variance_power/2)
    # Use the derivative of the implemented clipped baseline; do not invent
    # nuisance sensitivity on its flat portions. Record their prevalence.
    projector = NuisanceProjector(data, tau1*F/root_mu*(~clipped))
    observed_x = projector.residual(np.column_stack([T_s,H_s])*(tau1/root_mu)[:,None])
    residual = projector.residual((data.y-mu1)/root_mu)
    poisson_estimate = np.array([a, b])
    if variance_power != 1.0:
        # one scoring step with the new weights, then re-centre the residuals
        step = np.linalg.solve(observed_x.T @ observed_x, observed_x.T @ residual)
        residual = residual - observed_x @ step
        a, b = poisson_estimate + step
        print(f"  one-step weighted estimate: a={a:.5g} b={b:.5g} (Poisson fit a={poisson_estimate[0]:.5g} "
              f"b={poisson_estimate[1]:.5g})")

    mT, mH = T[T>0].mean()/T_scale, H[H>0].mean()/H_scale
    to_pct = np.array([mT, mH])*100
    week = np.array([d.isocalendar()[0]*100 + d.isocalendar()[1] for d in pd.DatetimeIndex(unique_dates)])[data.date_idx]
    boot = week_block_wild_bootstrap(observed_x, residual, week, [a, b], BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    boot_ci_pct = boot['ci']*to_pct[:, None]
    print(f"\nWeek-block wild bootstrap ({boot['n_blocks']} weeks, {BOOTSTRAP_DRAWS} draws):")
    for i, name in enumerate(['TV', 'home']):
        print(f"  {name}: mean uplift {[a, b][i]*to_pct[i]:+.3f}%, 95% CI [{boot_ci_pct[i,0]:+.3f}%, {boot_ci_pct[i,1]:+.3f}%], "
              f"p={boot['p_zero'][i]:.4g}, week-clustered SE {boot['se_cluster'][i]*to_pct[i]:.3f}%")
    boot_row = {'input_label':os.environ.get('T_INPUT_LABEL', M.INFECTION_PATH.name),
                'source_input':os.environ.get('T_SOURCE_INPUT', str(M.INFECTION_PATH)),
                'design_signature': fit_signature(data), 'input': str(M.INFECTION_PATH), 'tv_unknown_as': TV_UNKNOWN_AS, 'fixture_rule': M.SUPPORTED_FIXTURE_RULE,
                'variance_weighting': VARIANCE_WEIGHTING, 'variance_power': variance_power,
                'C_m_tv_poisson_uplift_pct': poisson_estimate[0]*to_pct[0],
                'D_s_home_poisson_uplift_pct': poisson_estimate[1]*to_pct[1],
                'n_weeks': boot['n_blocks'], 'n_draws': BOOTSTRAP_DRAWS,
                'C_m_tv_uplift_pct': a*to_pct[0], 'D_s_home_uplift_pct': b*to_pct[1],
                'C_m_tv_boot_ci_low_pct': boot_ci_pct[0,0], 'C_m_tv_boot_ci_high_pct': boot_ci_pct[0,1],
                'D_s_home_boot_ci_low_pct': boot_ci_pct[1,0], 'D_s_home_boot_ci_high_pct': boot_ci_pct[1,1],
                'boot_p_tv': boot['p_zero'][0], 'boot_p_home': boot['p_zero'][1],
                'C_m_tv_week_cluster_se_pct': boot['se_cluster'][0]*to_pct[0],
                'D_s_home_week_cluster_se_pct': boot['se_cluster'][1]*to_pct[1]}
    pd.DataFrame([boot_row]).to_csv(M.RESULTS_DIR/'T_home_tv_bootstrap.csv', index=False)
    export_bootstrap_fit(df, joint, tau0, T, H, T_scale, H_scale, boot_row, diag,
                         ll0, ll1, phi, weights, len(s_raw))
    return True


def run_fit():
    status_path = M.RESULTS_DIR/'T_home_tv_run_status.json'
    status_path.write_text(json.dumps({'status':'running','inference_complete':False}))
    try:
        inference_complete = _main()
    except Exception as exc:
        status_path.write_text(json.dumps({'status':'failed','inference_complete':False,
                                          'error':str(exc)},indent=2))
        raise
    status_path.write_text(json.dumps({'status':'completed',
                                      'inference_complete':bool(inference_complete)},indent=2))



TERMS = {"TV": "televised match", "home": "home match"}

def collect(results_dir):
    results_dir = Path(results_dir).resolve()
    for filename in ["T_home_tv_run_status.json", "T_home_tv_frozen_status.json"]:
        state = json.loads((results_dir/filename).read_text())
        if state.get("status") != "completed" or not state.get("inference_complete"):
            raise ValueError(f"Incomplete analysis: {filename}")
    summary = pd.read_csv(results_dir/"T_home_tv_summary.csv").iloc[0]
    boot = pd.read_csv(results_dir/"T_home_tv_bootstrap.csv").iloc[0]
    frozen = pd.read_csv(results_dir/"T_home_tv_frozen_baseline_averaged.csv")
    if len(frozen) != 4 or frozen.duplicated(['scheme', 'term']).any():
        raise ValueError('Expected exactly four frozen permutation tests')
    if set(zip(frozen.scheme, frozen.term)) != {(s,t) for s in ('label','club') for t in ('TV','home')}:
        raise ValueError('Missing label/club test')
    signatures = {str(summary['design_signature']), str(boot['design_signature']), *frozen.design_signature.astype(str)}
    if len(signatures) != 1:
        raise ValueError('Summary, bootstrap and frozen permutations use different input/design signatures')
    if not (frozen.optimizer == 'constrained-poisson-v1').all():
        raise ValueError('Frozen results were not produced by the validated optimizer')
    rows = []
    for term, label in TERMS.items():
        key = "C_m_tv" if term == "TV" else "D_s_home"
        by_scheme = {s: frozen[(frozen.scheme == s) & (frozen.term == term)].iloc[0] for s in ("label", "club")}
        b = {c: boot[f"{key}_boot_{c}"] for c in ("ci_low_pct", "ci_high_pct")}
        b["p"] = boot[f"boot_p_{'tv' if term == 'TV' else 'home'}"]
        rows.append(dict(
            input=str(summary.get("input_label", Path(str(summary["input"])).name)), term=label,
            club_map_mode=by_scheme["club"].club_map_mode,
            uplift_pct=summary[f"{key}_uplift_pct"],
            boot_ci_low_pct=b["ci_low_pct"], boot_ci_high_pct=b["ci_high_pct"], boot_p=b["p"],
            p_label=by_scheme["label"].p_uncentred, p_club=by_scheme["club"].p_uncentred,
            frozen_label_uplift_pct=by_scheme["label"].observed_uplift_pct,
            frozen_club_uplift_pct=by_scheme["club"].observed_uplift_pct,
            p_score=summary.get("permutation_p_tv" if term == "TV" else "permutation_p_home", float("nan")),
            n_permutations=int(by_scheme["label"].n_permutations), results_dir=str(results_dir)))
    return rows


def main():
    import subprocess
    import sys
    run_fit()
    if '--build-only' not in sys.argv and '--bootstrap-only' not in sys.argv:
        subprocess.run([sys.executable, str(Path(__file__).with_name('permutation_tests.py'))], check=True)
        pd.DataFrame(collect(M.RESULTS_DIR)).to_csv(M.RESULTS_DIR / 'T_home_tv_main_results.csv', index=False)


if __name__ == '__main__':
    main()
