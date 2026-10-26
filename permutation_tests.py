"""Label/club permutation tests used by fit.py; this is not another fitted model.

Reads the same example outcome and fixture inputs configured in model_core.py
and fit.py. Original relabelling and frozen-baseline calculations are retained.
"""
import contextlib
import io
import os
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.special import xlogy
import model_core as M
import fit as V
from model_core import NuisanceProjector
def _mass_table(path, value, season=False):
    table = pd.read_csv(path)
    needed = ['team', 'lad_code', 'total_population', value] + (['season'] if season else [])
    if not set(needed).issubset(table):
        raise ValueError(f'{path}: full source maps with population are required for geography-only permutations')
    table['team'] = table.team.map(M.standardise_team_name)
    groups = ['team']
    if season:
        table['year'] = table.season.map(M.normalize_season_key)
        groups += ['year']
    if table.duplicated(groups+['lad_code']).any():
        raise ValueError(f'Duplicate full-map rows in {path}')
    vals = table[[value, 'total_population']].to_numpy(float)
    if not np.isfinite(vals).all() or (vals[:, 0] < 0).any() or (vals[:, 1] <= 0).any():
        raise ValueError(f'Invalid exposure or population in {path}')
    if (table.groupby('lad_code').total_population.nunique() != 1).any():
        raise ValueError(f'Inconsistent LAD populations in {path}')
    table['mass'] = vals[:, 0]*vals[:, 1]
    return table.groupby(groups).mass.sum().to_dict()


def source_masses(P):
    s = _mass_table(M.S_PATH, 'S_x')
    if V.TV_USE_SUPPORTER_WEIGHTS:
        if P.weights is None:
            raise ValueError('Supporter weights are required')
        s = {c: v*P.weights[c] for c,v in s.items() if c in P.s_tv}
    else:
        s = {c:s[c] for c in P.s_tv}
    if V.HOME_EXPOSURE != 'W':
        raise ValueError('Geography-only attendance preservation currently requires HOME_EXPOSURE=W')
    w = _mass_table(M.W_PATH, 'W', season=True)
    h = {}
    for fx in P.fixtures.values():
        for part in fx['h_parts']:
            if part['vec'] is not None:
                club, requested = part['club'], part['season']
                _, used, _ = M.resolve_W_vector(club, requested, P.w_lookup, P.seasons_by_team)
                h[club, requested] = w[club, used]
    return s, h


def validate_masses(masses, keys):
    if masses is None:
        raise ValueError('Geography-only permutations require complete national map masses')
    for key in keys:
        if key not in masses or not np.isfinite(masses[key]) or masses[key] <= 0:
            raise ValueError(f'Missing/nonpositive national map mass: {key}')


def ratio(mode, masses, recipient, donor):
    if mode == 'whole_map' or recipient == donor:
        return 1.
    return masses[recipient]/masses[donor]
CLUB_MAP_MODE = os.environ.get("T_CLUB_MAP_MODE", "geography_only")


CLUB_TIERS = [
    ["Manchester United", "Liverpool", "Arsenal"],
    ["Chelsea", "Manchester City", "Tottenham Hotspur", "Newcastle United"],
    ["Aston Villa", "West Ham United", "Brighton", "Crystal Palace", "Everton", "Wolverhampton Wanderers"],
]


HOME_TIERS = [
    ["Manchester United", "Arsenal", "West Ham United", "Tottenham Hotspur", "Liverpool", "Manchester City",
     "Newcastle United"],
    ["Aston Villa", "Everton", "Chelsea", "Brighton", "Wolverhampton Wanderers"],
    ["Crystal Palace", "Brentford"],
]


def iso_week(date):
    iso = pd.Timestamp(date).isocalendar()
    return (iso[0], iso[1])


class ClubRelabeller:
    """
    TV exposure with club identities relabelled within size tiers, afresh each
    ISO week: the relabelled club's fan map is written onto the rows of each
    televised fixture. With label_groups, each draw first shuffles the TV labels
    within those groups (V.tv_permutation_groups).
    """
    def __init__(self, fixtures, tv_flags, s_tv, lad_idx, n, tiers, label_groups=None, *, map_mode="whole_map", masses=None):
        if map_mode not in {"whole_map", "geography_only"}: raise ValueError(map_mode)
        self.map_mode, self.masses = map_mode, masses
        if map_mode == "geography_only": validate_masses(masses, s_tv)
        self.fixtures, self.tv_flags, self.label_groups = fixtures, dict(tv_flags), label_groups
        self.s_tv, self.lad_idx, self.n = s_tv, lad_idx, n
        missing = [c for tier in tiers for c in tier if c not in s_tv]
        if missing:
            raise ValueError(f"CLUB_TIERS clubs without S data: {missing}")
        self.tiers = [list(t) for t in tiers]
        self.parts = [(fid, club, iso_week(fx["date"]), r) for fid, fx in fixtures.items()
                      for club, r in zip(fx["t_clubs"], fx["t_rows"])]
        self.weeks = sorted({w for _, _, w, _ in self.parts})

    def exposure(self, mapping=None, flags=None):
        """mapping: {(club, week): club whose identity is used}; flags: TV labels. None = observed."""
        flags = self.tv_flags if flags is None else flags
        T = np.zeros(self.n)
        for fid, club, week, r in self.parts:
            if flags[fid]:
                source = club if mapping is None else mapping.get((club, week), club)
                np.add.at(T, r, self.s_tv[source][self.lad_idx[r]] * ratio(self.map_mode, self.masses, club, source))
        return T

    def mapping(self, rng):
        return {(club, week): source for week in self.weeks for tier in self.tiers
                for club, source in zip(tier, rng.permutation(tier))}

    def expected(self):
        """Exact mean of draw() over the relabelling (and label shuffle): each tier club's
        rows get the tier's average fan map; with label groups, each fixture is
        weighted by its group's televised fraction. The same for every labelling."""
        tier_of = {c: tier for tier in self.tiers for c in tier}
        weight = ({f: np.mean([self.tv_flags[g] for g in fids]) for fids in self.label_groups.values() for f in fids}
                  if self.label_groups is not None else {f: float(v) for f, v in self.tv_flags.items()})
        T = np.zeros(self.n)
        for fid, club, _, r in self.parts:
            if weight[fid]:
                sources = tier_of.get(club, [club])
                vals = np.mean([self.s_tv[c][self.lad_idx[r]] * ratio(self.map_mode, self.masses, club, c) for c in sources], axis=0)
                np.add.at(T, r, weight[fid]*vals)
        return T

    def draw(self, rng):
        flags = None
        if self.label_groups is not None:
            flags, _ = V.combined_assignment(self.fixtures, self.label_groups, {}, rng, self.tv_flags, {})
        return self.exposure(self.mapping(rng), flags)


class HomeRelabeller:
    """
    Home exposure with the home clubs' stadium catchments relabelled within
    attendance tiers, afresh each ISO week (same season's catchment of the
    relabelled club). Neutral venues never enter. With label_groups, each draw
    first shuffles the home labels among team-events within those groups
    (V.home_permutation_groups).
    """
    def __init__(self, fixtures, lad_idx, n, tiers, label_groups=None, *, map_mode="whole_map", masses=None):
        if map_mode not in {"whole_map", "geography_only"}: raise ValueError(map_mode)
        self.map_mode, self.masses = map_mode, masses
        self.fixtures, self.label_groups, self.lad_idx, self.n = fixtures, label_groups, lad_idx, n
        self.vecs = {(p["club"], p["season"]): p["vec"] for fx in fixtures.values()
                     for p in fx["h_parts"] if p["vec"] is not None}
        if map_mode == "geography_only": validate_masses(masses, self.vecs)
        clubs = {c for c, _ in self.vecs}
        self.tiers = [[c for c in tier if c in clubs] for tier in tiers]
        self.tiers = [tier for tier in self.tiers if tier]
        self.labels = V.observed_home_labels(fixtures)
        self.weeks = sorted({iso_week(fixtures[f]["date"]) for f, _ in self.labels})

    def exposure(self, mapping=None, labels=None):
        labels = self.labels if labels is None else labels
        H = np.zeros(self.n)
        for (fid, i), home in labels.items():
            if not home:
                continue
            part = self.fixtures[fid]["h_parts"][i]
            week = iso_week(self.fixtures[fid]["date"])
            source = part["club"] if mapping is None else mapping.get((part["club"], week), part["club"])
            vec = self.vecs.get((source, part["season"]))
            if vec is None:
                raise ValueError(f"No {part['season']} catchment for {source}")
            factor = ratio(self.map_mode, self.masses, (part["club"], part["season"]), (source, part["season"]))
            np.add.at(H, part["rows"], factor * vec[self.lad_idx[part["rows"]]])
        return H

    def sources(self, club, season):
        tier = next((t for t in self.tiers if club in t), [club])
        return [c for c in tier if (c, season) in self.vecs]

    def mapping(self, rng):
        result = {}
        for week in self.weeks:
            seasons = {p['season'] for fx in self.fixtures.values() if iso_week(fx['date']) == week
                       for p in fx['h_parts'] if p['vec'] is not None}
            if len(seasons) != 1:
                raise ValueError(f"Home permutation week {week} has ambiguous seasons: {seasons}")
            season = next(iter(seasons))
            for tier in self.tiers:
                available = [c for c in tier if (c, season) in self.vecs]
                result.update({(club, week): source for club, source in zip(available, rng.permutation(available))})
        return result

    def expected(self):
        """Exact mean of draw() over the relabelling (and label shuffle): each tier club's
        home rows get the tier's average catchment (same season); with label groups,
        each team-event is weighted by its group's home fraction."""
        tier_of = {c: tier for tier in self.tiers for c in tier}
        weight = ({u: np.mean([self.labels[v] for v in units]) for units in self.label_groups.values() for u in units}
                  if self.label_groups is not None else {u: float(v) for u, v in self.labels.items()})
        H = np.zeros(self.n)
        for (fid, i), w in weight.items():
            if not w:
                continue
            part = self.fixtures[fid]["h_parts"][i]
            sources = self.sources(part["club"], part["season"])
            vecs = [self.vecs.get((c, part["season"])) for c in sources]
            if any(v is None for v in vecs):
                raise ValueError(f"No {part['season']} catchment for a club in {sources}")
            np.add.at(H, part["rows"], w*np.mean([v * ratio(self.map_mode, self.masses, (part["club"], part["season"]), (c, part["season"])) for c,v in zip(sources,vecs)], axis=0)[self.lad_idx[part["rows"]]])
        return H

    def draw(self, rng):
        labels = None
        if self.label_groups is not None:
            _, labels = V.combined_assignment(self.fixtures, {}, self.label_groups, rng, {}, self.labels)
        return self.exposure(self.mapping(rng), labels)


def club_tiers(P):
    """Preserve the original members; place added clubs using explicit size cutoffs.

    TV: >=10%, >=3%, smaller. Home additions: >=50k, >=30k, smaller.
    TV donors may have no fixture in the current week (as in the supervisor code).
    Home donors must have a usable vector among this design's team-events.
    """
    import json
    tv = [list(t) for t in CLUB_TIERS]
    home = [list(t) for t in HOME_TIERS]
    weights = pd.read_csv(M.SUPPORTER_PATH)
    weights['team'] = weights.club.map(M.standardise_team_name)
    sizes = pd.to_numeric(weights.set_index('team')[M.SUPPORTER_WEIGHT_COLUMN], errors='raise').to_dict()
    known_tv, known_home = {c for t in tv for c in t}, {c for t in home for c in t}
    w = pd.read_csv(M.W_PATH)
    w['team'] = w.team.map(M.standardise_team_name)
    w['year'] = w.season.map(M.normalize_season_key)
    attendance = w.groupby(['team', 'year']).avg_home_attendance.first().to_dict()
    parts = [p for fx in P.fixtures.values() for p in fx['h_parts'] if p['vec'] is not None]
    active_home = {p['club'] for p in parts}
    rows = []
    for club in sorted(P.s_tv):
        value = float(sizes.get(club, np.nan))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"No valid supporter size for club tiers: {club}")
        if club not in known_tv:
            tv[0 if value >= 10 else 1 if value >= 3 else 2].append(club)
        used_season, home_size = None, None
        if club in active_home:
            requested = sorted({p['season'] for p in parts if p['club'] == club})
            reference = 2021 if 2021 in requested else requested[0]
            _, used_season, _ = M.resolve_W_vector(club, reference, P.w_lookup, P.seasons_by_team)
            home_size = float(attendance[club, used_season])
            if club not in known_home:
                home[0 if home_size >= 50000 else 1 if home_size >= 30000 else 2].append(club)
        rows.append(dict(team=club, supporter_size_pct=value,
                         tv_tier=next(i+1 for i,t in enumerate(tv) if club in t),
                         home_tier=next((i+1 for i,t in enumerate(home) if club in t), None) if club in active_home else None,
                         home_tier_attendance=home_size, home_tier_attendance_season=used_season,
                         home_permutation_eligible=club in active_home,
                         note='' if club in active_home else 'No usable home team-events in selected fixture/date design'))
    tv = [[c for c in tier if c in P.s_tv] for tier in tv]
    home = [[c for c in tier if c in active_home] for tier in home]
    tv, home = [t for t in tv if t], [t for t in home if t]
    pd.DataFrame(rows).to_csv(M.RESULTS_DIR/'T_home_tv_club_tiers.csv', index=False)
    (M.RESULTS_DIR/'T_home_tv_club_tiers.json').write_text(json.dumps(dict(
        mode=CLUB_MAP_MODE, TV=tv, home=home, allocation='original members retained; added clubs use documented cutoffs',
        home_donors='usable team-season vectors in selected fixtures; no missing attendance invented'), indent=2))
    return tv, home


def make_club_relabellers(P, tv_tiers, home_tiers, mode=None, tv_groups=None, home_groups=None):
    mode = CLUB_MAP_MODE if mode is None else mode
    if mode == 'geography_only':
        if hasattr(P, 'tv_masses') and hasattr(P, 'home_masses'):
            sm, hm = P.tv_masses, P.home_masses
        else:
            sm, hm = source_masses(P)
    else:
        sm, hm = None, None
    return {
        'TV': ClubRelabeller(P.fixtures, V.observed_tv_flags(P.fixtures), P.s_tv,
                            P.data.lad_idx, len(P.df), tv_tiers, tv_groups, map_mode=mode, masses=sm),
        'home': HomeRelabeller(P.fixtures, P.data.lad_idx, len(P.df), home_tiers,
                              home_groups, map_mode=mode, masses=hm),
    }


def prepare(score_projection=True):
    """Inputs, design, fits (resumed from checkpoints) and the nuisance-projected
    score ingredients shared by the permutation scripts."""
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        df, lads, lad_to_idx, unique_dates = M.load_infections()
        B_s, _, _ = M.bspline_basis(unique_dates, spacing_days=M.S_KNOT_SPACING_DAYS, degree=M.SPLINE_DEGREE)
        _, s_raw = M.load_S(lads, lad_to_idx)
        weights = M.load_supporter_weights(sorted(s_raw))
        s_weighted = M.apply_supporter_weights(dict(s_raw), weights)
        _, w_lookup, seasons_by_team = M.load_W(lads, lad_to_idx)
        events = M.load_events()
    if (V.TV_USE_SUPPORTER_WEIGHTS or V.HOME_USE_SUPPORTER_WEIGHTS) and weights is None:
        raise ValueError("Requested supporter weighting could not be applied; preparation stopped.")
    s_tv = s_weighted if V.TV_USE_SUPPORTER_WEIGHTS else s_raw
    s_home = s_weighted if V.HOME_USE_SUPPORTER_WEIGHTS else s_raw
    tv_lookup, fixture_id, slot_lookup = V.load_tv_lookup()
    shell = M.ModelData(df, lads, unique_dates, B_s)
    T, H, fixtures, _ = V.build_exposures(df, unique_dates, shell, s_tv, s_home, w_lookup,
                                          seasons_by_team, events, tv_lookup, fixture_id, slot_lookup)
    if not (T > 0).any() or not (H > 0).any():
        raise ValueError("Both TV and home exposures are required")
    T_scale, H_scale = V.scale_of(T), V.scale_of(H)
    Q = M.delta_basis(B_s, unique_dates) if V.USE_DATE_EFFECT else None
    data = M.ModelData(df, lads, unique_dates, B_s, A_scaled=T / T_scale, H_scaled=H / H_scale,
                       A_scale=T_scale, H_scale=H_scale, Q_delta=Q)
    print(f"input {M.INFECTION_PATH.name}; results {M.RESULTS_DIR}/; fixture rule {M.SUPPORTED_FIXTURE_RULE}; "
          f"unknown TV {V.TV_UNKNOWN_AS}")

    print("fits (resumed from saved checkpoints where available)...", flush=True)
    base = V.checked_fit(data, football=False, label="baseline", resume=True)
    joint = V.checked_fit(data, football=True, start=base, label="joint", resume=True)
    tau1, mu1, a, b = joint["ev"]["tau"], joint["ev"]["mu"], joint["a"], joint["b"]
    F = joint["ev"]["football_multiplier"]
    eta = joint["ev"]["smooth_log"] + joint["ev"]["weekday_log"] + joint["ev"]["delta_log"]
    clipped = (eta <= M.ETA_BOUNDS[0]) | (eta >= M.ETA_BOUNDS[1])
    root_mu = np.sqrt(mu1)
    projector = NuisanceProjector(data, tau1*F/root_mu*(~clipped)) if score_projection else None
    x = projector.residual(np.column_stack([data.A_scaled, data.H_scaled])*(tau1/root_mu)[:, None]) if score_projection else None
    residual = projector.residual((data.y - mu1)/root_mu) if score_projection else None
    counts = np.bincount(data.date_idx, minlength=len(unique_dates))

    def date_demeaned(v):
        return v - (np.bincount(data.date_idx, weights=v, minlength=len(unique_dates))/np.maximum(counts, 1))[data.date_idx]

    return SimpleNamespace(w_lookup=w_lookup, seasons_by_team=seasons_by_team, weights=weights, df=df, unique_dates=unique_dates, data=data, fixtures=fixtures, s_tv=s_tv, T=T, H=H,
                           tau0=base["ev"]["tau"].copy(), tau1=tau1.copy(),
                           joint=dict(beta=joint["beta"], wfree=joint["wfree"], gamma=joint["gamma"]),
                           T_scale=T_scale, H_scale=H_scale, a=a, b=b, projector=projector, x=x,
                           residual=residual, weight=tau1/root_mu, date_demeaned=date_demeaned,
                           to_pct=100*np.array([T[T > 0].mean()/T_scale, H[H > 0].mean()/H_scale]))
VERSION = "constrained-poisson-v1"


@dataclass
class FitDiagnostics:
    converged: bool
    solver: str
    termination_reason: str
    iterations: int
    objective: float
    minimum_multiplier: float
    constraint_violation: float
    optimality_error: float
    attempts: int

    def to_dict(self):
        return asdict(self)


class FrozenFitter:
    def __init__(self, y, bg, tau0, min_multiplier=1e-8):
        self.y, self.bg, self.tau0 = [np.asarray(a, float) for a in (y, bg, tau0)]
        if (self.y.ndim != 1 or self.y.shape != self.bg.shape or self.y.shape != self.tau0.shape
                or not all(np.isfinite(a).all() for a in (self.y, self.bg, self.tau0))
                or (self.y < 0).any() or (self.bg < 0).any() or (self.tau0 <= 0).any()):
            raise ValueError("Frozen fit needs finite aligned counts/background >= 0 and baseline > 0")
        if not 0 < min_multiplier < 1:
            raise ValueError("Minimum multiplier must lie between zero and one")
        self.min_mult = float(min_multiplier)
        self.last_result = None

    def fit(self, T_s, H_s, max_iter=500, tol=1e-13):
        self.last_result = None
        Z = np.column_stack([T_s, H_s]).astype(float)
        if Z.shape != (len(self.y), 2) or not np.isfinite(Z).all() or (Z < 0).any():
            raise ValueError("Exposure columns must be finite, aligned and nonnegative")
        keep = np.any(Z != 0, axis=1)
        Z, y, bg, tau = Z[keep], self.y[keep], self.bg[keep], self.tau0[keep]
        if len(Z) < 2 or np.linalg.matrix_rank(Z) < 2:
            raise ValueError("Frozen coefficients not separately identified: zero/collinear exposure columns")
        points = np.unique(Z, axis=0)
        try:
            vertices = points[ConvexHull(points).vertices] if len(points) > 2 else points
        except QhullError:
            # Keep all constraints when the hull is numerically degenerate.
            vertices = points
        scale = max(float(y.sum()), float(len(y)), 1.)

        def objective(coef):
            mu = bg + tau * (1 + Z @ coef)
            if not np.isfinite(mu).all() or (mu <= 0).any():
                return np.inf, np.zeros(2)
            # Half deviance removes the irrelevant constant from the NLL.
            value = np.sum(mu - y + xlogy(y, y / mu)) / scale
            gradient = Z.T @ ((1 - y / mu) * tau) / scale
            return float(value), gradient

        def hessian(coef):
            mu = bg + tau * (1 + Z @ coef)
            return (Z * (y * tau**2 / mu**2)[:, None]).T @ Z / scale

        def diagnose(result, solver, attempts):
            coef = result.x
            F = 1 + Z @ coef
            violation = max(0., self.min_mult - float(F.min()))
            value, gradient = objective(coef)
            error = np.inf
            if np.isfinite(value) and violation <= 1e-10:
                # Minimize gradient.d over feasible unit-box movements. At a
                # constrained optimum there can be no negative directional slope.
                slack = np.maximum(1 + vertices @ coef - self.min_mult, 0.)
                certificate = linprog(gradient, A_ub=-vertices, b_ub=slack,
                                      bounds=[(-1., 1.), (-1., 1.)], method="highs",
                                      options={"primal_feasibility_tolerance": 1e-9,
                                               "dual_feasibility_tolerance": 1e-9})
                if certificate.success:
                    error = max(0., -float(certificate.fun))
            ok = bool(result.success and np.isfinite(value) and violation <= 1e-10 and error <= 1e-7)
            return FitDiagnostics(ok, solver, str(result.message), int(result.nit),
                                  value * scale, float(F.min()), violation, error, attempts)

        constraint = LinearConstraint(vertices, self.min_mult - 1, np.inf)
        messages = []
        for attempt, start in enumerate((np.zeros(2), np.array([.05, .05])), 1):
            result = minimize(objective, start, jac=True, method="SLSQP", constraints=[constraint],
                              options={"maxiter": max_iter, "ftol": tol})
            diagnostic = diagnose(result, "SLSQP", attempt)
            self.last_result = diagnostic
            if diagnostic.converged:
                return result.x.copy()
            messages.append(diagnostic.to_dict())
        # Independent fallback, on the same objective, exposures and constraints.
        feasible = LinearConstraint(vertices, self.min_mult - 1, np.inf, keep_feasible=True)
        result = minimize(objective, np.zeros(2), jac=True, hess=hessian, method="trust-constr",
                          constraints=[feasible], options={"maxiter": max_iter, "gtol": 1e-12,
                                                          "initial_barrier_parameter": 1e-9,
                                                          "initial_barrier_tolerance": 1e-9,
                                                          "xtol": 1e-12, "barrier_tol": 1e-12})
        diagnostic = diagnose(result, "trust-constr", 3)
        self.last_result = diagnostic
        if diagnostic.converged:
            return result.x.copy()
        messages.append(diagnostic.to_dict())
        raise RuntimeError(f"Frozen optimizer failed validation; this draw must not enter inference: {messages}")

OPTIMIZER_VERSION = VERSION
FROZEN_BASELINE = os.environ.get("T_FROZEN_BASELINE", "averaged")


MASK_QUANTILE = float(os.environ.get("T_FROZEN_MASK_QUANTILE", 0.5))


SCHEMES = [s.strip() for s in os.environ.get("T_FROZEN_SCHEMES", "label,club").split(",") if s.strip()]


N_PERMUTATIONS = V.N_PERMUTATIONS


SEED = V.PERMUTATION_SEED + 21


def summarise(observed, null):
    null = np.asarray(null)
    centre, sd = null.mean(), null.std(ddof=1)
    p_centred = (1 + np.sum(np.abs(null - centre) >= abs(observed - centre) - 1e-12)) / (len(null) + 1)
    p_zero = (1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (len(null) + 1)
    return dict(null_mean_pct=centre, null_sd_pct=sd, z=(observed - centre)/sd if sd > 0 else np.nan,
                p_centred=p_centred, p_uncentred=p_zero)


def refit_baseline(P, name, T_s, H_s, rows=None):
    """Baseline tau on all rows, from a fit on `rows` (all if None) with exposures T_s, H_s
    (football coefficients estimated unless both are zero). Warm start from the complete
    fit; cached by a hash of the inputs."""
    key = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in
                                  [T_s, H_s, np.array([] if rows is None else rows), P.data.y, P.data.bg, P.tau1])).hexdigest()
    key = hashlib.sha256((key + V.fit_signature(P.data) + OPTIMIZER_VERSION + CLUB_MAP_MODE).encode()).hexdigest()
    cache = M.RESULTS_DIR/f"T_home_tv_frozen_tau_{name}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if saved["key"].item() == key:
                return saved["tau"], saved["coef"]
    Q = P.data.Q_delta
    df = P.df if rows is None else P.df.iloc[rows]
    part = M.ModelData(df, P.data.lads, P.unique_dates, P.data.B_s_by_date,
                       A_scaled=T_s if rows is None else T_s[rows], H_scaled=H_s if rows is None else H_s[rows],
                       A_scale=P.T_scale, H_scale=P.H_scale, Q_delta=Q)
    football = bool(np.any(T_s) or np.any(H_s))
    start = dict(beta=P.joint["beta"], wfree=P.joint["wfree"], gamma=P.joint["gamma"])
    result = V.fit_model(part, football=football, start=start, coef0=(P.a, P.b) if football else (0., 0.),
                         label=name)
    if not result["converged"]:
        raise RuntimeError(f"baseline fit {name} did not converge")
    whole = M.ModelData(P.df, P.data.lads, P.unique_dates, P.data.B_s_by_date, Q_delta=Q)
    whole.set_delta(result["gamma"])
    tau = M.evaluate_A_arrays(whole, result["beta"], result["wfree"])["tau"]
    coef = np.array([result["a"], result["b"]])
    np.savez_compressed(cache, tau=tau, coef=coef, key=key)
    return tau, coef


def _main():
    import json
    status = M.RESULTS_DIR/"T_home_tv_frozen_status.json"
    status.write_text(json.dumps({"status":"running", "inference_complete":False}))
    if N_PERMUTATIONS < 19:
        raise ValueError("At least 19 permutation draws are required; 999 recommended")
    (M.RESULTS_DIR/"T_home_tv_frozen_optimizer.csv").write_text("")
    P = prepare(score_projection=False)
    data, fixtures, n = P.data, P.fixtures, len(P.df)
    if FROZEN_BASELINE not in ("no_football", "joint", "averaged", "masked"):
        raise ValueError("T_FROZEN_BASELINE must be 'no_football', 'joint', 'averaged' or 'masked'")
    y, bg = data.y.astype(float), data.bg.astype(float)
    T_s, H_s = P.T/P.T_scale, P.H/P.H_scale
    to_pct = P.to_pct                                   # uplift per unit coefficient, observed exposures
    print(f"\nfrozen baseline: {FROZEN_BASELINE}" + (f" (mask quantile {MASK_QUANTILE})" if FROZEN_BASELINE == "masked" else ""))
    print(f"joint fit: TV {P.a*to_pct[0]:+.3f}%, home {P.b*to_pct[1]:+.3f}%", flush=True)

    CLUB_TIERS, HOME_TIERS = club_tiers(P)
    tv_flags, home_labels = V.observed_tv_flags(fixtures), V.observed_home_labels(fixtures)
    tv_groups, home_groups = V.tv_permutation_groups(fixtures), V.home_permutation_groups(fixtures)
    # the label shuffles are relabellers with no club tiers: expected() then averages the labels only
    relabellers = {
        "label": {"TV": ClubRelabeller(fixtures, tv_flags, P.s_tv, data.lad_idx, n, [], tv_groups),
                  "home": HomeRelabeller(fixtures, data.lad_idx, n, [], home_groups)},
    }
    if "club" in SCHEMES:
        relabellers["club"] = make_club_relabellers(P, CLUB_TIERS, HOME_TIERS)
    drawers = {
        "label": {
            "TV": lambda rng: V.assemble_T(n, fixtures, V.combined_assignment(fixtures, tv_groups, {}, rng,
                                                                                  tv_flags, home_labels)[0]),
            "home": lambda rng: V.assemble_H(n, fixtures, V.combined_assignment(fixtures, {}, home_groups, rng,
                                                                                    tv_flags, home_labels)[1]),
        },
    }
    if "club" in SCHEMES:
        drawers["club"] = {term: rel.draw for term, rel in relabellers["club"].items()}

    def baseline(scheme, term, k):
        """(tau, description) for one test."""
        if FROZEN_BASELINE == "no_football":
            return P.tau0, ""
        if FROZEN_BASELINE == "joint":
            return P.tau1, ""
        rel = relabellers[scheme][term]
        if not np.allclose(rel.exposure(), P.T if term == "TV" else P.H):
            raise ValueError(f"{scheme} {term} relabeller does not reproduce the observed exposure")
        avg = rel.expected()/(P.T_scale if term == "TV" else P.H_scale)
        name = f"{FROZEN_BASELINE}{'' if FROZEN_BASELINE == 'averaged' else int(round(100*MASK_QUANTILE))}_{scheme}_{term}"
        if FROZEN_BASELINE == "averaged":
            tau, coef = refit_baseline(P, name, avg if term == "TV" else T_s, H_s if term == "TV" else avg)
            return tau, f"averaged-design fit: TV {coef[0]*to_pct[0]:+.3f}%, home {coef[1]*to_pct[1]:+.3f}%"
        exposed = avg > 0
        keep = ~exposed | (avg <= np.quantile(avg[exposed], MASK_QUANTILE))
        rows = np.flatnonzero(keep)
        tau, _ = refit_baseline(P, name, np.zeros(n), np.zeros(n), rows)
        return tau, (f"masked {1-keep.mean():.1%} of LTLA-days, "
                     f"{1-avg[keep].sum()/avg.sum():.0%} of the averaged {term} exposure")

    rows, null_frames = [], []
    for scheme in SCHEMES:
        if scheme not in drawers:
            raise ValueError(f"unknown scheme {scheme!r}; use label and/or club")
        for k, term in enumerate(["TV", "home"]):
            tau, note = baseline(scheme, term, k)
            fitter = FrozenFitter(y, bg, tau, min_multiplier=V.MIN_MULTIPLIER)
            observed = fit_and_record(fitter, T_s, H_s, scheme, term, -1)
            obs_pct = observed[k]*to_pct[k]
            rng = np.random.default_rng(SEED + 10*SCHEMES.index(scheme) + k)
            null = []
            for draw in range(N_PERMUTATIONS):
                E = drawers[scheme][term](rng)
                coef = fit_and_record(fitter, E/P.T_scale if term == "TV" else T_s,
                                      H_s if term == "TV" else E/P.H_scale, scheme, term, draw)
                null.append(coef[k]*to_pct[k])
                if (draw+1) % 200 == 0:
                    print(f"  {scheme} {term} {draw+1}/{N_PERMUTATIONS}", flush=True)
            stats = summarise(obs_pct, null)
            rows.append(dict(input=str(M.INFECTION_PATH), tv_unknown_as=V.TV_UNKNOWN_AS,
                             tv_supporter_scale=M.TV_SUPPORTER_SCALE,
                             tv_audience_total=M.TV_AUDIENCE_TOTAL if V.TV_USE_SUPPORTER_WEIGHTS and M.TV_SUPPORTER_SCALE=='assumed_audience' else None,
                             fixture_rule=M.SUPPORTED_FIXTURE_RULE, frozen_baseline=FROZEN_BASELINE,
                             mask_quantile=MASK_QUANTILE if FROZEN_BASELINE == "masked" else np.nan,
                             design_signature=V.fit_signature(data), optimizer=OPTIMIZER_VERSION, club_map_mode=CLUB_MAP_MODE,
                             scheme=scheme, term=term,
                             grouping=(V.TV_PERMUTATION_WITHIN if term == "TV" else V.HOME_PERMUTATION_WITHIN)
                             if scheme == "label" else "week, size tiers",
                             n_permutations=N_PERMUTATIONS, observed_uplift_pct=obs_pct,
                             other_term_uplift_pct=observed[1-k]*to_pct[1-k], baseline_note=note, **stats))
            null_frames.append(pd.DataFrame({"scheme": scheme, "term": term, "null_uplift_pct": null}))
            r = rows[-1]
            if note:
                print(f"  [{scheme} {term}] {note}")
            print(f"  {scheme:5s} {term:4s}: observed {r['observed_uplift_pct']:+.3f}%  null mean {r['null_mean_pct']:+.3f}%  "
                  f"sd {r['null_sd_pct']:.3f}  z {r['z']:+.2f}  p (centred) {r['p_centred']:.3f}  "
                  f"p (uncentred) {r['p_uncentred']:.3f}", flush=True)
    suffix = {"no_football": "", "joint": "_joint", "averaged": "_averaged",
              "masked": f"_masked{int(round(100*MASK_QUANTILE))}"}[FROZEN_BASELINE]
    stem = "T_home_tv_frozen_baseline" + suffix
    pd.DataFrame(rows).to_csv(M.RESULTS_DIR/f"{stem}.csv", index=False)
    pd.concat(null_frames).to_csv(M.RESULTS_DIR/f"{stem}_null.csv", index=False)
    print(f"wrote {M.RESULTS_DIR/(stem + '.csv')}")
    status.write_text(json.dumps({"status":"completed", "inference_complete":True, "optimizer":OPTIMIZER_VERSION}))


def fit_and_record(fitter, T, H, scheme, term, draw):
    import csv
    path = M.RESULTS_DIR/'T_home_tv_frozen_optimizer.csv'
    try:
        return fitter.fit(T, H)
    finally:
        row = dict(scheme=scheme, term=term, draw=draw,
                   **(fitter.last_result.to_dict() if fitter.last_result else dict(
                       converged=False, solver='not_started', termination_reason='input validation failed',
                       iterations=0, objective=None, minimum_multiplier=None, constraint_violation=None,
                       optimality_error=None, attempts=0)))
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if path.stat().st_size == 0: writer.writeheader()
            writer.writerow(row)


def main():
    import json
    try:
        _main()
    except BaseException as exc:
        (M.RESULTS_DIR/'T_home_tv_frozen_status.json').write_text(json.dumps(dict(
            status='failed_or_interrupted', inference_complete=False, error=str(exc))))
        raise

if __name__ == "__main__":
    main()
