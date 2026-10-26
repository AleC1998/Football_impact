"""Input readers, baseline mathematics and bootstrap projection used by fit.py.

Unused Model A/B/C comparison runners have been removed. The retained
calculation functions come from Model/T_estimation_joint_models.py.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import csv
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
ALIASES = {
    'Man City': 'Manchester City', 'Man United': 'Manchester United',
    'Newcastle': 'Newcastle United', 'Tottenham': 'Tottenham Hotspur',
    'West Ham': 'West Ham United', 'Wolves': 'Wolverhampton Wanderers',
    "Nott'm Forest": 'Nottingham Forest', 'Forest': 'Nottingham Forest',
    'Leeds United': 'Leeds', 'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton', 'Brighton and Hove Albion': 'Brighton',
}


def canonical_team(name):
    name = str(name).strip()
    if name.endswith(' FC'):
        name = name[:-3]
    return ALIASES.get(name, name)


def clubs():
    with (DATA/'clubs.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def stadiums():
    return {row['team']: (float(row['longitude']), float(row['latitude'])) for row in clubs()}


def teams():
    return [row['team'] for row in clubs()]

"""Allocate an assumed audience pool without changing the saved S geography.

S_e = audience_total * p_e / sum(p). The pool is an assumption, not measured
viewers or a constraint on each week's total. A televised club appearance uses
its allocation once; repeated appearances contribute repeatedly. No LAD cap,
weekly renormalization, or population redistribution is applied.
"""

import math

audience_DEFAULT_MODE = 'assumed_audience'

audience_DEFAULT_TOTAL = 15_000_000.0

audience_MODES = ('assumed_audience', 'relative_mean_one')

def audience_validate_policy(mode, total):
    if mode not in audience_MODES:
        raise ValueError(f'Unknown TV supporter scale: {mode!r}; expected {audience_MODES}')
    if not math.isfinite(float(total)) or float(total) <= 0:
        raise ValueError('Assumed TV audience total must be finite and positive')

def audience_allocations(raw, mode=audience_DEFAULT_MODE, total=audience_DEFAULT_TOTAL):
    """Normalize over the supplied S roster, not over fixtures or retained LADs."""
    audience_validate_policy(mode, total)
    if not raw or any(not math.isfinite(float(v)) or float(v) < 0 for v in raw.values()):
        raise ValueError('Club supporter values must be finite, nonnegative and nonempty')
    denominator = math.fsum(float(v) for v in raw.values())
    if denominator <= 0:
        raise ValueError('Club supporter values must have a positive sum')
    mass = float(total) if mode == 'assumed_audience' else float(len(raw))
    return {club: mass * (float(value) / denominator) for club, value in raw.items()}

def audience_metadata(weights, mode=audience_DEFAULT_MODE, total=audience_DEFAULT_TOTAL, enabled=True):
    audience_validate_policy(mode, total)
    assumed = enabled and mode == 'assumed_audience'
    fractions = ({club: value / math.fsum(weights.values()) for club, value in weights.items()}
                 if weights and math.fsum(weights.values()) > 0 else {})
    return dict(tv_supporter_scale=mode,
                tv_audience_total=float(total) if assumed else None,
                tv_audience_is_assumed=bool(assumed),
                tv_exposure_units=('assumed_audience_events_per_resident' if assumed else
                                   'relative_fanbase_units_per_resident' if enabled else
                                   'unit_club_fanbase_per_resident'),
                supporter_fractions=fractions if enabled else {},
                club_audience_allocations=dict(weights) if assumed and weights else {},
                tv_audience_scope='Fixed pool across the S-file club roster; applied per televised club appearance; no weekly renormalization',
                tv_audience_geography='Allocated entirely through the unchanged England S maps; no UK-to-England conversion',
                tv_audience_headcounts_observed=False,
                tv_audience_population_capping=False)

from pathlib import Path

import json

import os

import re

import warnings

import sys

import numpy as np

import pandas as pd

from scipy.interpolate import BSpline

from scipy.linalg import null_space

from scipy.optimize import OptimizeResult, minimize

from scipy.special import gammaln

PROJECT_DIR = Path(__file__).resolve().parent

sys.path.insert(0,str(PROJECT_DIR.parent))

study_clubs = clubs

DATA_DIR = DATA

RESULTS_DIR = Path(os.environ.get("T_RESULTS_DIR", str(RESULTS / "fit")))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Here load the proper outcome/background panel (not supplied).
# Columns: ltla, date (YYYY-MM-DD), number_infections, number_bg.
# One row per area/date; numeric fields must be finite and nonnegative.
# For contact/exposure fits, number_infections holds the chosen contact/exposure outcome.
INFECTION_PATH = DATA / os.environ.get("T_INFECTION_FILE", "example_outcome.csv")

S_PATH = Path(os.environ.get("T_S_FILE", str(DATA_DIR / "lad_level_fan_distribution.csv")))

W_PATH = Path(os.environ.get("T_W_FILE", str(DATA_DIR / "regional_non_big_six_W_reported_average.csv")))

RESTRICT_TO_SUPPORTED_FIXTURES = True

SUPPORTED_FIXTURE_RULE = os.environ.get("T_FIXTURE_RULE", "any")

def supported_fixture_mask(home_teams, away_teams, teams):
    home_ok = home_teams.map(standardise_team_name).isin(teams)
    away_ok = away_teams.map(standardise_team_name).isin(teams)
    if SUPPORTED_FIXTURE_RULE == "both":
        return home_ok & away_ok
    if SUPPORTED_FIXTURE_RULE == "any":
        return home_ok | away_ok
    raise ValueError(f"T_FIXTURE_RULE must be 'both' or 'any', not {SUPPORTED_FIXTURE_RULE!r}")

# Here load the proper league and cup fixture files (not supplied).
# League columns: Date, HomeTeam, AwayTeam, Season, Home_LAD_code_Apr2020, Home_LAD_name_Apr2020.
# Cup columns: Date, HomeTeam, AwayTeam, Season, Competition, NeutralVenue, LAD_code, LAD_name.
# Dates are YYYY-MM-DD; Season uses e.g. 2021/22; NeutralVenue is True/False.
EVENT_PATHS = [
    DATA / "example_league_matches.csv",
    DATA / "example_cup_matches.csv",
]

if os.environ.get("T_EVENT_FILES_JSON"):
    _event_files = json.loads(os.environ["T_EVENT_FILES_JSON"])
    if not isinstance(_event_files, list) or not _event_files or not all(isinstance(p, str) for p in _event_files):
        raise ValueError("T_EVENT_FILES_JSON must be a nonempty JSON list of fixture paths")

    EVENT_PATHS = [Path(p) for p in _event_files]

SUPPORTER_PATH = Path(os.environ.get("T_SUPPORTER_FILE", str(DATA / "premier_league_supporter_surveys.csv")))

NEUTRAL_VENUE_COL = "NeutralVenue"

SUPPORTER_WEIGHT_COLUMN = os.environ.get("T_SUPPORTER_WEIGHT_COLUMN", "working_2023_favourite_main_team_pct")

SUPPORTER_IMPUTE_FROM = None

TV_SUPPORTER_SCALE = os.environ.get('T_TV_SUPPORTER_SCALE', audience_DEFAULT_MODE)

TV_AUDIENCE_TOTAL = float(os.environ.get('T_TV_AUDIENCE_TOTAL', audience_DEFAULT_TOTAL))

audience_validate_policy(TV_SUPPORTER_SCALE, TV_AUDIENCE_TOTAL)

LOCATION_COL = "ltla"

DATE_COL = "date"

INFECTIONS_COL = "number_infections"

BACKGROUND_COL = "number_bg"

DATE_MIN = os.environ.get("T_DATE_MIN", "2021-06-01") or None

DATE_MAX = os.environ.get("T_DATE_MAX", "2022-07-09") or None

EXCLUDE_DATE_RANGES = []

MIN_LTLAS_PER_DATE = None

S_KNOT_SPACING_DAYS = 21

OMEGA_KNOT_SPACING_DAYS = 35

SPLINE_DEGREE = 3

S_SMOOTH_PENALTY = 25.0

OMEGA_SMOOTH_PENALTY = 10.0

MATCH_WINDOW_START_DAYS = 0

MATCH_WINDOW_END_DAYS = 0

HOME_WINDOW_START_DAYS = 0

HOME_WINDOW_END_DAYS = 0

MAX_BLOCK_ITER = 25

BLOCK_REL_TOL = 1e-7

MAX_BLOCK_ITER_C = 250

MAXITER_LTLA = 1000

MAXITER_GLOBAL = 2000

FTOL = 1e-11

GTOL = 1e-6

COEF_BOUNDS = (1e-12, 1e6)

CONTRIBUTION_RATIO_BOUNDS = (1e-8, 1e8)

LOG_SAFE_CLIP = (-700.0, 700.0)

FOOTBALL_START_SCALED_CA = 0.01

FOOTBALL_START_LOG_R_S = 0.0

FOOTBALL_MULTISTART_COEFS = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)

LOG_BOUND_TOLERANCE = 1e-6

FOOTBALL_NEGLIGIBLE_CONTRIBUTION = 1e-6

FOOTBALL_NEGLIGIBLE_RATIO = 1e-3

W_SEASON_FALLBACK = "nearest"  # "nearest" or "none"

W_MAX_SEASON_GAP = 2

EPS = 1e-12

ETA_BOUNDS = (-700.0, 300.0)

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Man City": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Man United": "Manchester United",
    "Newcastle United FC": "Newcastle United",
    "Newcastle": "Newcastle United",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "West Ham United FC": "West Ham United",
    "West Ham": "West Ham United",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Wolves": "Wolverhampton Wanderers",
}

def standardise_team_name(value: str) -> str:
    value = str(value).strip()
    return canonical_team(TEAM_NAME_MAP.get(value, value))

EXPECTED_STADIUM_LADS = {
    "Arsenal": ("Emirates Stadium", "E09000019", "Islington"),
    "Aston Villa": ("Villa Park", "E08000025", "Birmingham"),
    "Brentford": ("Brentford Community Stadium", "E09000018", "Hounslow"),
    "Brighton": ("Amex Stadium", "E06000043", "Brighton and Hove"),
    "Burnley": ("Turf Moor", "E07000117", "Burnley"),
    "Chelsea": ("Stamford Bridge", "E09000013", "Hammersmith and Fulham"),
    "Crystal Palace": ("Selhurst Park", "E09000008", "Croydon"),
    "Everton": ("Goodison Park", "E08000012", "Liverpool"),
    "Leeds": ("Elland Road", "E08000035", "Leeds"),
    "Leicester": ("King Power Stadium", "E06000016", "Leicester"),
    "Liverpool": ("Anfield", "E08000012", "Liverpool"),
    "Manchester City": ("Etihad Stadium", "E08000003", "Manchester"),
    "Manchester United": ("Old Trafford", "E08000009", "Trafford"),
    "Newcastle United": ("St James' Park", "E08000021", "Newcastle upon Tyne"),
    "Norwich": ("Carrow Road", "E07000148", "Norwich"),
    "Southampton": ("St Mary's Stadium", "E06000045", "Southampton"),
    "Tottenham Hotspur": ("Tottenham Hotspur Stadium", "E09000014", "Haringey"),
    "Watford": ("Vicarage Road", "E07000103", "Watford"),
    "West Ham United": ("London Stadium", "E09000025", "Newham"),
    "Wolverhampton Wanderers": ("Molineux Stadium", "E08000031", "Wolverhampton"),
}

_lad_names = pd.read_csv(DATA/'ONS_LAD_April2020_Region_Lookup.csv').set_index('LAD20CD').LAD20NM.to_dict()

for row in study_clubs():
    stadium_name = EXPECTED_STADIUM_LADS.get(row['team'], (row['stadium'],))[0]
    EXPECTED_STADIUM_LADS[row['team']] = (stadium_name, row['stadium_lad'], _lad_names[row['stadium_lad']])

def validate_stadium_lads(raw: pd.DataFrame) -> None:
    code_column = next((c for c in ["Home_LAD_code_Apr2020", "LAD_code", "Home_LAD_code"] if c in raw), None)
    name_column = next((c for c in ["Home_LAD_name_Apr2020", "LAD_name", "Home_LAD_name"] if c in raw), None)
    if "HomeTeam" not in raw or code_column is None or name_column is None:
        warnings.warn("Fixture file has no complete stadium/LAD metadata; validation skipped.")
        return

    # A neutral venue (e.g. an FA Cup semi-final at Wembley) has no home club, so
    # HomeTeam is nominal and its stadium/LAD must not be checked against the
    # club's own ground.
    checkable = raw
    if NEUTRAL_VENUE_COL in raw.columns:
        neutral = raw[NEUTRAL_VENUE_COL].astype(bool)
        if neutral.any():
            print(f"Neutral-venue fixtures excluded from stadium validation: {int(neutral.sum())}")
        checkable = raw[~neutral]

    columns = ["HomeTeam", code_column, name_column]
    if "HomeStadium" in raw:
        columns.append("HomeStadium")
    check = checkable[columns].drop_duplicates().copy()
    check["team_std"] = check["HomeTeam"].map(standardise_team_name)

    problems = []
    checked_teams = set()
    for _, row in check.iterrows():
        team = row["team_std"]
        expected = EXPECTED_STADIUM_LADS.get(team)
        if expected is None:
            continue
        checked_teams.add(team)
        observed = (str(row[code_column]).strip(), str(row[name_column]).strip())
        if "HomeStadium" in check:
            observed = (str(row["HomeStadium"]).strip(), *observed)
        else:
            expected = expected[1:]
        if observed != expected:
            problems.append((team, observed, expected))

    if problems:
        msg = "\n".join(
            f"  {team}: observed={observed}, expected={expected}"
            for team, observed, expected in problems
        )
        raise ValueError("Stadium/LAD validation failed:\n" + msg)

    print(f"Validated stadium/LAD assignments for {len(checked_teams)} known home teams.")

def season_start_year_from_date(date: pd.Timestamp) -> int:
    date = pd.Timestamp(date)
    return int(date.year if date.month >= 7 else date.year - 1)

def normalize_season_key(value):
    """
    Convert common season representations to the season start year.

    Examples
    --------
    2019-2020 -> 2019
    2019/20   -> 2019
    2019_20   -> 2019
    19-20     -> 2019
    20-21     -> 2020
    21-22     -> 2021
    2021-22   -> 2021
    2021/2022 -> 2021
    2021      -> 2021
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    # --------------------------------------------------------
    # Four-digit start year, e.g. 2021-22, 2021/2022
    # --------------------------------------------------------
    match = re.search(r"\b(19|20)\d{2}\b", text)

    if match:
        return int(match.group(0))

    # --------------------------------------------------------
    # Short season format, e.g. 21-22, 20-21, 19-20
    # --------------------------------------------------------
    match = re.search(r"\b(\d{2})\s*[-_/]\s*\d{2}\b", text)

    if match:
        yy = int(match.group(1))

        # Your data are modern football seasons.
        # 19 -> 2019, 20 -> 2020, 21 -> 2021, etc.
        if yy <= 30:
            return 2000 + yy

        return 1900 + yy

    # --------------------------------------------------------
    # Plain numeric year
    # --------------------------------------------------------
    try:
        value_int = int(float(text))

        if 1900 <= value_int <= 2200:
            return value_int

        if 0 <= value_int <= 99:
            return 2000 + value_int

    except Exception:
        pass

    return None

def choose_column(columns, candidates):
    lower = {str(c).lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower().strip() in lower:
            return lower[candidate.lower().strip()]
    return None

def bspline_basis(dates, spacing_days: int, degree: int = 3):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    origin = dates.min()
    x = (dates - origin).days.to_numpy(dtype=float)
    xmax = float(x.max())

    if xmax <= 0:
        return np.ones((len(x), 1), dtype=float), np.array([0.0]), x

    interior = np.arange(float(spacing_days), xmax, float(spacing_days))
    knots = np.concatenate([
        np.repeat(0.0, degree + 1),
        interior,
        np.repeat(xmax, degree + 1),
    ])
    n_basis = len(knots) - degree - 1
    basis = BSpline(knots, np.eye(n_basis), degree, extrapolate=True)(x)
    return np.asarray(basis, dtype=float), knots, x

def roughness_1d(beta: np.ndarray, lam: float):
    beta = np.asarray(beta, dtype=float)
    if lam <= 0 or len(beta) < 3:
        return 0.0, np.zeros_like(beta)

    d2 = beta[2:] - 2.0 * beta[1:-1] + beta[:-2]
    penalty = 0.5 * lam * float(np.dot(d2, d2))

    grad = np.zeros_like(beta)
    grad[:-2] += lam * d2
    grad[1:-1] += -2.0 * lam * d2
    grad[2:] += lam * d2
    return penalty, grad

def roughness_2d(beta: np.ndarray, lam: float):
    penalty = 0.0
    grad = np.zeros_like(beta, dtype=float)
    for i in range(beta.shape[0]):
        p, g = roughness_1d(beta[i], lam)
        penalty += p
        grad[i] = g
    return penalty, grad

def constrained_weekday_effects(free_effects: np.ndarray) -> np.ndarray:
    """
    Six free log effects + one reference log effect chosen so all seven sum to 0.
    Therefore the geometric mean weekday multiplier is exactly 1.
    """
    free_effects = np.asarray(free_effects, dtype=float)
    reference = -np.sum(free_effects, axis=-1, keepdims=True)
    return np.concatenate([free_effects, reference], axis=-1)

def poisson_loglik(y, mu):
    mu = np.maximum(np.asarray(mu, dtype=float), EPS)
    y = np.asarray(y, dtype=float)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))

def relative_change(old, new):
    if not np.isfinite(old):
        return np.inf
    return abs(old - new) / max(1.0, abs(old))

def apply_date_filters(df):
    """
    Restrict the panel to the dates the model should see. Runs before any index
    is built, so lads, unique_dates, date_idx and the spline bases are all
    derived from the retained dates only.
    """
    n_before, dates_before = len(df), df[DATE_COL].nunique()
    reasons = []

    if DATE_MIN is not None:
        keep = df[DATE_COL] >= pd.Timestamp(DATE_MIN)
        reasons.append((f"before {DATE_MIN}", int((~keep).sum())))
        df = df[keep]

    if DATE_MAX is not None:
        keep = df[DATE_COL] <= pd.Timestamp(DATE_MAX)
        reasons.append((f"after {DATE_MAX}", int((~keep).sum())))
        df = df[keep]

    for start, end in EXCLUDE_DATE_RANGES:
        drop = df[DATE_COL].between(pd.Timestamp(start), pd.Timestamp(end))
        reasons.append((f"excluded range {start}..{end}", int(drop.sum())))
        df = df[~drop]

    if MIN_LTLAS_PER_DATE is not None:
        counts = df.groupby(DATE_COL)[LOCATION_COL].nunique()
        sparse = set(counts.index[counts < MIN_LTLAS_PER_DATE])
        drop = df[DATE_COL].isin(sparse)
        reasons.append(
            (f"fewer than {MIN_LTLAS_PER_DATE} LTLAs ({len(sparse)} dates)", int(drop.sum()))
        )
        df = df[~drop]

    df = df.reset_index(drop=True)

    if any(count for _, count in reasons):
        print("\nDate filtering:")
        for reason, count in reasons:
            if count:
                print(f"  dropped {count:,} rows: {reason}")
        print(
            f"  panel: {n_before:,} rows / {dates_before} dates "
            f"-> {len(df):,} rows / {df[DATE_COL].nunique()} dates"
        )

    if df.empty:
        raise ValueError("Date filtering removed every row; loosen the DATE_* settings.")

    return df

def load_infections():
    if not INFECTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing model input CSV: {INFECTION_PATH}. "
            "The placeholder is data/example_outcome.csv; select another filename "
            "with T_INFECTION_FILE. Required columns: ltla,date,number_infections,number_bg. "
            "Load the proper outcome/background panel at INFECTION_PATH."
        )

    df = pd.read_csv(INFECTION_PATH)
    required = [LOCATION_COL, DATE_COL, INFECTIONS_COL, BACKGROUND_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{INFECTION_PATH} is missing columns: {missing}")

    df = df[required].copy()
    if df[LOCATION_COL].isna().any():
        raise ValueError("Missing LTLA code in infection input")
    df[LOCATION_COL] = df[LOCATION_COL].astype(str).str.strip()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[INFECTIONS_COL] = pd.to_numeric(df[INFECTIONS_COL], errors="coerce")
    df[BACKGROUND_COL] = pd.to_numeric(df[BACKGROUND_COL], errors="coerce")
    if df[required].isna().any().any():
        raise ValueError("Infection input contains missing/invalid required values.")
    if not np.isfinite(df[[INFECTIONS_COL, BACKGROUND_COL]].to_numpy()).all():
        raise ValueError("Infection/background values must be finite.")
    if df.duplicated([LOCATION_COL, DATE_COL]).any():
        raise ValueError("Duplicate LTLA/date observations in infection input.")
    df = df.sort_values([DATE_COL, LOCATION_COL]).reset_index(drop=True)
    df = apply_date_filters(df)

    if (df[INFECTIONS_COL] < 0).any():
        raise ValueError("number_infections must be non-negative.")
    if (df[BACKGROUND_COL] < 0).any():
        raise ValueError("number_bg must be non-negative.")

    # Requested raw no-football quantity. It is retained as a diagnostic/output;
    # the likelihood itself is fit to integer I with mean I_bg + tau.
    df["T_raw_I_minus_Ibg"] = df[INFECTIONS_COL] - df[BACKGROUND_COL]
    df["weekday"] = df[DATE_COL].dt.dayofweek.astype(int)
    df["season_key"] = df[DATE_COL].map(season_start_year_from_date).astype(int)

    lads = np.array(sorted(df[LOCATION_COL].unique()))
    lad_to_idx = {lad: i for i, lad in enumerate(lads)}
    df["lad_idx"] = df[LOCATION_COL].map(lad_to_idx).astype(int)

    unique_dates = pd.DatetimeIndex(sorted(df[DATE_COL].unique()))
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}
    df["date_idx"] = df[DATE_COL].map(date_to_idx).astype(int)

    return df, lads, lad_to_idx, unique_dates

def load_S(lads, lad_to_idx):
    if not S_PATH.exists():
        raise FileNotFoundError(
            f"Missing S output: {S_PATH}\nRun run_all.py first."
        )

    s = pd.read_csv(S_PATH)
    required = ["team", "lad_code", "S_x"]
    missing = [c for c in required if c not in s.columns]
    if missing:
        raise ValueError(f"{S_PATH} is missing columns: {missing}")

    s = s[required].copy()
    s["team"] = s["team"].map(standardise_team_name)
    s["lad_code"] = s["lad_code"].astype(str).str.strip()
    s["S_x"] = pd.to_numeric(s["S_x"], errors="coerce")
    validate_exposure_table(s, ["team", "lad_code"], "S_x", S_PATH)
    report_geographic_coverage(s, lads, "S")
    s = s[s["lad_code"].isin(lad_to_idx)].copy()

    lookup = {}
    for team, part in s.groupby("team"):
        vector = np.zeros(len(lads), dtype=float)
        idx = part["lad_code"].map(lad_to_idx).to_numpy(dtype=int)
        vector[idx] = part["S_x"].to_numpy(dtype=float)
        lookup[team] = vector

    return s, lookup

def load_W(lads, lad_to_idx):
    if not W_PATH.exists():
        raise FileNotFoundError(
            f"Missing W output: {W_PATH}\nRun fit_catchment_model.py first."
        )

    w = pd.read_csv(W_PATH)
    for metadata_column in ["scenario", "attendance_basis"]:
        if metadata_column in w and w[metadata_column].nunique(dropna=False) != 1:
            raise ValueError(f"{W_PATH} mixes {metadata_column} values; select one W scenario explicitly.")
    required = ["season", "team", "lad_code", "W"]
    missing = [c for c in required if c not in w.columns]
    if missing:
        raise ValueError(f"{W_PATH} is missing columns: {missing}")

    w = w[required].copy()
    w["team"] = w["team"].map(standardise_team_name)
    w["lad_code"] = w["lad_code"].astype(str).str.strip()
    w["W"] = pd.to_numeric(w["W"], errors="coerce")
    w["season_key"] = w["season"].map(normalize_season_key).astype("Int64")
    validate_exposure_table(w, ["team", "season_key", "lad_code"], "W", W_PATH)
    report_geographic_coverage(w, lads, "W")
    w["season_key"] = w["season_key"].astype(int)
    print("\nW season conversion:")
    print(
    w[["season", "season_key"]]
    .drop_duplicates()
    .sort_values(["season_key", "season"])
    .to_string(index=False)
    )
    w = w[w["lad_code"].isin(lad_to_idx)].copy()

    lookup = {}
    for (team, season_key), part in w.groupby(["team", "season_key"]):
        vector = np.zeros(len(lads), dtype=float)
        idx = part["lad_code"].map(lad_to_idx).to_numpy(dtype=int)
        vector[idx] = part["W"].to_numpy(dtype=float)
        lookup[(team, int(season_key))] = vector

    seasons_by_team = {
        team: sorted(part["season_key"].unique().astype(int).tolist())
        for team, part in w.groupby("team")
    }

    return w, lookup, seasons_by_team

def validate_exposure_table(table, keys, column, path):
    if table[keys + [column]].isna().any().any():
        raise ValueError(f"Missing/invalid exposure values or keys in {path}")
    if table.duplicated(keys).any():
        raise ValueError(f"Duplicate exposure keys {keys} in {path}")
    values = table[column].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{column} must be finite and nonnegative in {path}")

def report_geographic_coverage(table, lads, label):
    """Preserve source geography; expose mismatches instead of inventing a crosswalk."""
    available, wanted = set(table.lad_code), set(lads)
    missing, extra = sorted(wanted - available), sorted(available - wanted)
    pd.DataFrame([{"lad_code": code, "status": status}
                  for status, codes in [("panel_without_exposure", missing),
                                        ("exposure_outside_panel", extra)] for code in codes],
                 columns=["lad_code", "status"]).to_csv(RESULTS_DIR / f"{label}_geography_coverage.csv", index=False)
    if missing or extra:
        warnings.warn(f"{label}: {len(missing)} panel LADs lack exposure rows; {len(extra)} source LADs are outside panel. "
                      f"Source geography preserved; inspect {label}_geography_coverage.csv.")

def supported_teams():
    table = pd.read_csv(S_PATH)
    if "team" not in table or table.team.isna().any():
        raise ValueError(f"Invalid team column in {S_PATH}")
    return set(table.team.map(standardise_team_name))

def resolve_W_vector(team, requested_season, w_lookup, seasons_by_team):
    exact = w_lookup.get((team, int(requested_season)))
    if exact is not None:
        return exact, int(requested_season), "exact"

    if W_SEASON_FALLBACK != "nearest":
        return None, None, "missing"

    available = seasons_by_team.get(team, [])
    if not available:
        return None, None, "missing_team"

    nearest = min(available, key=lambda s: abs(int(s) - int(requested_season)))
    gap = abs(int(nearest) - int(requested_season))
    if gap > W_MAX_SEASON_GAP:
        return None, None, "missing_too_far"

    return w_lookup[(team, int(nearest))], int(nearest), "nearest"

def load_events_file(path):
    """Home/away team-events from one fixture file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture file: {path}")

    raw = pd.read_csv(path)
    validate_stadium_lads(raw)

    date_col = choose_column(raw.columns, ["date", "match_date", "fixture_date", "kickoff_date"])
    home_col = choose_column(raw.columns, ["home_team", "hometeam", "home"])
    away_col = choose_column(raw.columns, ["away_team", "awayteam", "away"])
    season_col = choose_column(raw.columns, ["season", "season_name", "season_id"])

    if date_col is None or home_col is None or away_col is None:
        raise ValueError(
            f"{path} must contain date, home-team and away-team columns.\n"
            f"Found: {list(raw.columns)}"
        )

    if RESTRICT_TO_SUPPORTED_FIXTURES:
        teams = supported_teams()
        keep = supported_fixture_mask(raw[home_col], raw[away_col], teams)
        print(f"  {path.name}: excluded {int((~keep).sum())} fixtures outside the {len(teams)}-club population "
              f"(rule: {SUPPORTED_FIXTURE_RULE})")
        raw = raw.loc[keep].copy()

    raw[date_col] = pd.to_datetime(raw[date_col])

    if NEUTRAL_VENUE_COL in raw.columns:
        neutral = raw[NEUTRAL_VENUE_COL].astype(str).str.lower().map({"true": True, "false": False})
        if neutral.isna().any():
            raise ValueError(f"Invalid neutral-venue flags in {path}")
        is_home_flag = ~neutral
    else:
        is_home_flag = pd.Series(True, index=raw.index)

    home = pd.DataFrame({
        "match_date": raw[date_col],
        "team": raw[home_col].map(standardise_team_name),
        "is_home": is_home_flag.to_numpy(),
    })
    away = pd.DataFrame({
        "match_date": raw[date_col],
        "team": raw[away_col].map(standardise_team_name),
        "is_home": False,
    })

    if season_col is not None:
        home["season"] = raw[season_col].to_numpy()
        away["season"] = raw[season_col].to_numpy()

    events = pd.concat([home, away], ignore_index=True)
    events["source"] = path.name
    competition = choose_column(raw.columns, ["competition"])
    if competition is not None:
        events["competition"] = np.concatenate([raw[competition].to_numpy()] * 2)
    else:
        events["competition"] = "League"
    return events.dropna(subset=["match_date", "team"]).copy()

def load_events():
    for label, start, end in [("match", MATCH_WINDOW_START_DAYS, MATCH_WINDOW_END_DAYS),
                              ("home", HOME_WINDOW_START_DAYS, HOME_WINDOW_END_DAYS)]:
        if start > end:
            raise ValueError(f"{label} event-window start must not exceed its end")
    parts = []
    for path in EVENT_PATHS:
        part = load_events_file(path)
        print(f"  {path.name}: {len(part)} team-events "
              f"({part['match_date'].min().date()}..{part['match_date'].max().date()})")
        parts.append(part)

    events = pd.concat(parts, ignore_index=True)

    # A fixture listed in two files must not be counted twice.
    before = len(events)
    events = events.drop_duplicates(subset=["match_date", "team", "is_home"]).copy()
    if before != len(events):
        print(f"  dropped {before - len(events)} duplicate team-events across files")

    # Nullable integer avoids the pandas dtype warning from the first version.
    if "season" in events.columns:
        season_key = events["season"].map(normalize_season_key).astype("Int64")
    else:
        season_key = pd.Series(pd.array([pd.NA] * len(events), dtype="Int64"), index=events.index)

    derived = events["match_date"].map(season_start_year_from_date).astype("Int64")
    events["season_key"] = season_key.fillna(derived).astype(int)

    events["t_m_start"] = events["match_date"] + pd.to_timedelta(MATCH_WINDOW_START_DAYS, unit="D")
    events["t_m_end"] = events["match_date"] + pd.to_timedelta(MATCH_WINDOW_END_DAYS, unit="D")
    events["t_h_start"] = events["match_date"] + pd.to_timedelta(HOME_WINDOW_START_DAYS, unit="D")
    events["t_h_end"] = events["match_date"] + pd.to_timedelta(HOME_WINDOW_END_DAYS, unit="D")

    return events.reset_index(drop=True)

def load_supporter_weights(teams):
    """
    Club allocations S_e, summing to the assumed audience pool over `teams`.
    Explicit relative_mean_one mode reproduces the previous weighting units.
    Returns None when disabled, absent or incomplete. The caller must decide
    whether that is allowed; the home/TV entry point rejects requested weighting
    when this helper cannot supply it.
    """
    if SUPPORTER_WEIGHT_COLUMN is None:
        return None
    if not SUPPORTER_PATH.exists():
        warnings.warn(f"{SUPPORTER_PATH} not found; no supporter weights returned.")
        return None

    table = pd.read_csv(SUPPORTER_PATH)
    table["team_std"] = table["club"].map(standardise_team_name)
    if table.loc[table.team_std.isin(teams), 'team_std'].duplicated().any():
        raise ValueError('Duplicate club rows in national supporter table')
    values = pd.to_numeric(table[SUPPORTER_WEIGHT_COLUMN], errors="coerce")

    if SUPPORTER_IMPUTE_FROM is not None:
        other = pd.to_numeric(table[SUPPORTER_IMPUTE_FROM], errors="coerce")
        both = values.notna() & other.notna()
        if both.sum() >= 3:
            slope, intercept = np.polyfit(other[both].to_numpy(), values[both].to_numpy(), 1)
            print(f"Supporter imputation: {SUPPORTER_WEIGHT_COLUMN} = "
                  f"{intercept:.4f} + {slope:.4f}*{SUPPORTER_IMPUTE_FROM} "
                  f"(fitted on {int(both.sum())} clubs)")
            values = values.where(values.notna(), intercept + slope * other)

    lookup = dict(zip(table["team_std"], values))
    missing = [t for t in teams if not np.isfinite(lookup.get(t, np.nan))]
    if missing:
        warnings.warn(
            f"No supporter weight for {missing}; no supporter weights returned. "
            "The home/TV entry point stops if weighting was requested."
        )
        return None

    raw = {t: float(lookup[t]) for t in teams}
    return audience_allocations(raw, TV_SUPPORTER_SCALE, TV_AUDIENCE_TOTAL)

def supporter_scaling_metadata(weights, enabled=True):
    return audience_metadata(weights, TV_SUPPORTER_SCALE, TV_AUDIENCE_TOTAL, enabled)

def apply_supporter_weights(s_lookup, weights):
    """Multiply the unchanged unit S map by its allocated S_e."""
    if weights is None:
        return s_lookup
    print(f"TV club allocations ({TV_SUPPORTER_SCALE}; total {sum(weights.values()):,.6g}):")
    for team in sorted(weights, key=lambda t: -weights[t]):
        print(f"  {team:28s} {weights[team]:6.3f}")
    return {t: v * weights[t] for t, v in s_lookup.items()}

def delta_basis(B_s_by_date, unique_dates):
    """
    Basis for a date effect delta_t that is orthogonal to the s_x(t) spline basis
    and to the seven weekday indicators, so it carries only the NON-smooth,
    non-weekday part of each date. Without that constraint delta_t would be
    collinear with the per-LTLA splines and the optimizer would wander a flat ridge.
    """
    n_dates = len(unique_dates)
    weekday = np.zeros((n_dates, 7), dtype=float)
    weekday[np.arange(n_dates), pd.DatetimeIndex(unique_dates).dayofweek.to_numpy()] = 1.0
    return null_space(np.hstack([B_s_by_date, weekday]).T)

class ModelData:
    def __init__(self, df, lads, unique_dates, B_s, B_omega=None, A_scaled=None,
                 H_scaled=None, A_scale=1.0, H_scale=1.0, Q_delta=None):
        self.df = df
        self.lads = lads
        self.unique_dates = unique_dates
        self.n_lad = len(lads)
        self.n = len(df)

        self.y = df[INFECTIONS_COL].to_numpy(dtype=float)
        self.bg = df[BACKGROUND_COL].to_numpy(dtype=float)
        self.lad_idx = df["lad_idx"].to_numpy(dtype=int)
        self.dow = df["weekday"].to_numpy(dtype=int)
        self.date_idx = df["date_idx"].to_numpy(dtype=int)

        self.B_s_by_date = B_s
        self.B_s = B_s[self.date_idx]
        self.Ks = self.B_s.shape[1]

        self.B_omega_by_date = B_omega
        if B_omega is not None:
            self.B_omega = B_omega[self.date_idx]
            self.Ko = self.B_omega.shape[1]
        else:
            self.B_omega = None
            self.Ko = 0

        self.A_scaled = np.zeros(self.n) if A_scaled is None else np.asarray(A_scaled, dtype=float)
        self.H_scaled = np.zeros(self.n) if H_scaled is None else np.asarray(H_scaled, dtype=float)

        # Rescaling factors that turn the internal coefficients back into the
        # reported C_m and D_s; needed to map (log C_m, log r_s) -> (cA, cH).
        self.A_scale = float(A_scale)
        self.H_scale = float(H_scale)

        # Date-effect basis (Model C only); delta_log is an offset on log tau.
        self.Q_delta = Q_delta
        self.q_delta = 0 if Q_delta is None else int(Q_delta.shape[1])
        self.gamma = None
        self.delta_log = np.zeros(self.n, dtype=float)

        self.rows_by_lad = [np.flatnonzero(self.lad_idx == x) for x in range(self.n_lad)]

    def set_delta(self, gamma):
        if self.Q_delta is None or gamma is None:
            self.gamma = None
            self.delta_log = np.zeros(self.n, dtype=float)
        else:
            self.gamma = np.asarray(gamma, dtype=float)
            self.delta_log = (self.Q_delta @ self.gamma)[self.date_idx]
        return self.delta_log

def initial_A(data: ModelData):
    residual = np.maximum(data.y - data.bg, 0.05)
    beta = np.zeros((data.n_lad, data.Ks), dtype=float)
    weekday_free = np.zeros((data.n_lad, 6), dtype=float)
    for x, rows in enumerate(data.rows_by_lad):
        mean_t = max(float(np.mean(residual[rows])), 0.05)
        beta[x, :] = np.log(mean_t)
    return beta, weekday_free

def evaluate_A_arrays(data: ModelData, beta, weekday_free, cA=0.0, cH=0.0):
    week_full = constrained_weekday_effects(weekday_free)
    smooth_log = np.sum(data.B_s * beta[data.lad_idx], axis=1)
    weekday_log = week_full[data.lad_idx, data.dow]
    eta = smooth_log + weekday_log + data.delta_log
    tau = np.exp(np.clip(eta, *ETA_BOUNDS))
    F = 1.0 + cA * data.A_scaled + cH * data.H_scaled
    mu = np.maximum(data.bg + tau * F, EPS)
    return {
        "beta": beta,
        "weekday_free": weekday_free,
        "weekday_full": week_full,
        "smooth_log": smooth_log,
        "weekday_log": weekday_log,
        "delta_log": data.delta_log.copy(),
        "gamma": None if data.gamma is None else data.gamma.copy(),
        "tau": tau,
        "football_multiplier": F,
        "mu": mu,
        "cA": float(cA),
        "cH": float(cH),
    }

def penalized_objective_A(data, beta, weekday_free, cA=0.0, cH=0.0):
    ev = evaluate_A_arrays(data, beta, weekday_free, cA, cH)
    nll = float(np.sum(ev["mu"] - data.y * np.log(ev["mu"])))
    p, _ = roughness_2d(beta, S_SMOOTH_PENALTY)
    return nll + p

def audit_tables(s_path, weights, metadata):
    table = pd.read_csv(s_path)
    required = {'team', 'lad_code', 'total_population', 'S_x'}
    if not required.issubset(table):
        raise ValueError('Audience audit needs the source S map and its population')
    table['team'] = table.team.map(canonical_team)
    if table.duplicated(['team', 'lad_code']).any() or set(table.team) != set(weights):
        raise ValueError('S map clubs/rows disagree with the audience allocations')
    values = table[['S_x', 'total_population']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any() or (values[:, 1] <= 0).any():
        raise ValueError('Invalid S map or LAD population')
    if table.groupby('lad_code').total_population.nunique().gt(1).any():
        raise ValueError('Inconsistent LAD populations across S maps')
    table['unscaled_mass'] = table.S_x * table.total_population
    mass = table.groupby('team').unscaled_mass.sum()
    if not np.allclose(mass, 1., rtol=1e-9, atol=1e-12):
        raise ValueError('Source S must retain unit population-weighted mass per club')
    table['S_e'] = table.team.map(weights)
    table['scaled_S_x'] = table.S_x * table.S_e
    table['allocated_audience_units'] = table.scaled_S_x * table.total_population
    # These are diagnostics, not grounds for clipping or reallocating an assumed scale.
    table['exceeds_population_if_interpreted_as_people'] = table.scaled_S_x > 1.
    clubs = table.groupby('team').agg(S_e=('S_e', 'first'),
        population_weighted_scaled_S=('allocated_audience_units', 'sum'),
        max_allocation_per_resident=('scaled_S_x', 'max'),
        lads_above_population=('exceeds_population_if_interpreted_as_people', 'sum')).reset_index()
    clubs['supporter_fraction'] = clubs.team.map(metadata['supporter_fractions'])
    clubs['scale'] = metadata['tv_supporter_scale']
    clubs['assumed_audience'] = metadata['tv_audience_is_assumed']
    if not np.allclose(clubs.S_e, clubs.population_weighted_scaled_S, rtol=1e-9, atol=1e-8):
        raise ValueError('Audience allocations do not preserve S_e')
    lads = table.groupby('lad_code').agg(total_population=('total_population', 'first'),
        allocated_pool_units=('allocated_audience_units', 'sum')).reset_index()
    lads['allocation_per_resident'] = lads.allocated_pool_units/lads.total_population
    lads['exceeds_population_if_interpreted_as_people'] = lads.allocation_per_resident > 1.
    if 'lad_name' in table:
        names = table.drop_duplicates('lad_code').set_index('lad_code').lad_name
        lads['lad_name'] = lads.lad_code.map(names)
    total = float(clubs.S_e.sum())
    if metadata['tv_audience_is_assumed'] and not np.isclose(total, metadata['tv_audience_total'], rtol=1e-12):
        raise ValueError('Allocated audience does not match the assumed pool')
    report = dict(allocated_pool_total=total, clubs=len(clubs), lads=len(lads),
        club_lad_cells_above_population=int(table.exceeds_population_if_interpreted_as_people.sum()),
        lads_with_total_pool_above_population=int(lads.exceeds_population_if_interpreted_as_people.sum()),
        max_club_allocation_per_resident=float(table.scaled_S_x.max()),
        max_total_allocation_per_resident=float(lads.allocation_per_resident.max()),
        map_changed=False, population_capping=False, fitting_performed=False,
        scope='Complete source England S maps, before fixture/time/LAD selection',
        interpretation='Assumed audience units, not measured or unique viewers. Above-population allocations are reported without clipping.',
        source_S_sha256=hashlib.sha256(Path(s_path).read_bytes()).hexdigest(),
        scaling=metadata)
    return clubs, table, lads, report


def write_audit(output, s_path, weights, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clubs, cells, lads, report = audit_tables(s_path, weights, metadata)
    clubs.to_csv(output/'T_home_tv_club_allocations.csv', index=False)
    columns = ['team', 'lad_code', 'total_population', 'S_x', 'S_e', 'scaled_S_x',
               'allocated_audience_units', 'exceeds_population_if_interpreted_as_people']
    cells[columns].to_csv(output/'T_home_tv_audience_club_lad_audit.csv', index=False)
    lads.to_csv(output/'T_home_tv_audience_lad_audit.csv', index=False)
    (output/'T_home_tv_audience_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report

class NuisanceProjector:
    """Orthogonal projection without allocating a panel x all-parameters matrix.

    Full date indicators span the same additional space as constrained Q_delta
    once location splines/weekdays are included. A Schur complement handles
    their overlap with the local blocks; a pseudoinverse removes redundancy.
    """
    def __init__(self, data, weight):
        self.data, self.weight = data, np.asarray(weight)
        self.blocks = []
        self.rank = 0
        for rows in data.rows_by_lad:
            weekday = np.eye(7)[data.dow[rows], :6] - (data.dow[rows] == 6)[:, None]
            z = np.column_stack([data.B_s[rows], weekday]) * self.weight[rows, None]
            u, singular, _ = np.linalg.svd(z, full_matrices=False)
            rank = int(np.sum(singular > max(singular.max(initial=0)*1e-10, 1e-14)))
            self.blocks.append((rows, u[:, :rank]))
            self.rank += rank
        self.date_inverse = None
        if data.Q_delta is not None and data.q_delta:
            nd = len(data.unique_dates)
            gram = np.diag(np.bincount(data.date_idx, weights=self.weight**2, minlength=nd))
            for rows, q in self.blocks:
                cross = np.zeros((q.shape[1], nd))
                for j in range(q.shape[1]):
                    cross[j] = np.bincount(data.date_idx[rows], weights=q[:, j]*self.weight[rows], minlength=nd)
                gram -= cross.T @ cross
            eigen, vectors = np.linalg.eigh((gram+gram.T)/2)
            keep = eigen > max(eigen.max(initial=0)*1e-9, 1e-12)
            self.date_inverse = (vectors[:, keep]/eigen[keep]) @ vectors[:, keep].T
            self.rank += int(keep.sum())

    def local_residual(self, value):
        value = np.asarray(value, dtype=float)
        result = value.copy()
        for rows, q in self.blocks:
            result[rows] -= q @ (q.T @ value[rows])
        return result

    def residual(self, value):
        was_vector = np.ndim(value) == 1
        value = np.asarray(value, dtype=float).reshape(self.data.n, -1)
        result = self.local_residual(value)
        if self.date_inverse is not None:
            rhs = np.column_stack([np.bincount(self.data.date_idx, weights=self.weight*result[:, j],
                                               minlength=len(self.data.unique_dates)) for j in range(result.shape[1])])
            gamma = self.date_inverse @ rhs
            result -= self.local_residual(self.weight[:, None]*gamma[self.data.date_idx])
        return result[:, 0] if was_vector else result
