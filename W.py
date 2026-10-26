"""Compute the paper's regional_non_big_six W from supplied public inputs.

Original distance calibration, peer allocation and attendance scaling are retained.
Sources: W/fit_catchment_model.py and W/catchment_robustness.py.
"""
from __future__ import annotations
import argparse
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

from pathlib import Path

import math

import re

import warnings

import sys

import numpy as np

import pandas as pd

from scipy.optimize import minimize_scalar

DATA_DIR = DATA

SURVEY_DIR = DATA_DIR

OUTPUT_DIR = RESULTS / "W"

LAD_CENTROIDS_PATH = DATA_DIR / "LAD_April2020_Population_Weighted_Centroids.csv"

GDP_PATH = DATA_DIR / "regionalgrossdomesticproductlocalauthorities.xlsx"

SURVEY_2005_PATH = SURVEY_DIR / "survey_2005_06_club_targets.csv"

SURVEY_2007_PATH = SURVEY_DIR / "survey_2007_08_mean_distance.csv"

SUPPLEMENTARY_DISTANCE_PATH = SURVEY_DIR / "supplementary_direct_distance_targets.csv"

SOCIOECONOMIC_PROFILE_PATH = SURVEY_DIR / "survey_socioeconomic_profile.csv"

GDP_YEARS = [2005, 2007, 2019]

FINAL_GDP_YEAR = 2019

DISTANCE_BAND_WEIGHT = 1.0

MEAN_DISTANCE_WEIGHT = 1.0

ALPHA_LOWER = 0.05

ALPHA_UPPER = 4.0

GAMMA_USER_UPPER = 2.0

MAX_ITER = 20

CONVERGENCE_TOL = 1e-5

MIN_DISTANCE_KM = 0.5

FINAL_TEAMS = [
    "Arsenal",
    "Aston Villa",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

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
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle United",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Wolves": "Wolverhampton Wanderers",
}

STADIUMS_2005 = {
    "Arsenal": (-0.1028, 51.5576),  # Highbury
    "Aston Villa": (-1.8847, 52.5092),
    "Birmingham City": (-1.8682, 52.4757),
    "Blackburn Rovers": (-2.4892, 53.7286),
    "Bolton Wanderers": (-2.5350, 53.5805),
    "Charlton Athletic": (0.0367, 51.4865),
    "Chelsea": (-0.1910, 51.4817),
    "Everton": (-2.9663, 53.4388),
    "Fulham": (-0.2217, 51.4749),
    "Liverpool": (-2.9608, 53.4308),
    "Manchester City": (-2.2004, 53.4831),
    "Manchester United": (-2.2913, 53.4631),
    "Middlesbrough": (-1.2178, 54.5781),
    "Newcastle United": (-1.6224, 54.9752),
    "Portsmouth": (-1.0638, 50.7964),
    "Sunderland": (-1.3882, 54.9146),
    "Tottenham Hotspur": (-0.0657, 51.6033),  # White Hart Lane
    "West Bromwich Albion": (-1.9639, 52.5090),
    "West Ham United": (0.0383, 51.5317),  # Upton Park
    "Wigan Athletic": (-2.6534, 53.5476),
}

STADIUMS_2007 = {
    "Arsenal": (-0.1084, 51.5549),  # Emirates
    "Aston Villa": (-1.8847, 52.5092),
    "Birmingham City": (-1.8682, 52.4757),
    "Blackburn Rovers": (-2.4892, 53.7286),
    "Bolton Wanderers": (-2.5350, 53.5805),
    "Chelsea": (-0.1910, 51.4817),
    "Derby County": (-1.4470, 52.9150),
    "Everton": (-2.9663, 53.4388),
    "Fulham": (-0.2217, 51.4749),
    "Liverpool": (-2.9608, 53.4308),
    "Manchester City": (-2.2004, 53.4831),
    "Manchester United": (-2.2913, 53.4631),
    "Middlesbrough": (-1.2178, 54.5781),
    "Newcastle United": (-1.6224, 54.9752),
    "Portsmouth": (-1.0638, 50.7964),
    "Reading": (-0.9827, 51.4222),
    "Sunderland": (-1.3882, 54.9146),
    "Tottenham Hotspur": (-0.0657, 51.6033),
    "West Ham United": (0.0383, 51.5317),
    "Wigan Athletic": (-2.6534, 53.5476),
}

FINAL_STADIUMS = {
    "Arsenal": (-0.1084, 51.5549),
    "Aston Villa": (-1.8847, 52.5092),
    "Brentford": (-0.2887, 51.4908),
    "Brighton": (-0.0837, 50.8616),
    "Chelsea": (-0.1910, 51.4817),
    "Crystal Palace": (-0.0856, 51.3983),
    "Everton": (-2.9663, 53.4388),  # Goodison Park for the study period
    "Liverpool": (-2.9608, 53.4308),
    "Manchester City": (-2.2004, 53.4831),
    "Manchester United": (-2.2913, 53.4631),
    "Newcastle United": (-1.6224, 54.9752),
    "Tottenham Hotspur": (-0.0665, 51.6043),
    "West Ham United": (-0.0165, 51.5386),  # London Stadium
    "Wolverhampton Wanderers": (-2.1306, 52.5902),
}

PEER_GROUPS = {
    "Brentford": [
        "Arsenal", "Chelsea", "Fulham", "Tottenham Hotspur",
        "West Ham United", "Charlton Athletic",
    ],
    "Crystal Palace": [
        "Arsenal", "Chelsea", "Fulham", "Tottenham Hotspur",
        "West Ham United", "Charlton Athletic",
    ],
    "Wolverhampton Wanderers": [
        "Aston Villa", "Birmingham City", "West Bromwich Albion",
    ],
}

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

study_teams = teams

study_stadiums = stadiums

FINAL_TEAMS = study_teams()

FINAL_STADIUMS = study_stadiums()

PEER_IMPUTATION_ENABLED = True

PEER_EXCLUDED_CLUBS = set()

def standardise_team_name(name):
    return canonical_team(name)

PEER_GROUPS.update({
    'Bournemouth': ['Portsmouth', 'Reading', 'Brighton'],
    'Burnley': ['Blackburn Rovers', 'Bolton Wanderers', 'Wigan Athletic'],
    'Nottingham Forest': ['Derby County', 'Birmingham City', 'West Bromwich Albion'],
    'Sheffield United': ['Derby County', 'Middlesbrough', 'Blackburn Rovers'],
    'Leeds': ['Middlesbrough', 'Sunderland', 'Blackburn Rovers'],
})

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Arrays are allowed for lat1/lon1."""
    r = 6371.0088
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return r * 2.0 * np.arcsin(np.sqrt(a))

def distance_to_stadium(lads: pd.DataFrame, coords) -> np.ndarray:
    lon, lat = coords
    d = haversine_km(
        lads["latitude"].to_numpy(),
        lads["longitude"].to_numpy(),
        lat,
        lon,
    )
    return np.maximum(d, MIN_DISTANCE_KM)

def load_lads() -> pd.DataFrame:
    if not LAD_CENTROIDS_PATH.exists():
        raise FileNotFoundError(f"Missing LAD centroid file: {LAD_CENTROIDS_PATH}")

    lads = pd.read_csv(LAD_CENTROIDS_PATH)
    required = ["lad_code", "lad_name", "total_population", "longitude", "latitude"]
    missing = [c for c in required if c not in lads.columns]
    if missing:
        raise ValueError(f"Missing columns in LAD centroid file: {missing}")

    lads = lads[required].copy()
    lads["total_population"] = pd.to_numeric(lads["total_population"], errors="coerce")
    lads["longitude"] = pd.to_numeric(lads["longitude"], errors="coerce")
    lads["latitude"] = pd.to_numeric(lads["latitude"], errors="coerce")
    lads = lads.dropna(subset=["total_population", "longitude", "latitude"])
    lads = lads[lads["lad_code"].astype(str).str.startswith("E")].copy()
    return lads

def _find_year_column(df: pd.DataFrame, year: int):
    for col in df.columns:
        if isinstance(col, (int, float)) and not pd.isna(col) and int(col) == int(year):
            return col
        text = str(col).strip()
        if text == str(year) or text.startswith(f"{year}\n") or text.startswith(f"{year} "):
            return col
    raise KeyError(f"Could not find GDP year {year} in Table 7.")

SUCCESSOR_BY_OLD_LAD_NAME = {
    "Aylesbury Vale": "Buckinghamshire",
    "Chiltern": "Buckinghamshire",
    "South Bucks": "Buckinghamshire",
    "Wycombe": "Buckinghamshire",
    "Corby": "North Northamptonshire",
    "East Northamptonshire": "North Northamptonshire",
    "Kettering": "North Northamptonshire",
    "Wellingborough": "North Northamptonshire",
    "Daventry": "West Northamptonshire",
    "Northampton": "West Northamptonshire",
    "South Northamptonshire": "West Northamptonshire",
}

def load_gdp_for_lads(lads: pd.DataFrame) -> pd.DataFrame:
    if not GDP_PATH.exists():
        raise FileNotFoundError(f"Missing GDP workbook: {GDP_PATH}")

    raw = pd.read_excel(GDP_PATH, sheet_name="Table 7", header=1)
    year_cols = {year: _find_year_column(raw, year) for year in GDP_YEARS}

    keep = ["ITL1 Region", "LA code", "LA name"] + list(year_cols.values())
    gdp = raw[keep].copy()
    rename = {
        "ITL1 Region": "region",
        "LA code": "gdp_lad_code",
        "LA name": "gdp_lad_name",
    }
    rename.update({col: f"gdp_{year}" for year, col in year_cols.items()})
    gdp = gdp.rename(columns=rename)

    for year in GDP_YEARS:
        gdp[f"gdp_{year}"] = pd.to_numeric(gdp[f"gdp_{year}"], errors="coerce")

    # First: exact code match.
    out = lads.merge(
        gdp,
        left_on="lad_code",
        right_on="gdp_lad_code",
        how="left",
    )

    # Second: map pre-2020/21 district names to successor authorities where needed.
    name_lookup = gdp.dropna(subset=["gdp_lad_name"]).drop_duplicates("gdp_lad_name").set_index("gdp_lad_name")
    for idx in out.index[out["region"].isna()]:
        old_name = out.at[idx, "lad_name"]
        successor = SUCCESSOR_BY_OLD_LAD_NAME.get(old_name)
        if successor and successor in name_lookup.index:
            row = name_lookup.loc[successor]
            out.at[idx, "region"] = row["region"]
            out.at[idx, "gdp_lad_name"] = successor
            for year in GDP_YEARS:
                out.at[idx, f"gdp_{year}"] = row[f"gdp_{year}"]

    # Third: exact LAD-name fallback.
    for idx in out.index[out["region"].isna()]:
        name = out.at[idx, "lad_name"]
        if name in name_lookup.index:
            row = name_lookup.loc[name]
            out.at[idx, "region"] = row["region"]
            out.at[idx, "gdp_lad_name"] = name
            for year in GDP_YEARS:
                out.at[idx, f"gdp_{year}"] = row[f"gdp_{year}"]

    # We need complete GDP coverage because gamma is part of the weight.
    missing_rows = out[[f"gdp_{y}" for y in GDP_YEARS]].isna().any(axis=1)
    if missing_rows.any():
        bad = out.loc[missing_rows, ["lad_code", "lad_name"] + [f"gdp_{y}" for y in GDP_YEARS]]
        raise ValueError(
            "Some LADs could not be assigned GDP for all calibration years.\n"
            + bad.to_string(index=False)
        )

    # Normalize GDP before applying 1 + gamma*g.
    # g_x = GDP_x / population-weighted mean GDP - 1.
    pop = out["total_population"].to_numpy(dtype=float)
    for year in GDP_YEARS:
        values = out[f"gdp_{year}"].to_numpy(dtype=float)
        weighted_mean = np.average(values, weights=pop)
        out[f"g_{year}"] = values / weighted_mean - 1.0

    return out

def load_survey_targets():
    for p in [SURVEY_2005_PATH, SURVEY_2007_PATH, SUPPLEMENTARY_DISTANCE_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing survey input file: {p}")

    survey05 = pd.read_csv(SURVEY_2005_PATH)
    survey07 = pd.read_csv(SURVEY_2007_PATH)
    extra = pd.read_csv(SUPPLEMENTARY_DISTANCE_PATH)

    for df in [survey05, survey07, extra]:
        df["team"] = df["team"].map(standardise_team_name)

    # Convert percentages to fractions for calculations.
    for col in ["share_lt_10_miles_pct", "share_10_49_miles_pct", "share_50plus_miles_pct"]:
        survey05[col.replace("_pct", "")] = pd.to_numeric(survey05[col], errors="coerce") / 100.0

    survey05["avg_income_gbp"] = pd.to_numeric(
        survey05["avg_income_gbp_thousands"], errors="coerce"
    ) * 1000.0
    survey07["mean_distance_miles"] = pd.to_numeric(survey07["mean_distance_miles"], errors="coerce")
    extra["mean_distance_miles"] = pd.to_numeric(extra["mean_distance_miles"], errors="coerce")

    return survey05, survey07, extra

def raw_weight(g: np.ndarray, distance_km: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    economic_term = 1.0 + gamma * g
    if np.any(economic_term <= 0):
        raise ValueError("1 + gamma*g must remain strictly positive for all LADs.")
    return economic_term / np.power(distance_km, alpha)

def catchment_share(
    lads: pd.DataFrame,
    distance_km: np.ndarray,
    alpha: float,
    gamma: float,
    g_col: str,
) -> np.ndarray:
    w = raw_weight(lads[g_col].to_numpy(dtype=float), distance_km, alpha, gamma)
    mass = lads["total_population"].to_numpy(dtype=float) * w
    return mass / mass.sum()

def implied_mean_distance_miles(q: np.ndarray, distance_km: np.ndarray) -> float:
    return float(np.sum(q * distance_km) / 1.609344)

def implied_distance_bands(q: np.ndarray, distance_km: np.ndarray):
    d_miles = distance_km / 1.609344
    return {
        "share_lt_10_miles": float(q[d_miles < 10.0].sum()),
        "share_10_49_miles": float(q[(d_miles >= 10.0) & (d_miles < 50.0)].sum()),
        "share_50plus_miles": float(q[d_miles >= 50.0].sum()),
    }

def fit_alpha_for_team(
    team: str,
    gamma: float,
    lads: pd.DataFrame,
    survey05: pd.DataFrame,
    survey07: pd.DataFrame,
    extra: pd.DataFrame,
):
    row05 = survey05.loc[survey05["team"] == team]
    row07 = survey07.loc[survey07["team"] == team]
    row_extra = extra.loc[extra["team"] == team]

    has05 = (not row05.empty) and team in STADIUMS_2005
    has07 = (not row07.empty) and team in STADIUMS_2007
    has_extra = (not row_extra.empty) and team in FINAL_STADIUMS

    if not (has05 or has07 or has_extra):
        return None

    d05 = distance_to_stadium(lads, STADIUMS_2005[team]) if has05 else None
    d07 = distance_to_stadium(lads, STADIUMS_2007[team]) if has07 else None
    dextra = distance_to_stadium(lads, FINAL_STADIUMS[team]) if has_extra else None

    def objective(alpha: float) -> float:
        loss = 0.0

        if has05:
            target = row05.iloc[0]
            q = catchment_share(lads, d05, alpha, gamma, "g_2005")
            pred = implied_distance_bands(q, d05)
            band_sse = (
                (pred["share_lt_10_miles"] - target["share_lt_10_miles"]) ** 2
                + (pred["share_10_49_miles"] - target["share_10_49_miles"]) ** 2
                + (pred["share_50plus_miles"] - target["share_50plus_miles"]) ** 2
            )
            loss += DISTANCE_BAND_WEIGHT * band_sse

        if has07:
            target_miles = float(row07.iloc[0]["mean_distance_miles"])
            q = catchment_share(lads, d07, alpha, gamma, "g_2007")
            pred_miles = implied_mean_distance_miles(q, d07)
            loss += MEAN_DISTANCE_WEIGHT * ((pred_miles - target_miles) / target_miles) ** 2

        if has_extra:
            # Supplementary modern direct-distance target (currently Brighton).
            target_miles = float(row_extra.iloc[0]["mean_distance_miles"])
            q = catchment_share(lads, dextra, alpha, gamma, "g_2019")
            pred_miles = implied_mean_distance_miles(q, dextra)
            loss += MEAN_DISTANCE_WEIGHT * ((pred_miles - target_miles) / target_miles) ** 2

        return float(loss)

    res = minimize_scalar(
        objective,
        bounds=(ALPHA_LOWER, ALPHA_UPPER),
        method="bounded",
        options={"xatol": 1e-7},
    )

    if not res.success or not np.isfinite(res.fun) or not np.isfinite(res.x):
        raise RuntimeError(f'Distance calibration failed for {team}: {res.message}')

    alpha = float(res.x)

    details = {
        "team": team,
        "alpha": alpha,
        "distance_loss": float(res.fun),
        "uses_2005_bands": has05,
        "uses_2007_mean": has07,
        "uses_supplementary_mean": has_extra,
    }

    if has05:
        target = row05.iloc[0]
        q = catchment_share(lads, d05, alpha, gamma, "g_2005")
        pred = implied_distance_bands(q, d05)
        details.update({
            "obs_2005_lt10": float(target["share_lt_10_miles"]),
            "pred_2005_lt10": pred["share_lt_10_miles"],
            "obs_2005_10_49": float(target["share_10_49_miles"]),
            "pred_2005_10_49": pred["share_10_49_miles"],
            "obs_2005_50plus": float(target["share_50plus_miles"]),
            "pred_2005_50plus": pred["share_50plus_miles"],
        })

    if has07:
        target_miles = float(row07.iloc[0]["mean_distance_miles"])
        q = catchment_share(lads, d07, alpha, gamma, "g_2007")
        details.update({
            "obs_2007_mean_miles": target_miles,
            "pred_2007_mean_miles": implied_mean_distance_miles(q, d07),
        })

    if has_extra:
        target_miles = float(row_extra.iloc[0]["mean_distance_miles"])
        q = catchment_share(lads, dextra, alpha, gamma, "g_2019")
        details.update({
            "obs_extra_mean_miles": target_miles,
            "pred_extra_mean_miles": implied_mean_distance_miles(q, dextra),
        })

    return alpha, details

def fit_all_direct_alphas(gamma, lads, survey05, survey07, extra):
    teams = sorted(set(survey05["team"]) | set(survey07["team"]) | set(extra["team"]))
    alphas = {}
    details = []

    for team in teams:
        fit = fit_alpha_for_team(team, gamma, lads, survey05, survey07, extra)
        if fit is None:
            continue
        alpha, detail = fit
        alphas[team] = alpha
        details.append(detail)

    return alphas, pd.DataFrame(details)

def season_from_filename(path: Path) -> str:
    m = re.search(r"teams_attendance_(.+)\.csv$", path.name)
    return m.group(1) if m else path.stem

BANDS = ['share_lt_10_miles', 'share_10_49_miles', 'share_50plus_miles']


def selective_peer_panels():
    """Explicit alternative donor eligibility; no supporter-share input is used."""
    original = {k: list(v) for k, v in PEER_GROUPS.items()}
    regional = {**original,
                'Brentford': ['Fulham', 'Charlton Athletic', 'West Ham United'],
                'Crystal Palace': ['Fulham', 'Charlton Athletic', 'West Ham United']}
    narrow = {**regional, 'Brentford': ['Fulham', 'Charlton Athletic'],
              'Crystal Palace': ['Fulham', 'Charlton Athletic']}
    wolves = {**original, 'Wolverhampton Wanderers': ['Birmingham City', 'West Bromwich Albion']}
    return {'regional_non_big_six': regional, 'london_fulham_charlton': narrow,
            'wolves_birmingham_west_brom': wolves}


def validate_inputs(lads, survey05, survey07, extra):
    if lads.lad_code.duplicated().any():
        raise ValueError('Duplicate LAD codes')
    vals = lads[['total_population', 'longitude', 'latitude']].to_numpy(float)
    if not np.isfinite(vals).all() or (lads.total_population <= 0).any():
        raise ValueError('Invalid LAD population or coordinates')
    if not lads.longitude.between(-180, 180).all() or not lads.latitude.between(-90, 90).all():
        raise ValueError('Coordinates outside WGS84 bounds')
    for df in (survey05, survey07, extra):
        if df.team.duplicated().any():
            raise ValueError('Duplicate survey team')
    bands = survey05[BANDS].to_numpy(float)
    if not np.isfinite(bands).all() or (bands < 0).any() or (bands > 1).any():
        raise ValueError('Invalid survey band shares')
    if not np.allclose(bands.sum(axis=1), 1, atol=1e-8):
        raise ValueError('Survey band shares must sum to one; review rounding explicitly')
    for df in (survey07, extra):
        values = df.mean_distance_miles.to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Invalid mean distance target')
    for year in GDP_YEARS:
        if not np.isfinite(lads[f'gdp_{year}']).all() or (lads[f'gdp_{year}'] <= 0).any():
            raise ValueError('GDP must be positive and finite')
    if not np.isfinite(survey05.avg_income_gbp).all() or (survey05.avg_income_gbp <= 0).any():
        raise ValueError('Survey income must be positive and finite')


def parse_count(value):
    """Explicit input contract: integer counts or dot/comma thousands groups."""
    value = str(value).strip()
    if re.fullmatch(r'\d+', value):
        return int(value)
    if re.fullmatch(r'\d{1,3}(?:\.\d{3})+', value):
        return int(value.replace('.', ''))
    if re.fullmatch(r'\d{1,3}(?:,\d{3})+', value):
        return int(value.replace(',', ''))
    raise ValueError(f'Ambiguous attendance count: {value!r}')


def load_attendance_audit(data_dir, home_matches):
    rows = []
    for path in sorted(data_dir.glob('teams_attendance_*.csv')):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            team = standardise_team_name(row['Club'])
            if team not in FINAL_TEAMS:
                continue
            average = parse_count(row['Average'])
            total = parse_count(row['Spectators'])
            implied = total / average if average else np.nan
            rows.append(dict(season=season_from_filename(path), team=team,
                             reported_average=average, total_spectators=total,
                             full_season_home_matches=home_matches,
                             full_season_average=total / home_matches,
                             implied_matches_in_reported_average=implied,
                             reported_average_excludes_matches=bool(average and implied < home_matches - .1),
                             capacity=parse_count(row['Capacity']), source_file=path.name))
    out = pd.DataFrame(rows)
    if out.empty or out.duplicated(['season', 'team']).any():
        raise ValueError('Missing attendance files or duplicate team-season records')
    if (out.total_spectators > out.reported_average * home_matches + home_matches).any():
        raise ValueError('Attendance totals incompatible with home-match count')
    return out


def final_parameters(direct, peer_rule='median'):
    rows = []
    for team in FINAL_TEAMS:
        if team in direct:
            rows.append(dict(team=team, alpha=direct[team], alpha_source='direct_distance', peers=''))
            continue
        if not PEER_IMPUTATION_ENABLED or team in PEER_EXCLUDED_CLUBS:
            continue
        peers = [p for p in PEER_GROUPS.get(team, []) if p in direct and p != team]
        if peer_rule.startswith('leave_out:'):
            excluded = peer_rule.split(':', 1)[1]
            peers = [p for p in peers if p != excluded]
        if not peers:
            raise ValueError(f'No available peers for {team}')
        values = [direct[p] for p in peers]
        reducer = {'min': np.min, 'max': np.max}.get(peer_rule, np.median)
        rows.append(dict(team=team, alpha=float(reducer(values)), alpha_source=f'peer_{peer_rule}',
                         peers='; '.join(peers)))
    return pd.DataFrame(rows,columns=['team','alpha','alpha_source','peers'])


def band_allocation(population, distance, target, within_band=None):
    """Exactly match three band masses, distributing mass by population or prior q."""
    population = np.asarray(population, float)
    distance = np.asarray(distance, float)
    target = np.asarray(target, float)
    if target.shape != (3,) or not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1):
        raise ValueError('Invalid band targets')
    prior = population if within_band is None else np.asarray(within_band, float)
    if prior.shape != population.shape or not np.isfinite(prior).all() or (prior < 0).any():
        raise ValueError('Invalid within-band weights')
    bands = np.digitize(distance / 1.609344, [10., 50.])
    result = np.zeros(len(population))
    for index, mass in enumerate(target):
        mask = bands == index
        denom = prior[mask].sum()
        if mass > 0 and denom <= 0:
            raise ValueError(f'Positive target for empty distance band {index}')
        if denom > 0:
            result[mask] = mass * prior[mask] / denom
    return result


def make_distribution(lads, params, gamma=0., targets=None, within='population'):
    rows = []
    for _, p in params.iterrows():
        d = distance_to_stadium(lads, FINAL_STADIUMS[p.team])
        q = catchment_share(lads, d, p.alpha, gamma, 'g_2019')
        method = 'power_law'
        if targets is not None and p.team in targets.index:
            q = band_allocation(lads.total_population, d, targets.loc[p.team, BANDS],
                                None if within == 'population' else q)
            method = f'2005_band_masses_{within}_within_band'
        elif targets is not None:
            method = 'baseline_fallback_no_2005_bands'
        if not np.isfinite(q).all() or (q < 0).any() or not np.isclose(q.sum(), 1):
            raise ValueError(f'Invalid distribution for {p.team}')
        frame = lads[['lad_code', 'lad_name', 'region', 'total_population', 'longitude', 'latitude']].copy()
        frame['team'] = p.team
        frame['q'] = q
        frame['distance_km'] = d
        frame['alpha'] = p.alpha
        frame['gamma'] = gamma
        frame['alpha_source'] = p.alpha_source
        frame['allocation_method'] = method
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_W(distributions, attendance):
    pieces = []
    for basis, column in [('reported_average', 'reported_average'), ('full_season_average', 'full_season_average')]:
        frame = distributions.merge(attendance[['season', 'team', column]], on='team', validate='many_to_many')
        frame = frame.rename(columns={column: 'avg_home_attendance'})
        frame['attendance_basis'] = basis
        frame['expected_attendees'] = frame.q * frame.avg_home_attendance
        frame['W'] = frame.expected_attendees / frame.total_population
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    keys = ['scenario', 'attendance_basis', 'season', 'team']
    check = result.groupby(keys).agg(modelled_attendance=('expected_attendees', 'sum'),
                                    avg_home_attendance=('avg_home_attendance', 'first'), n_lads=('lad_code', 'nunique')).reset_index()
    check['difference'] = check.modelled_attendance-check.avg_home_attendance
    if not np.allclose(check.modelled_attendance, check.avg_home_attendance, atol=1e-7):
        raise ValueError('Attendance normalisation failed')
    if result.duplicated(keys+['lad_code']).any():
        raise ValueError('Duplicate W key')
    return result, check


def main():
    global PEER_GROUPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=RESULTS/'W')
    args = parser.parse_args()
    lads = load_gdp_for_lads(load_lads())
    expected = pd.read_csv(LAD_CENTROIDS_PATH).lad_code
    expected = expected[expected.str.startswith('E')]
    if set(lads.lad_code) != set(expected) or len(lads) != len(expected):
        raise ValueError('LAD loader dropped or duplicated English LADs')
    s05, s07, extra = load_survey_targets()
    validate_inputs(lads, s05, s07, extra)
    attendance = load_attendance_audit(DATA, 19)
    direct, _ = fit_all_direct_alphas(0, lads, s05, s07, extra)
    PEER_GROUPS = selective_peer_panels()['regional_non_big_six']
    parameters = final_parameters(direct)
    distributions = make_distribution(lads, parameters, gamma=0.)
    distributions['scenario'] = 'regional_non_big_six'
    weights, checks = build_W(distributions, attendance)
    weights = weights[weights.attendance_basis.eq('reported_average')]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(args.output_dir/'regional_non_big_six_W_reported_average.csv', index=False)
    parameters.to_csv(args.output_dir/'parameters.csv', index=False)
    distributions.to_csv(args.output_dir/'lad_distributions.csv', index=False)
    checks[checks.attendance_basis.eq('reported_average')].to_csv(args.output_dir/'sanity_checks.csv', index=False)
    print(args.output_dir)


if __name__ == '__main__':
    main()
