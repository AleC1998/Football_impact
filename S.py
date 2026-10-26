"""Compute S from the supplied fan surveys and population geography."""
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

"""
LAD-level fan distribution model.

Implements the constrained spatial (gravity-type) allocation model described
in the write-up:

    S_{x,e} = Sigma_e * ( 1 / d_{x,s(e)}^beta_e + u_e )

subject to

    S_e(R) = sum_{x in R} S_{x,e} * P_x   (aggregated up to region R)

matching the known regional fan shares S_e(R) for team e. Sigma_e is the
normalizing constant that makes the total (England-wide) allocation sum to 1.

beta_e and u_e are inferred per team by minimizing the distance between the
model-implied regional shares and the real regional survey shares.

LONDON EXCLUSION RULE
----------------------
For non-London clubs, London is excluded from the regions used to fit
beta_e/u_e. London's fan share is driven by population density/migration
effects that a pure distance-decay model cannot explain for a club whose
home is hundreds of km away (e.g. Manchester City's genuine second peak in
London), so forcing the fit to also match it distorts beta for every other,
geographically-explicable region. This is effectively the same idea as a
London-specific uniform term, implemented instead by leaving London out of
the calibration target for those teams. London-based clubs still fit
against all 9 regions as before, since London *is* their home region.

The final LAD-level allocation (allocate_lads) is unaffected in structure:
every LAD, including London boroughs, still receives a distance-based
share from the fitted beta_e/u_e -- London is only excluded from the
*fitting* step, not from the output.

Usage:
    from S import run_team
    result = run_team("Aston Villa", "/path/to/dataset_astonvilla.csv")
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
GEOGRAPHY_DIR = DATA
REGIONAL_S_DIR = DATA
DATA_DIR = GEOGRAPHY_DIR
TEAM_DATA_DIR = REGIONAL_S_DIR

# April 2020 geography with mid-2020 LSOA population weights.
# Rebuild: Geography/build_geography.py. These are derived LAD centroids.
LAD_CENTROIDS_PATH = DATA_DIR / "LAD_April2020_Population_Weighted_Centroids.csv"
LAD_REGION_LOOKUP_PATH = DATA_DIR / "lad_region_lookup_April2020.csv"

# Stadium locations (longitude, latitude), WGS84.
# Clubs whose home region IS London -- these are fit against all 9 regions.
# Every other club excludes London from the fitting step (see module docstring).
TEAM_STADIUMS = stadiums()
LONDON_CLUBS = {r['team'] for r in clubs() if r['is_london_club'] == 'True'}

# Mapping from the raw region labels found in the team CSVs to the
# standardised English-region names used in the LAD lookup table.
REGION_NAME_MAP = {
    "North East England": "North East",
    "North West England": "North West",
    "Yorkshire and the Humber": "Yorkshire and The Humber",
    "East Midlands": "East Midlands",
    "West Midlands": "West Midlands",
    "East of England": "East of England",
    "London": "London",
    "South East England": "South East",
    "South West England and Gibraltar": "South West",
    "South West England": "South West",
    # non-England -> dropped
    "Scotland": None,
    "Wales": None,
    "Northern Ireland": None,
}

ENGLISH_REGIONS = [
    "North East", "North West", "Yorkshire and The Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East", "South West",
]

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lads():
    """Load April 2020 English LADs with mid-2020 populations."""
    lads = pd.read_csv(LAD_CENTROIDS_PATH)
    region_lookup = pd.read_csv(LAD_REGION_LOOKUP_PATH)
    if set(lads["lad_code"]) != set(region_lookup["lad_code"]):
        raise ValueError("Centroids and regions use different LAD code sets")
    lads = lads.merge(region_lookup, on="lad_code", how="left", validate="one_to_one")
    if lads["region"].isna().any():
        missing = lads.loc[lads["region"].isna(), ["lad_code", "lad_name"]]
        raise ValueError(f"LADs missing region mapping:\n{missing}")
    return lads


def load_team_regional_shares(csv_path):
    """
    Load a team's regional-share CSV (dw_name, VALUE, Names correct) and
    return a Series indexed by the 9 standard English regions, England-only
    shares renormalised to sum to 1.
    """
    df = pd.read_csv(csv_path)
    df["region_std"] = df["dw_name"].map(REGION_NAME_MAP)
    df = df.dropna(subset=["region_std", "VALUE"])
    shares = df.groupby("region_std")["VALUE"].sum()
    # keep English regions only, fill any absent region with 0
    shares = shares.reindex(ENGLISH_REGIONS, fill_value=0.0)
    total = shares.sum()
    if total <= 0:
        raise ValueError(f"No valid England share data found in {csv_path}")
    shares = shares / total  # renormalise to sum to 1 over England only
    return shares


def stadium_bng(team):
    lon, lat = TEAM_STADIUMS[team]
    x, y = _transformer.transform(lon, lat)
    return x, y


def fit_regions_for(team):
    """
    Regions used when FITTING beta/u for this team. London-based clubs fit
    against all 9 regions; every other club excludes London (see module
    docstring for rationale).
    """
    if team in LONDON_CLUBS:
        return list(ENGLISH_REGIONS)
    return [r for r in ENGLISH_REGIONS if r != "London"]


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def compute_distances(lads, team):
    sx, sy = stadium_bng(team)
    d_m = np.sqrt((lads["x_pwc"] - sx) ** 2 + (lads["y_pwc"] - sy) ** 2)
    d_km = d_m / 1000.0
    # floor to avoid singularities for the LAD containing (or very close to)
    # the stadium itself
    return np.maximum(d_km, 0.5)


def raw_score(d_km, beta, u):
    return 1.0 / np.power(d_km, beta) + u


def model_regional_shares(lads, d_km, beta, u):
    """
    Return the FULL model-implied share per English region (all 9 regions,
    summing to 1) for given beta, u. This is the model's complete output,
    used for reporting/diagnostics -- it always includes London, regardless
    of whether London was used to fit beta/u.
    """
    raw = raw_score(d_km, beta, u)
    weighted = raw * lads["total_population"].values
    by_region = pd.Series(weighted, index=lads.index).groupby(lads["region"]).sum()
    by_region = by_region.reindex(ENGLISH_REGIONS, fill_value=0.0)
    total = by_region.sum()
    return by_region / total


def fit_beta_u(lads, d_km, target_shares, fit_regions):
    """
    Fit (beta, u) by minimizing squared error between modeled and target
    regional shares, restricted to `fit_regions` (both renormalised to sum
    to 1 over just those regions -- e.g. the 8 non-London regions for a
    non-London club).
    """
    target_restricted = target_shares.reindex(fit_regions)
    target_restricted = (target_restricted / target_restricted.sum()).values

    region_to_i = {r: i for i, r in enumerate(fit_regions)}
    # LADs outside fit_regions (e.g. London LADs, for a non-London club)
    # still exist and still get raw scores, but are dropped from the
    # aggregation used to score the fit.
    mask = lads["region"].isin(fit_regions).values
    codes = lads.loc[mask, "region"].map(region_to_i).values.astype(int)
    P_masked = lads.loc[mask, "total_population"].values
    d_masked = np.asarray(d_km)[mask]
    n_regions = len(fit_regions)

    def objective(params):
        beta, u = params
        if beta <= 0 or u < 0:
            return 1e6
        raw = 1.0 / np.power(d_masked, beta) + u
        weighted = raw * P_masked
        by_region = np.bincount(codes, weights=weighted, minlength=n_regions)
        model = by_region / by_region.sum()
        return float(np.sum((model - target_restricted) ** 2))

    best = None
    for beta0 in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        res = minimize(
            objective, x0=[beta0, 1e-4],
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 3000},
        )
        if best is None or res.fun < best.fun:
            best = res
    beta_hat, u_hat = best.x
    return beta_hat, max(u_hat, 0.0), best.fun


def allocate_lads(lads, d_km, beta, u):
    """
    Return a copy of `lads` with columns:
      - distance_km
      - raw_score
      - S_x            (S_{x,e}, the per-capita share/density term, already
                         normalised so that sum_x S_x * P_x = 1)
      - fan_share       (S_x * P_x): fraction of the team's ENGLAND fanbase
                         estimated to live in that LAD (sums to 1 over
                         England, INCLUDING London LADs -- London is only
                         excluded from the fitting step, not the output)
      - fan_share_pct
    """
    raw = raw_score(d_km, beta, u)
    total = np.sum(raw * lads["total_population"].values)
    sigma_e = 1.0 / total
    S_x = sigma_e * raw

    out = lads.copy()
    out["distance_km"] = d_km
    out["raw_score"] = raw
    out["S_x"] = S_x
    out["fan_share"] = S_x * lads["total_population"].values
    out["fan_share_pct"] = out["fan_share"] * 100.0
    return out


def run_team(team, csv_path):
    """
    Full pipeline for one team: load data, fit beta/u (excluding London from
    the fit for non-London clubs), allocate to LADs.

    Returns dict with: lad_table (DataFrame), beta, u, fit_sse,
    fit_regions (regions actually used for the fit), target_shares (full
    9-region survey shares), modeled_shares (full 9-region model output).
    """
    if team not in TEAM_STADIUMS:
        raise ValueError(f"No stadium location registered for '{team}'.")

    lads = load_lads()
    d_km = compute_distances(lads, team)
    target_shares = load_team_regional_shares(csv_path)
    fit_regions = fit_regions_for(team)

    beta, u, fit_sse = fit_beta_u(lads, d_km, target_shares, fit_regions)
    modeled_shares = model_regional_shares(lads, d_km, beta, u)

    lad_table = allocate_lads(lads, d_km, beta, u)
    lad_table = lad_table.sort_values("fan_share", ascending=False).reset_index(drop=True)

    return {
        "team": team,
        "is_london_club": team in LONDON_CLUBS,
        "fit_regions": fit_regions,
        "beta": beta,
        "u": u,
        "fit_sse": fit_sse,
        "target_shares": target_shares,
        "modeled_shares": modeled_shares,
        "lad_table": lad_table,
    }


import argparse
import shutil
import json
import hashlib

SCRIPT_DIR = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=RESULTS/'S')
    parser.add_argument('--publish-model',action='store_true',help='Copy verified S to data/')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    long_rows, summaries=[],[]
    for row in clubs():
        print(f"Fitting {row['team']} (YouGov {row['survey_year']})",flush=True)
        result=run_team(row['team'], TEAM_DATA_DIR/row['regional_file'])
        table=result['lad_table'].copy()
        assert len(table)==314 and table.lad_code.is_unique
        assert np.isfinite(table[['S_x','fan_share']]).all().all()
        assert (table[['S_x','fan_share']]>=0).all().all()
        assert np.isclose(table.fan_share.sum(),1,atol=1e-12)
        table.insert(0,'team',row['team'])
        table['survey_year']=int(row['survey_year'])
        table['population_reference_date']='2020-06-30'
        table['geography_vintage']='April 2020'
        long_rows.append(table)
        summaries.append(dict(team=row['team'],survey_year=int(row['survey_year']),
                              is_london_club=result['is_london_club'],beta=result['beta'],u=result['u'],
                              fit_sse=result['fit_sse'],
                              **{f'target_{r}':result['target_shares'][r] for r in ENGLISH_REGIONS},
                              **{f'modeled_{r}':result['modeled_shares'][r] for r in ENGLISH_REGIONS}))
    combined=pd.concat(long_rows,ignore_index=True)
    assert len(combined)==20*314 and not combined.duplicated(['team','lad_code']).any()
    target=args.output_dir/'lad_level_fan_distribution.csv'
    combined.to_csv(target,index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir/'fitted_parameters_and_fit_quality.csv',index=False)
    sources=[DATA/'clubs.csv',*sorted(TEAM_DATA_DIR.glob('*.csv')),
             *sorted(DATA.glob('*.csv'))]
    (args.output_dir/'run_metadata.json').write_text(json.dumps(dict(clubs=20,LADs=314,
        geography='April 2020',population='mid-2020',
        inputs={str(p.relative_to(PROJECT_DIR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    if args.publish_model:
        shutil.copy2(target,DATA/'lad_level_fan_distribution.csv')
    print(f'Wrote {len(combined)} rows to {target}',flush=True)

if __name__ == "__main__":
    main()
