import pandas as pd
import numpy as np

# Columns that are normally empty in ball-by-ball data
EXPECTED_MISSING = {
    'extras_detail', 'fielders', 'city', 'winner', 'player_of_match',
    'wicket_type', 'dismissal_kind', 'wides', 'noballs', 'byes',
    'legbyes', 'penalty', 'wkt_fielder'
}

# Known spin bowler name fragments (fallback if no bowler_type column)
SPIN_KEYWORDS = ['spin', 'off break', 'leg break', 'left arm spin', 'slow']
FAST_KEYWORDS = ['fast', 'medium', 'pace', 'seam', 'swing']


def detect_columns(df):
    """Auto-detect key column names from the dataframe."""
    cols = df.columns.str.lower().tolist()
    original = df.columns.tolist()

    def find(candidates):
        for c in candidates:
            if c in cols:
                return original[cols.index(c)]
        return None

    return {
        'runs_col':      find(['runs_total', 'runs_off_bat', 'batter_runs', 'runs_scored', 'runs_batter']),
        'batter':        find(['batter', 'striker', 'batsman']),
        'bowler':        find(['bowler']),
        'batting_team':  find(['batting_team', 'team1', 'bat_team']),
        'bowling_team':  find(['bowling_team', 'team2', 'bowl_team']),
        'over':          find(['over', 'ball_in_over', 'over_number']),
        'ball':          find(['ball', 'ball_number', 'ball_str']),
        'match_id':      find(['match_id', 'match_no', 'id']),
        'innings':       find(['innings_number', 'innings', 'inning']),
        'wicket_type':   find(['wicket_type', 'dismissal_kind', 'kind']),
        'wicket_player': find(['player_out', 'player_dismissed', 'out_batter', 'batter_out']),
        'fielder':       find(['fielders', 'fielder', 'wkt_fielder', 'catcher']),
        'extras':        find(['runs_extras', 'extras', 'extra_runs']),
        'wides':         find(['wides', 'wide']),
        'noballs':       find(['noballs', 'no_balls', 'noball']),
        'is_boundary':   find(['is_boundary', 'boundary']),
        'is_dot':        find(['is_dot_ball', 'dot_ball']),
        'bowler_type':   find(['bowler_type', 'bowling_type', 'bowl_style', 'bowl_kind']),
        'non_striker':   find(['non_striker', 'non_batter']),
        'venue':         find(['venue', 'ground', 'stadium']),
        'date':          find(['date', 'match_date']),
        'winner':        find(['winner', 'match_winner']),
        'toss_winner':   find(['toss_winner']),
        'toss_decision': find(['toss_decision']),
    }


def safe_col(df, key, cols):
    """Return column value only if it exists in df."""
    val = cols.get(key)
    return val if val and val in df.columns else None


def compute_all_metrics(df):
    """
    Compute all cricket statistics from a ball-by-ball dataframe.
    Returns a dict of metric dataframes and a dict of availability notes.
    """
    metrics = {}
    notes = {}
    df = df.copy()
    c = detect_columns(df)

    # Ensure runs column is numeric
    rc = c['runs_col']
    if not rc:
        for candidate in ['runs_batter', 'runs_off_bat', 'batter_runs']:
            if candidate in df.columns:
                rc = candidate
                break
    if not rc:
        notes['runs'] = 'No runs column detected. Most stats unavailable.'
        return metrics, notes

    # Fast mode skips expensive row-by-row computations for large datasets
    fast_mode = len(df) > 100000
    if fast_mode:
        print(f"DEBUG: Fast mode enabled ({len(df):,} rows) - skipping hat tricks and partnerships")

    df[rc] = pd.to_numeric(df[rc], errors='coerce').fillna(0)

    # Build group key lists
    def gkeys(*keys):
        return [c[k] for k in keys if c.get(k) and c[k] in df.columns]

    match = safe_col(df, 'match_id', c)
    innings = safe_col(df, 'innings', c)
    bat_team = safe_col(df, 'batting_team', c)
    bowl_team = safe_col(df, 'bowling_team', c)
    batter = safe_col(df, 'batter', c)
    bowler = safe_col(df, 'bowler', c)
    over = safe_col(df, 'over', c)
    wicket = safe_col(df, 'wicket_type', c)
    fielder = safe_col(df, 'fielder', c)
    bowler_type = safe_col(df, 'bowler_type', c)
    non_striker = safe_col(df, 'non_striker', c)
    wides = safe_col(df, 'wides', c)
    noballs = safe_col(df, 'noballs', c)
    extras = safe_col(df, 'extras', c)
    winner = safe_col(df, 'winner', c)
    toss_winner = safe_col(df, 'toss_winner', c)
    toss_decision = safe_col(df, 'toss_decision', c)

    inning_keys = [k for k in [match, innings, bat_team] if k]
    bowl_inning_keys = [k for k in [match, innings, bowl_team] if k]

    # ----------------------------------------------------------------
    # 1. BATTING: runs, balls, strike rate, 4s, 6s, dots
    # ----------------------------------------------------------------
    if batter:
        bat_keys = [k for k in [match, innings, bat_team, batter] if k]
        batting = df.groupby(bat_keys).agg(
            runs_scored=(rc, 'sum'),
            balls_faced=(rc, 'count'),
        ).reset_index()
        batting['strike_rate'] = (batting['runs_scored'] / batting['balls_faced'] * 100).round(2)

        # 4s
        if c.get('is_boundary') and c['is_boundary'] in df.columns:
            fours = df[df[c['is_boundary']] == 1].groupby(bat_keys).size().reset_index(name='fours')
        else:
            fours = df[df[rc] == 4].groupby(bat_keys).size().reset_index(name='fours')
        batting = batting.merge(fours, on=bat_keys, how='left')
        batting['fours'] = batting['fours'].fillna(0).astype(int)

        # 6s
        sixes = df[df[rc] == 6].groupby(bat_keys).size().reset_index(name='sixes')
        batting = batting.merge(sixes, on=bat_keys, how='left')
        batting['sixes'] = batting['sixes'].fillna(0).astype(int)

        # Dot balls faced
        dots = df[df[rc] == 0].groupby(bat_keys).size().reset_index(name='dot_balls_faced')
        batting = batting.merge(dots, on=bat_keys, how='left')
        batting['dot_balls_faced'] = batting['dot_balls_faced'].fillna(0).astype(int)

        metrics['batting'] = batting

        # Ducks: batsmen dismissed for 0
        if wicket:
            dismissed = df[df[wicket].notna() & (df[wicket] != '') & (df[wicket] != 'Unknown')]
            if batter in dismissed.columns:
                duck_keys = [k for k in [match, innings, batter] if k]
                batter_runs_per_inning = batting.groupby(duck_keys)['runs_scored'].sum().reset_index()
                dismissed_batters = dismissed.groupby(duck_keys).size().reset_index(name='dismissed')
                ducks_df = batter_runs_per_inning.merge(dismissed_batters, on=duck_keys, how='inner')
                ducks_df = ducks_df[ducks_df['runs_scored'] == 0][duck_keys + ['runs_scored']].copy()
                ducks_df['duck'] = 1
                metrics['ducks'] = ducks_df
        else:
            notes['ducks'] = 'No wicket/dismissal column found. Duck stats unavailable.'

        # Milestones: 50s and 100s
        bat_totals = batting.groupby([k for k in [match, innings, batter] if k])['runs_scored'].sum().reset_index()
        bat_totals['half_century'] = ((bat_totals['runs_scored'] >= 50) & (bat_totals['runs_scored'] < 100)).astype(int)
        bat_totals['century'] = (bat_totals['runs_scored'] >= 100).astype(int)
        metrics['milestones'] = bat_totals

        # Top scorer per inning
        if inning_keys:
            top_scorer = batting.loc[batting.groupby(inning_keys)['runs_scored'].idxmax()].reset_index(drop=True)
            metrics['top_scorer_per_inning'] = top_scorer
    else:
        notes['batting'] = 'No batter column found. Batting stats unavailable.'

    # ----------------------------------------------------------------
    # 2. PARTNERSHIPS
    # ----------------------------------------------------------------
    if not fast_mode and batter and non_striker:
        part_keys = [k for k in [match, innings, bat_team] if k]
        df['_p1'] = df.apply(lambda r: min(str(r[batter]), str(r[non_striker])), axis=1)
        df['_p2'] = df.apply(lambda r: max(str(r[batter]), str(r[non_striker])), axis=1)
        part_group = part_keys + ['_p1', '_p2']
        partnerships = df.groupby(part_group)[rc].sum().reset_index(name='partnership_runs')
        partnerships['half_century_partnership'] = ((partnerships['partnership_runs'] >= 50) & (partnerships['partnership_runs'] < 100)).astype(int)
        partnerships['century_partnership'] = (partnerships['partnership_runs'] >= 100).astype(int)
        metrics['partnerships'] = partnerships
        df.drop(columns=['_p1', '_p2'], inplace=True)
    elif fast_mode:
        notes['partnerships'] = 'Partnership stats skipped for large datasets.'
    else:
        notes['partnerships'] = 'No non_striker column found. Partnership stats unavailable.'

    # ----------------------------------------------------------------
    # 3. BOWLING: wickets, economy, dots, maidens
    # ----------------------------------------------------------------
    if bowler:
        bowl_keys = [k for k in [match, innings, bowl_team, bowler] if k]
        bowling = df.groupby(bowl_keys).agg(
            balls_bowled=(rc, 'count'),
            runs_conceded=(rc, 'sum'),
        ).reset_index()
        bowling['overs_bowled'] = (bowling['balls_bowled'] / 6).round(2)
        bowling['economy_rate'] = (bowling['runs_conceded'] / (bowling['balls_bowled'] / 6)).round(2)

        # Wickets
        if wicket:
            wk_df = df[df[wicket].notna() & (df[wicket] != '') & (df[wicket] != 'Unknown') &
                       ~df[wicket].str.lower().str.contains('run out', na=False)]
            wk_counts = wk_df.groupby(bowl_keys).size().reset_index(name='wickets')
            bowling = bowling.merge(wk_counts, on=bowl_keys, how='left')
            bowling['wickets'] = bowling['wickets'].fillna(0).astype(int)
        else:
            bowling['wickets'] = 0
            notes['wickets'] = 'No wicket column found. Wicket counts set to 0.'

        # Dot balls
        dots_b = df[df[rc] == 0].groupby(bowl_keys).size().reset_index(name='dot_balls_bowled')
        bowling = bowling.merge(dots_b, on=bowl_keys, how='left')
        bowling['dot_balls_bowled'] = bowling['dot_balls_bowled'].fillna(0).astype(int)

        # Maiden overs
        if over and match:
            over_group = [k for k in [match, innings, bowler, over] if k]
            over_runs = df.groupby(over_group)[rc].sum().reset_index(name='over_runs')
            over_balls = df.groupby(over_group)[rc].count().reset_index(name='over_balls')
            over_stats = over_runs.merge(over_balls, on=over_group)
            maiden_overs = over_stats[(over_stats['over_runs'] == 0) & (over_stats['over_balls'] == 6)]
            maiden_group = [k for k in [match, innings, bowler] if k]
            maiden_counts = maiden_overs.groupby(maiden_group).size().reset_index(name='maiden_overs')
            bowling = bowling.merge(maiden_counts, on=maiden_group, how='left')
            bowling['maiden_overs'] = bowling['maiden_overs'].fillna(0).astype(int)
        else:
            bowling['maiden_overs'] = 0
            notes['maidens'] = 'No over column found. Maiden overs set to 0.'

        # Best bowling figures per match
        if match:
            best_figures = bowling.loc[bowling.groupby([k for k in [match, innings] if k])['wickets'].idxmax()].reset_index(drop=True)
            metrics['best_bowling_per_inning'] = best_figures

        metrics['bowling'] = bowling

        # Spin vs fast wickets
        if bowler_type:
            spin_keywords = ['spin', 'off break', 'leg break', 'left-arm spin', 'slow left']
            fast_keywords = ['fast', 'medium', 'pace', 'seam']
            team_bowl_keys = [k for k in [match, innings, bowl_team] if k]
            if wicket:
                wk_all = df[df[wicket].notna() & (df[wicket] != '') & (df[wicket] != 'Unknown')].copy()
                wk_all[bowler_type] = wk_all[bowler_type].astype(str).str.lower()
                spin_mask = wk_all[bowler_type].str.contains('|'.join(spin_keywords), na=False)
                fast_mask = wk_all[bowler_type].str.contains('|'.join(fast_keywords), na=False)
                spin_wk = wk_all[spin_mask].groupby(team_bowl_keys).size().reset_index(name='spin_wickets')
                fast_wk = wk_all[fast_mask].groupby(team_bowl_keys).size().reset_index(name='fast_wickets')
                metrics['spin_wickets'] = spin_wk
                metrics['fast_wickets'] = fast_wk
        else:
            notes['spin_fast'] = 'No bowler_type column found. Spin/fast wicket split unavailable. Add a bowler_type column to enable this.'

        # Top wicket taker per inning
        if bowl_inning_keys and 'wickets' in bowling.columns:
            top_bowler = bowling.loc[bowling.groupby(bowl_inning_keys)['wickets'].idxmax()].reset_index(drop=True)
            metrics['top_wicket_taker_per_inning'] = top_bowler

    else:
        notes['bowling'] = 'No bowler column found. Bowling stats unavailable.'

    # ----------------------------------------------------------------
    # 4. HAT TRICKS
    # ----------------------------------------------------------------
    if bowler and wicket and match and over:
        hat_tricks = []
        # Deduplicate first to avoid false positives from duplicate rows
        ht_keys = [k for k in [match, innings, over, 'ball_in_over', batter, bowler] if k and k in df.columns]
        if 'ball_in_over' not in df.columns:
            ht_keys = [k for k in [match, innings, over, 'ball_str', batter, bowler] if k and k in df.columns]
        ht_df = df.drop_duplicates(subset=ht_keys) if len(ht_keys) >= 4 else df.copy()
        wk_col_ht = ht_df[wicket].notna() & (ht_df[wicket] != '') & (ht_df[wicket] != 'Unknown')
        wicket_balls = ht_df[wk_col_ht].copy()
        # Group by match and bowler, check for 3 consecutive deliveries with wickets
        for (match_id, bowl_name), grp in wicket_balls.groupby([match, bowler]):
            if len(grp) >= 3:
                # Check if any 3 consecutive rows in the full match sequence are all wickets by same bowler
                grp_idx = grp.index.tolist()
                for i in range(len(grp_idx) - 2):
                    i1, i2, i3 = grp_idx[i], grp_idx[i+1], grp_idx[i+2]
                    # Must be truly consecutive balls (no other balls in between)
                    all_idx = ht_df.index.tolist()
                    if all_idx.index(i2) == all_idx.index(i1) + 1 and all_idx.index(i3) == all_idx.index(i2) + 1:
                        hat_tricks.append({'match_id': match_id, 'bowler': bowl_name})
                        break
        metrics['hat_tricks'] = pd.DataFrame(hat_tricks).drop_duplicates() if hat_tricks else pd.DataFrame(columns=['match_id','bowler'])
        if not hat_tricks:
            notes['hat_tricks'] = 'No hat tricks found in this dataset.'
    elif fast_mode:
        notes['hat_tricks'] = 'Hat trick detection skipped for large datasets.'
    else:
        notes['hat_tricks'] = 'Insufficient columns for hat trick detection.'

    # ----------------------------------------------------------------
    # 5. FIELDING: run outs, catches, wicketkeeper dismissals
    # ----------------------------------------------------------------
    if wicket and match:
        field_keys = [k for k in [match, innings, bat_team] if k]
        wk_df_all = df[df[wicket].notna() & (df[wicket] != '') & (df[wicket] != 'Unknown')].copy()
        wk_lower = wk_df_all[wicket].str.lower()

        run_outs = wk_df_all[wk_lower.str.contains('run out', na=False)]
        catches = wk_df_all[wk_lower.str.contains('caught', na=False)]
        keeper = wk_df_all[wk_lower.str.contains('stumped|caught.*keeper', na=False)]

        if len(run_outs) > 0:
            metrics['run_outs'] = run_outs.groupby(field_keys).size().reset_index(name='run_outs')
        else:
            notes['run_outs'] = 'No run outs found in this dataset.'

        if len(catches) > 0:
            metrics['catches'] = catches.groupby(field_keys).size().reset_index(name='catches')
        else:
            notes['catches'] = 'No caught dismissals found in this dataset.'

        if len(keeper) > 0:
            metrics['wicketkeeper_dismissals'] = keeper.groupby(field_keys).size().reset_index(name='keeper_dismissals')
        else:
            notes['wicketkeeper_dismissals'] = 'No wicketkeeper dismissals found or stumped/caught behind not in data.'
    else:
        notes['fielding'] = 'No wicket column found. Fielding stats unavailable.'

    # ----------------------------------------------------------------
    # 6. TEAM INNINGS: run rates by phase
    # ----------------------------------------------------------------
    if bat_team and over:
        over_vals = df[over].dropna()
        if len(over_vals) > 0:
            min_over = int(over_vals.min())
            pp_end = min_over + 5
            mid_end = min_over + 14

            team_innings = df.groupby(inning_keys).agg(
                total_runs=(rc, 'sum'),
                total_balls=(rc, 'count')
            ).reset_index()
            team_innings['run_rate'] = (team_innings['total_runs'] / (team_innings['total_balls'] / 6)).round(2)

            pp_df = df[df[over] <= pp_end].groupby(inning_keys).agg(
                pp_runs=(rc,'sum'), pp_balls=(rc,'count')).reset_index()
            pp_df['powerplay_run_rate'] = (pp_df['pp_runs'] / (pp_df['pp_balls'] / 6)).round(2)

            mid_df = df[(df[over] > pp_end) & (df[over] <= mid_end)].groupby(inning_keys).agg(
                mid_runs=(rc,'sum'), mid_balls=(rc,'count')).reset_index()
            mid_df['middle_overs_run_rate'] = (mid_df['mid_runs'] / (mid_df['mid_balls'] / 6)).round(2)

            death_df = df[df[over] > mid_end].groupby(inning_keys).agg(
                death_runs=(rc,'sum'), death_balls=(rc,'count')).reset_index()
            death_df['death_overs_run_rate'] = (death_df['death_runs'] / (death_df['death_balls'] / 6)).round(2)

            team_innings = team_innings.merge(pp_df[inning_keys + ['pp_runs','powerplay_run_rate']], on=inning_keys, how='left')
            team_innings = team_innings.merge(mid_df[inning_keys + ['mid_runs','middle_overs_run_rate']], on=inning_keys, how='left')
            team_innings = team_innings.merge(death_df[inning_keys + ['death_runs','death_overs_run_rate']], on=inning_keys, how='left')
            metrics['team_innings'] = team_innings

            # Powerplay wickets lost
            if wicket:
                pp_wk = df[(df[over] <= pp_end) & df[wicket].notna() &
                           (df[wicket] != '') & (df[wicket] != 'Unknown')]
                if len(pp_wk) > 0:
                    metrics['powerplay_wickets'] = pp_wk.groupby(inning_keys).size().reset_index(name='powerplay_wickets_lost')
    else:
        notes['team_innings'] = 'No batting_team or over column. Team run rate stats unavailable.'

    # ----------------------------------------------------------------
    # 7. EXTRAS PER INNING
    # ----------------------------------------------------------------
    extra_cols = [c for c in [extras, wides, noballs] if c]
    if extra_cols and inning_keys:
        extras_agg = {}
        for col in extra_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            extras_agg[col] = (col, 'sum')
        extras_df = df.groupby(inning_keys).agg(**{k: v for k,v in extras_agg.items()}).reset_index()
        metrics['extras'] = extras_df
    else:
        notes['extras'] = 'No extras columns found in this dataset.'

    # ----------------------------------------------------------------
    # 8. TOSS AND WIN ANALYSIS
    # ----------------------------------------------------------------
    if toss_winner and winner and match:
        toss_df = df[[match, toss_winner, winner, toss_decision]].drop_duplicates(subset=[match])
        if len(toss_df) > 0:
            toss_df['toss_match_win'] = (toss_df[toss_winner] == toss_df[winner]).astype(int)
            toss_summary = toss_df.groupby(toss_winner).agg(
                toss_wins=('toss_match_win', 'count'),
                match_wins_after_toss=('toss_match_win', 'sum')
            ).reset_index()
            toss_summary['toss_to_win_pct'] = (toss_summary['match_wins_after_toss'] / toss_summary['toss_wins'] * 100).round(1)
            metrics['toss_analysis'] = toss_summary
    else:
        notes['toss'] = 'No toss_winner or winner column. Toss analysis unavailable.'

    # ----------------------------------------------------------------
    # 9. WIN/LOSS RECORD PER TEAM
    # ----------------------------------------------------------------
    if winner and bat_team and match:
        all_teams = df[[match, bat_team, winner]].drop_duplicates(subset=[match, bat_team])
        if len(all_teams) > 0:
            all_teams['win'] = (all_teams[bat_team] == all_teams[winner]).astype(int)
            win_loss = all_teams.groupby(bat_team).agg(
                matches_played=('win','count'),
                wins=('win','sum')
            ).reset_index()
            win_loss['losses'] = win_loss['matches_played'] - win_loss['wins']
            win_loss['win_pct'] = (win_loss['wins'] / win_loss['matches_played'] * 100).round(1)
            metrics['win_loss'] = win_loss
    else:
        notes['win_loss'] = 'No winner column. Win/loss record unavailable.'

    # ----------------------------------------------------------------
    # 10. BOWLER DISMISSAL RECORDS
    # ----------------------------------------------------------------
    wicket_player = safe_col(df, 'wicket_player', c)
    if bowler and wicket and wicket_player:
        dismissed_df = df[
            df[wicket].notna() &
            (df[wicket] != '') &
            (df[wicket] != 'Unknown') &
            (~df[wicket].str.lower().str.contains('run out', na=False)) &
            df[wicket_player].notna() &
            (df[wicket_player] != '')
        ].copy()
        if len(dismissed_df) > 0:
            dism_keys = [k for k in [match, innings, bowler, wicket_player, wicket, bat_team] if k and k in dismissed_df.columns]
            metrics['bowler_dismissals'] = dismissed_df[dism_keys].reset_index(drop=True)
    else:
        notes['bowler_dismissals'] = 'No player_out column found. Individual dismissal records unavailable.'

    # Summary
    print(f"\nDEBUG: Metrics computed: {list(metrics.keys())}")
    print(f"DEBUG: Notes: {list(notes.keys())}")
    for k, v in metrics.items():
        print(f"  {k}: {len(v)} rows")

    return metrics, notes


def get_metric_summary(metrics, notes):
    parts = []
    if 'batting' in metrics:
        parts.append(f"{len(metrics['batting'])} batting records")
    if 'bowling' in metrics:
        parts.append(f"{len(metrics['bowling'])} bowling records")
    if 'team_innings' in metrics:
        parts.append(f"{len(metrics['team_innings'])} team innings with run rates")
    if 'hat_tricks' in metrics and len(metrics['hat_tricks']) > 0:
        parts.append(f"{len(metrics['hat_tricks'])} hat tricks")
    if 'win_loss' in metrics:
        parts.append(f"{len(metrics['win_loss'])} teams in win/loss table")
    if notes:
        parts.append(f"{len(notes)} stats unavailable due to missing columns")
    return ', '.join(parts)
