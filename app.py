from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anthropic
import json
import uuid
from metrics import compute_all_metrics, get_metric_summary, EXPECTED_MISSING
from yaml_converter import convert_yaml_to_df

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_df = None
original_df = None
dataset_info = {}
computed_metrics = {}
metric_notes = {}


def analyse_dataset(df):
    from metrics import detect_columns
    c = detect_columns(df)
    info = {
        'runs_col':     c.get('runs_col') or 'runs_total',
        'batting_team': c.get('batting_team') or 'batting_team',
        'bowling_team': c.get('bowling_team') or 'bowling_team',
        'batter_col':   c.get('batter') or 'batter',
        'bowler_col':   c.get('bowler') or 'bowler',
    }
    print(f"DEBUG dataset_info: {info}")
    return info


def detect_issues(df):
    issues = []
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        if col not in EXPECTED_MISSING:
            issues.append({'type': f'Missing values in "{col}"',
                           'detail': f'{missing[col]} empty cells found'})
    # Use over + ball_in_over as unique key, not ball_str which repeats across overs
    key_cols = [c for c in ['match_id', 'innings_number', 'over', 'ball_in_over'] if c in df.columns]
    if len(key_cols) < 3:
        key_cols = [c for c in ['match_id', 'innings_number', 'ball_str'] if c in df.columns]
    if len(key_cols) >= 3:
        dupes = df.duplicated(subset=key_cols).sum()
        if dupes > 0:
            issues.append({'type': 'Duplicate deliveries detected',
                           'detail': f'{dupes} repeated balls found. Apply fixes to remove them and ensure accurate statistics.'})
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        if col not in EXPECTED_MISSING:
            has_spaces = df[col].dropna().apply(lambda x: x != x.strip()).sum()
            if has_spaces > 0:
                issues.append({'type': f'Extra whitespace in "{col}"',
                               'detail': f'{has_spaces} cells have leading/trailing spaces. These will be trimmed automatically.'})

    # Check for runs column sanity
    rc = None
    for candidate in ['runs_total', 'runs_batter', 'runs_off_bat']:
        if candidate in df.columns:
            rc = candidate
            break
    if rc:
        mean_runs = df[rc].mean()
        if mean_runs < 0.5:
            issues.append({'type': f'WARNING: Runs column "{rc}" looks incorrect',
                           'detail': f'Average runs per ball is {mean_runs:.3f} (expected 1.0-1.5 for T20). The runs data may be missing or incorrectly exported.'})
        elif mean_runs > 3.0:
            issues.append({'type': f'WARNING: Runs column "{rc}" has unusually high values',
                           'detail': f'Average runs per ball is {mean_runs:.3f} (expected 1.0-1.5). This may indicate cumulative totals rather than per-ball runs.'})

    # Check for mixed data types in key columns
    for col in ['over', 'ball_in_over', 'innings_number']:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            if non_numeric > 0:
                issues.append({'type': f'Non-numeric values in "{col}"',
                               'detail': f'{non_numeric} rows have text where numbers are expected. These rows may cause calculation errors.'})

    # Check for negative runs
    if rc and (df[rc] < 0).any():
        neg_count = (df[rc] < 0).sum()
        issues.append({'type': f'Negative run values found',
                       'detail': f'{neg_count} rows have negative runs in "{rc}". These will be set to 0 during cleaning.'})

    # Check for invalid run values per ball using cricket rules:
    # 0-7: Valid (max is no ball 1 + six 6 = 7)
    # 8-9: Invalid - overthrows cannot happen after a six, wide+six impossible
    # 10: Extremely rare but valid - wide (1) + helmet penalty (5) + boundary four (4)
    # 11+: Invalid - no known cricket scenario produces this
    if rc:
        rows_8_9 = df[df[rc].isin([8, 9])]
        rows_10 = df[df[rc] == 10]
        rows_11_plus = df[df[rc] >= 11]

        if len(rows_8_9) > 0:
            issues.append({
                'type': 'Invalid run values (8 or 9 runs on one ball)',
                'detail': (
                    f'{len(rows_8_9)} rows have 8 or 9 runs on a single ball. '
                    f'This is not possible in cricket. Maximum valid total is 7 (no ball + six). '
                    f'Overthrows cannot occur after a six, and a wide cannot combine with a six. '
                    f'These will be capped at 7 during cleaning.'
                )
            })
        if len(rows_10) > 0:
            issues.append({
                'type': 'Rare scenario: 10 runs on one ball - please verify',
                'detail': (
                    f'{len(rows_10)} rows have 10 runs on a single ball. '
                    f'This is only valid in one scenario: wide (1) + fielder helmet penalty (5) + boundary four (4). '
                    f'Please verify these deliveries match this scenario. '
                    f'If correct, click Skip Cleaning to keep them. If incorrect, Apply All Fixes will cap them at 7.'
                )
            })
        if len(rows_11_plus) > 0:
            issues.append({
                'type': 'Invalid run values (11+ runs on one ball)',
                'detail': (
                    f'{len(rows_11_plus)} rows have 11 or more runs on a single ball. '
                    f'No valid cricket scenario produces this total. '
                    f'These are definite data entry errors and will be capped at 7 during cleaning.'
                )
            })

    if not issues:
        issues.append({'type': 'Data looks clean!',
                       'detail': 'No significant issues found. Ready to query.'})
    return issues


def apply_cleaning(df):
    cleaned = df.copy()
    if 'date' in cleaned.columns:
        cleaned['year'] = pd.to_datetime(cleaned['date'], errors='coerce').dt.year.astype('Int64')
    # Use over + ball_in_over as unique key per delivery
    key_cols = [c for c in ['match_id', 'innings_number', 'over', 'ball_in_over'] if c in cleaned.columns]
    if len(key_cols) < 3:
        key_cols = [c for c in ['match_id', 'innings_number', 'ball_str'] if c in cleaned.columns]
    if len(key_cols) >= 3:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=key_cols, keep='first')
        after = len(cleaned)
        print(f"DEBUG: Dedup on {key_cols}: {before} -> {after} rows (removed {before-after})")
    else:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        print(f"DEBUG: Full dedup: {before} -> {len(cleaned)} rows")
    str_cols = cleaned.select_dtypes(include=['object']).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    num_cols = cleaned.select_dtypes(include=['number']).columns
    for col in num_cols:
        if cleaned[col].isnull().sum() > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Fix negative runs - always errors
    # Cap invalid high values:
    #   runs_batter max = 6 (boundary six)
    #   runs_total max = 10 (wide + helmet penalty + four, extremely rare)
    #   Values of 8, 9, 11+ are not valid in cricket
    for candidate in ['runs_batter', 'runs_off_bat']:
        if candidate in cleaned.columns:
            cleaned[candidate] = pd.to_numeric(cleaned[candidate], errors='coerce').fillna(0).clip(lower=0, upper=6)
    if 'runs_total' in cleaned.columns:
        # Keep 7 (no ball six) and 10 (rare wide+penalty+four), cap 8,9,11+ to 7
        def fix_total(x):
            if x < 0: return 0
            if x <= 7: return x
            if x == 10: return x  # rare but valid
            return 7  # cap 8,9,11,12+ to 7
        cleaned['runs_total'] = cleaned['runs_total'].apply(fix_total)

    # Fix non-numeric values in key columns
    for col in ['over', 'ball_in_over', 'innings_number']:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce').fillna(0).astype(int)

    for col in str_cols:
        if col not in EXPECTED_MISSING and cleaned[col].isnull().sum() > 0:
            cleaned[col] = cleaned[col].fillna('Unknown')

    return cleaned


def filter_by_year(df, user_query):
    if 'year' not in df.columns:
        if 'date' in df.columns:
            df = df.copy()
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year.astype('Int64')
        else:
            return df, None
    ql = user_query.lower()
    range_match = re.search(r'(\d{4})\s*[-to]+\s*(\d{4})', user_query)
    since_match = re.search(r'since\s+(\d{4})', ql)
    before_match = re.search(r'before\s+(\d{4})', ql)
    in_match = re.search(r'\bin\s+(\d{4})\b', ql)
    year_note = None
    if range_match:
        y1, y2 = int(range_match.group(1)), int(range_match.group(2))
        df = df[df['year'].between(y1, y2)]
        year_note = f"{y1}-{y2}"
    elif since_match:
        y1 = int(since_match.group(1))
        df = df[df['year'] >= y1]
        year_note = f"since {y1}"
    elif before_match:
        y2 = int(before_match.group(1))
        df = df[df['year'] < y2]
        year_note = f"before {y2}"
    elif in_match:
        y = int(in_match.group(1))
        df = df[df['year'] == y]
        year_note = f"in {y}"
    return df, year_note


def build_context(user_query, computed_metrics, metric_notes, dataset_info):
    bat = dataset_info.get('batter_col', 'batter')
    bwl = dataset_info.get('bowler_col', 'bowler')
    bt  = dataset_info.get('batting_team', 'batting_team')
    ql  = user_query.lower()
    ctx = []

    if 'batting' in computed_metrics:
        top_bat = computed_metrics['batting'].groupby(bat)['runs_scored'].sum().sort_values(ascending=False).head(10)
        ctx.append(f"Top 10 run scorers (total):\n{top_bat.to_string()}")
        innings_count = computed_metrics['batting'].groupby(bat).size().reset_index(name='innings')
        total_runs = computed_metrics['batting'].groupby(bat)['runs_scored'].sum().reset_index()
        avg_df = total_runs.merge(innings_count, on=bat)
        avg_df['avg_per_innings'] = (avg_df['runs_scored'] / avg_df['innings']).round(2)
        top_avg = avg_df[avg_df['innings'] >= 5].sort_values('avg_per_innings', ascending=False).head(10)
        ctx.append(f"Top 10 batting averages per innings (min 5 innings):\n{top_avg[[bat,'avg_per_innings','innings','runs_scored']].to_string()}")

    if 'bowling' in computed_metrics and 'wickets' in computed_metrics['bowling'].columns:
        top_wk = computed_metrics['bowling'].groupby(bwl)['wickets'].sum().sort_values(ascending=False).head(10)
        ctx.append(f"Top 10 wicket takers:\n{top_wk.to_string()}")

    if any(w in ql for w in ['economy', 'dot ball', 'maiden']):
        if 'bowling' in computed_metrics:
            eco = computed_metrics['bowling'].groupby(bwl).agg(
                total_runs=('runs_conceded', 'sum'),
                total_balls=('balls_bowled', 'sum'),
                dot_balls_bowled=('dot_balls_bowled', 'sum'),
                maiden_overs=('maiden_overs', 'sum')
            ).reset_index()
            eco['economy_rate'] = (eco['total_runs'] / (eco['total_balls'] / 6)).round(2)
            eco = eco[eco['total_balls'] >= 18].sort_values('economy_rate').head(10)
            ctx.append(f"Top 10 economy bowlers (min 3 overs):\n{eco[[bwl,'economy_rate','dot_balls_bowled','maiden_overs']].to_string()}")

    if any(w in ql for w in ['six', 'boundary', 'four', 'strike rate']):
        if 'batting' in computed_metrics:
            bat_extra = computed_metrics['batting'].groupby(bat).agg(
                sixes=('sixes', 'sum'), fours=('fours', 'sum'), strike_rate=('strike_rate', 'mean')
            ).reset_index().sort_values('sixes', ascending=False).head(10)
            ctx.append(f"Top 10 by sixes/fours/strike rate:\n{bat_extra.to_string()}")

    if any(w in ql for w in ['run rate', 'powerplay', 'middle over', 'death over']):
        if 'team_innings' in computed_metrics:
            rr = computed_metrics['team_innings'].groupby(bt).agg(
                run_rate=('run_rate', 'mean'),
                powerplay_run_rate=('powerplay_run_rate', 'mean'),
                middle_overs_run_rate=('middle_overs_run_rate', 'mean'),
                death_overs_run_rate=('death_overs_run_rate', 'mean')
            ).reset_index()
            ctx.append(f"Run rates by team:\n{rr.to_string()}")

    if any(w in ql for w in ['win', 'loss', 'record', 'toss']):
        if 'win_loss' in computed_metrics:
            ctx.append(f"Win/loss record:\n{computed_metrics['win_loss'].to_string()}")
        if 'toss_analysis' in computed_metrics:
            ctx.append(f"Toss analysis:\n{computed_metrics['toss_analysis'].to_string()}")

    if any(w in ql for w in ['hat trick', 'hat-trick', 'hattrick']):
        if 'hat_tricks' in computed_metrics:
            ht = computed_metrics['hat_tricks']
            if len(ht) > 0:
                ctx.append(f"Hat tricks found: {len(ht)}")
                ctx.append(f"Full hat trick details:\n{ht.to_string()}")
                ctx.append("NOTE: 'hat tricks against England' means batting_team=England (England was batting, opponent was bowling)")
            else:
                ctx.append("No hat tricks found in this dataset.")

    if any(w in ql for w in ['duck', 'golden duck', 'dismissed for 0']):
        if 'ducks' in computed_metrics:
            duck_counts = computed_metrics['ducks'].groupby(bat).size().sort_values(ascending=False).head(10)
            ctx.append(f"Most ducks (dismissed for 0):\n{duck_counts.to_string()}")

    if any(w in ql for w in ['centur', 'hundred', '100', 'fifty', '50', 'half cent', 'ton', 'duck', 'blob']):
        if 'milestones' in computed_metrics:
            fifties = computed_metrics['milestones']['half_century'].sum()
            hundreds = computed_metrics['milestones']['century'].sum()
            ctx.append(f"Half centuries: {fifties}, Centuries: {hundreds}")
            top_scores = computed_metrics['milestones'].sort_values('runs_scored', ascending=False).head(10)
            ctx.append(f"Top innings scores:\n{top_scores[[bat,'runs_scored']].to_string()}")
            century_bat = computed_metrics['milestones'].groupby(bat)['century'].sum().sort_values(ascending=False)
            ctx.append(f"Batsmen with centuries:\n{century_bat[century_bat > 0].to_string()}")

    if any(w in ql for w in ['run out', 'catch', 'keeper', 'stumped', 'fielding', 'dismissal']):
        for key, label in [('run_outs', 'Run outs'), ('catches', 'Catches'), ('wicketkeeper_dismissals', 'Keeper dismissals')]:
            if key in computed_metrics:
                ctx.append(f"{label}:\n{computed_metrics[key].head(10).to_string()}")
            else:
                ctx.append(f"{label}: not available in this dataset.")

    if any(w in ql for w in ['partnership', 'pair', 'batting pair']):
        if 'partnerships' in computed_metrics:
            top_p = computed_metrics['partnerships'].sort_values('partnership_runs', ascending=False).head(10)
            ctx.append(f"Top partnerships:\n{top_p.to_string()}")
            century_p = computed_metrics['partnerships']['century_partnership'].sum()
            fifty_p = computed_metrics['partnerships']['half_century_partnership'].sum()
            ctx.append(f"Century partnerships: {century_p}, Half century partnerships: {fifty_p}")

    if any(w in ql for w in ['extra', 'wide', 'no ball', 'noball']):
        if 'extras' in computed_metrics:
            ctx.append(f"Extras per innings:\n{computed_metrics['extras'].head(10).to_string()}")

    if any(w in ql for w in ['powerplay wicket', 'wicket powerplay']):
        if 'powerplay_wickets' in computed_metrics:
            ctx.append(f"Powerplay wickets lost:\n{computed_metrics['powerplay_wickets'].head(10).to_string()}")

    if any(w in ql for w in ['best bowling', 'bowling figures', 'best figures']):
        if 'best_bowling_per_inning' in computed_metrics:
            ctx.append(f"Best bowling figures per inning:\n{computed_metrics['best_bowling_per_inning'].head(10).to_string()}")

    if metric_notes:
        ctx.append(f"Stats unavailable (missing columns): {', '.join(metric_notes.keys())}")

    return '\n\n'.join(ctx)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'CricIQ is running'})


@app.route('/session-status')
def session_status():
    """Check if data is already loaded in this session."""
    global current_df, dataset_info, computed_metrics
    if current_df is None:
        return jsonify({'loaded': False})
    try:
        issues = detect_issues(current_df)
        preview = current_df.head(5).fillna('').to_dict(orient='records')
        matches = current_df['match_id'].nunique() if 'match_id' in current_df.columns else 0
        bt = dataset_info.get('batting_team', 'batting_team')
        teams = current_df[bt].nunique() if bt in current_df.columns else 0
        return jsonify({
            'loaded': True,
            'rows': len(current_df),
            'columns': len(current_df.columns),
            'filename': dataset_info.get('filename', 'dataset'),
            'matches': matches,
            'teams': teams,
            'issues': issues,
            'preview': preview
        })
    except Exception as e:
        return jsonify({'loaded': False, 'error': str(e)})


@app.route('/debug')
def debug():
    global current_df, computed_metrics, dataset_info
    if current_df is None:
        return jsonify({'error': 'No data loaded'})
    rc = dataset_info.get('runs_col', 'runs_total')
    result = {
        'runs_col': rc,
        'runs_col_dtype': str(current_df[rc].dtype) if rc in current_df.columns else 'NOT FOUND',
        'runs_col_sample': current_df[rc].head(10).tolist() if rc in current_df.columns else [],
        'runs_col_mean': float(current_df[rc].mean()) if rc in current_df.columns else 0,
        'total_rows': len(current_df),
        'columns': current_df.columns.tolist(),
    }
    if 'bowling' in computed_metrics:
        bdf = computed_metrics['bowling']
        result['bowling_sample'] = bdf.head(5).to_dict(orient='records')
        result['bowling_cols'] = bdf.columns.tolist()
    return jsonify(result)


@app.route('/upload', methods=['POST'])
def upload():
    global current_df, original_df, dataset_info, computed_metrics, metric_notes
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    try:
        filename = file.filename.lower()
        original_filename = file.filename
        yaml_match_count = 0

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            file_content = file.read()
            df, err, yaml_match_count = convert_yaml_to_df(file_content, original_filename)
            if err:
                return jsonify({'success': False, 'error': f'YAML conversion failed: {err}'})
        elif filename.endswith('.zip'):
            file_content = file.read()
            team_filter = request.form.get('team_filter', '').strip()
            year_from = request.form.get('year_from', '').strip()
            year_to = request.form.get('year_to', '').strip()
            df, err, yaml_match_count = convert_yaml_to_df(
                file_content, original_filename,
                team_filter=team_filter or None,
                year_from=year_from or None,
                year_to=year_to or None
            )
            if err:
                return jsonify({'success': False, 'error': f'ZIP conversion failed: {err}'})
            if df is not None and len(df) == 0:
                return jsonify({'success': False, 'error': f'No matches found matching your filters. Try broadening your team or year range.'})
        else:
            return jsonify({'success': False, 'error': 'Supported formats: CSV, Excel (.xlsx), YAML (.yaml/.yml), or ZIP of YAML files'})
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year.astype('Int64')
        current_df = df.copy()
        original_df = df.copy()
        dataset_info = analyse_dataset(df)
        dataset_info['filename'] = file.filename

        # For large datasets use fast computation, for small use full metrics
        if len(df) > 100000:
            print(f"DEBUG: Large dataset ({len(df)} rows) - using fast metrics")
            rc  = dataset_info.get('runs_col', 'runs_total')
            bat = dataset_info.get('batter_col', 'batter')
            bwl = dataset_info.get('bowler_col', 'bowler')
            bt  = dataset_info.get('batting_team', 'batting_team')
            computed_metrics, metric_notes = _fast_metrics(df, rc, bat, bwl, bt)
        else:
            computed_metrics, metric_notes = compute_all_metrics(df)

        issues = detect_issues(df)
        preview = df.head(5).fillna('').to_dict(orient='records')
        summary = get_metric_summary(computed_metrics, metric_notes)
        response = {
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'issues': issues,
            'preview': preview,
            'metric_summary': summary,
            'unavailable_stats': list(metric_notes.keys())
        }
        if yaml_match_count > 0:
            response['yaml_matches'] = yaml_match_count
            response['yaml_message'] = f'Successfully converted {yaml_match_count} match(es) from YAML to ball-by-ball format.'
        return jsonify(response)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/upload-more', methods=['POST'])
def upload_more():
    """Append data from additional ZIP/YAML uploads to existing dataset."""
    global current_df, original_df, dataset_info, computed_metrics, metric_notes

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        filename = file.filename.lower()
        original_filename = file.filename

        if filename.endswith('.yaml') or filename.endswith('.yml'):
            file_content = file.read()
            new_df, err, match_count = convert_yaml_to_df(file_content, original_filename)
        elif filename.endswith('.zip'):
            file_content = file.read()
            new_df, err, match_count = convert_yaml_to_df(file_content, original_filename)
        else:
            return jsonify({'success': False, 'error': 'Only YAML or ZIP files supported for additional uploads'})

        if err:
            return jsonify({'success': False, 'error': err})

        if new_df is None or len(new_df) == 0:
            return jsonify({'success': False, 'error': 'No data found in file'})

        # Add year column
        if 'date' in new_df.columns:
            new_df['year'] = pd.to_datetime(new_df['date'], errors='coerce').dt.year.astype('Int64')

        # Append to existing dataset
        if current_df is not None:
            current_df = pd.concat([current_df, new_df], ignore_index=True)
            # Remove duplicates across uploads
            key_cols = [c for c in ['match_id', 'innings_number', 'over', 'ball_in_over'] if c in current_df.columns]
            if len(key_cols) >= 3:
                before = len(current_df)
                current_df = current_df.drop_duplicates(subset=key_cols, keep='first')
                print(f"DEBUG: After merge dedup: {before} -> {len(current_df)} rows")
        else:
            current_df = new_df.copy()

        original_df = current_df.copy()
        dataset_info = analyse_dataset(current_df)
        # Skip full metrics computation here for speed
        # Will be computed when coach clicks Finalise

        match_total = current_df['match_id'].nunique() if 'match_id' in current_df.columns else match_count

        return jsonify({
            'success': True,
            'rows': len(current_df),
            'matches': match_total,
            'new_matches': match_count,
            'message': f'Added {match_count} matches. Total: {match_total} matches, {len(current_df):,} rows.'
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/finalise', methods=['POST'])
def finalise():
    """Quick finalise - just validate data and compute essential stats only."""
    global current_df, computed_metrics, metric_notes, dataset_info
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    try:
        dataset_info = analyse_dataset(current_df)

        # For large datasets, only compute fast essential metrics
        rc  = dataset_info.get('runs_col', 'runs_total')
        bat = dataset_info.get('batter_col', 'batter')
        bwl = dataset_info.get('bowler_col', 'bowler')
        bt  = dataset_info.get('batting_team', 'batting_team')

        # Quick essential stats only
        computed_metrics = {}
        metric_notes = {}

        if bat in current_df.columns and rc in current_df.columns:
            current_df[rc] = pd.to_numeric(current_df[rc], errors='coerce').fillna(0)
            bat_grp = [k for k in ['match_id','innings_number', bt, bat] if k in current_df.columns]
            batting = current_df.groupby(bat_grp).agg(
                runs_scored=(rc,'sum'), balls_faced=(rc,'count')).reset_index()
            batting['strike_rate'] = (batting['runs_scored']/batting['balls_faced']*100).round(2)
            if 'is_boundary' in current_df.columns:
                fours = current_df[current_df['is_boundary']==1].groupby(bat_grp).size().reset_index(name='fours')
                batting = batting.merge(fours, on=bat_grp, how='left')
                batting['fours'] = batting['fours'].fillna(0).astype(int)
            else:
                batting['fours'] = 0
            sixes = current_df[current_df[rc]==6].groupby(bat_grp).size().reset_index(name='sixes')
            batting = batting.merge(sixes, on=bat_grp, how='left')
            batting['sixes'] = batting['sixes'].fillna(0).astype(int)
            dots = current_df[current_df[rc]==0].groupby(bat_grp).size().reset_index(name='dot_balls_faced')
            batting = batting.merge(dots, on=bat_grp, how='left')
            batting['dot_balls_faced'] = batting['dot_balls_faced'].fillna(0).astype(int)
            computed_metrics['batting'] = batting

        if bwl in current_df.columns:
            bowl_grp = [k for k in ['match_id','innings_number',
                dataset_info.get('bowling_team','bowling_team'), bwl] if k in current_df.columns]
            bowling = current_df.groupby(bowl_grp).agg(
                balls_bowled=(rc,'count'), runs_conceded=(rc,'sum')).reset_index()
            bowling['economy_rate'] = (bowling['runs_conceded']/(bowling['balls_bowled']/6)).round(2)

            wk_col = 'dismissal_kind' if 'dismissal_kind' in current_df.columns else None
            if wk_col:
                wk_df = current_df[current_df[wk_col].notna() &
                    (current_df[wk_col] != '') &
                    (~current_df[wk_col].str.lower().str.contains('run out', na=False))]
                wk_counts = wk_df.groupby(bowl_grp).size().reset_index(name='wickets')
                bowling = bowling.merge(wk_counts, on=bowl_grp, how='left')
                bowling['wickets'] = bowling['wickets'].fillna(0).astype(int)
            dots_b = current_df[current_df[rc]==0].groupby(bowl_grp).size().reset_index(name='dot_balls_bowled')
            bowling = bowling.merge(dots_b, on=bowl_grp, how='left')
            bowling['dot_balls_bowled'] = bowling['dot_balls_bowled'].fillna(0).astype(int)
            bowling['maiden_overs'] = 0
            computed_metrics['bowling'] = bowling

        # Team innings and run rates
        if bt in current_df.columns and 'over' in current_df.columns:
            inn_keys = [k for k in ['match_id','innings_number', bt] if k in current_df.columns]
            over_vals = current_df['over'].dropna()
            min_over = int(over_vals.min()) if len(over_vals) > 0 else 1
            pp_end = min_over + 5
            mid_end = min_over + 14

            team_inn = current_df.groupby(inn_keys).agg(
                total_runs=(rc,'sum'), total_balls=(rc,'count')).reset_index()
            team_inn['run_rate'] = (team_inn['total_runs']/(team_inn['total_balls']/6)).round(2)

            pp = current_df[current_df['over']<=pp_end].groupby(inn_keys).agg(
                pp_runs=(rc,'sum'), pp_balls=(rc,'count')).reset_index()
            pp['powerplay_run_rate'] = (pp['pp_runs']/(pp['pp_balls']/6)).round(2)

            mid = current_df[(current_df['over']>pp_end)&(current_df['over']<=mid_end)].groupby(inn_keys).agg(
                mid_runs=(rc,'sum'), mid_balls=(rc,'count')).reset_index()
            mid['middle_overs_run_rate'] = (mid['mid_runs']/(mid['mid_balls']/6)).round(2)

            death = current_df[current_df['over']>mid_end].groupby(inn_keys).agg(
                death_runs=(rc,'sum'), death_balls=(rc,'count')).reset_index()
            death['death_overs_run_rate'] = (death['death_runs']/(death['death_balls']/6)).round(2)

            team_inn = team_inn.merge(pp[inn_keys+['pp_runs','powerplay_run_rate']], on=inn_keys, how='left')
            team_inn = team_inn.merge(mid[inn_keys+['mid_runs','middle_overs_run_rate']], on=inn_keys, how='left')
            team_inn = team_inn.merge(death[inn_keys+['death_runs','death_overs_run_rate']], on=inn_keys, how='left')
            computed_metrics['team_innings'] = team_inn

        # Milestones
        if 'batting' in computed_metrics:
            mil_keys = [k for k in ['match_id','innings_number', bat] if k in current_df.columns]
            mil = computed_metrics['batting'].groupby(mil_keys)['runs_scored'].sum().reset_index()
            mil['half_century'] = ((mil['runs_scored']>=50)&(mil['runs_scored']<100)).astype(int)
            mil['century'] = (mil['runs_scored']>=100).astype(int)
            computed_metrics['milestones'] = mil

        # Win loss
        if 'winner' in current_df.columns and bt in current_df.columns and 'match_id' in current_df.columns:
            matches = current_df[['match_id', bt, 'winner']].drop_duplicates(subset=['match_id', bt])
            matches['win'] = (matches[bt]==matches['winner']).astype(int)
            wl = matches.groupby(bt).agg(matches_played=('win','count'), wins=('win','sum')).reset_index()
            wl['losses'] = wl['matches_played'] - wl['wins']
            wl['win_pct'] = (wl['wins']/wl['matches_played']*100).round(1)
            computed_metrics['win_loss'] = wl

        # Bowler dismissals
        if 'player_out' in current_df.columns and bwl in current_df.columns:
            wk_col = 'dismissal_kind' if 'dismissal_kind' in current_df.columns else None
            if wk_col:
                dism = current_df[current_df[wk_col].notna() &
                    (current_df[wk_col]!='') &
                    (~current_df[wk_col].str.lower().str.contains('run out', na=False)) &
                    current_df['player_out'].notna() &
                    (current_df['player_out']!='')]
                dism_keys = [k for k in ['match_id','innings_number', bwl, 'player_out', wk_col, bt] if k in dism.columns]
                computed_metrics['bowler_dismissals'] = dism[dism_keys].reset_index(drop=True)

        metric_notes['hat_tricks'] = 'Hat trick detection skipped for large datasets.'
        metric_notes['partnerships'] = 'Partnership stats skipped for large datasets.'

        issues = detect_issues(current_df)
        preview = current_df.head(5).fillna('').to_dict(orient='records')
        matches = current_df['match_id'].nunique() if 'match_id' in current_df.columns else 0

        print(f"DEBUG: Finalise complete. Metrics: {list(computed_metrics.keys())}")
        return jsonify({
            'success': True,
            'rows': len(current_df),
            'columns': len(current_df.columns),
            'issues': issues,
            'preview': preview,
            'metric_summary': f"{matches} matches, {len(current_df):,} rows",
            'matches': matches
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/players')
def players():
    """Search for player names in the dataset."""
    global current_df, dataset_info
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    search = request.args.get('q', '').lower()
    bat = dataset_info.get('batter_col', 'batter')
    bwl = dataset_info.get('bowler_col', 'bowler')
    all_players = set()
    if bat in current_df.columns:
        all_players.update(current_df[bat].dropna().unique().tolist())
    if bwl in current_df.columns:
        all_players.update(current_df[bwl].dropna().unique().tolist())
    if search:
        matches = [p for p in all_players if search in p.lower()]
    else:
        matches = list(all_players)
    matches = sorted(matches)[:50]
    return jsonify({'success': True, 'players': matches, 'total': len(all_players)})


@app.route('/download-csv')
def download_csv():
    """Allow coach to download the current dataset as a clean CSV file."""
    global current_df
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    try:
        from flask import Response
        import io
        csv_buffer = io.StringIO()
        current_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return Response(
            csv_buffer.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=criciq_dataset.csv',
                'Content-Type': 'text/csv'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/clear-data', methods=['POST'])
def clear_data():
    """Clear the current dataset so coach can start fresh."""
    global current_df, original_df, dataset_info, computed_metrics, metric_notes
    current_df = None
    original_df = None
    dataset_info = {}
    computed_metrics = {}
    metric_notes = {}
    return jsonify({'success': True, 'message': 'Data cleared. Ready for new upload.'})


@app.route('/clean', methods=['POST'])
def clean():
    global current_df, computed_metrics, metric_notes
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    try:
        data = request.get_json()
        if data.get('action') == 'apply':
            current_df = apply_cleaning(current_df)
            # For large datasets skip full recompute - metrics already computed at finalise
            if len(current_df) <= 100000:
                computed_metrics, metric_notes = compute_all_metrics(current_df)
            else:
                print(f"DEBUG: Large dataset - skipping metrics recompute after cleaning")
            preview = current_df.head(5).fillna('').to_dict(orient='records')
            return jsonify({'success': True, 'preview': preview})
        else:
            return jsonify({'success': False, 'error': 'Unknown action'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def _fast_metrics(df, rc, bat, bwl, bt):
    """Fast metrics computation for large datasets - skips expensive operations."""
    metrics = {}
    notes = {}

    df = df.copy()
    if rc in df.columns:
        df[rc] = pd.to_numeric(df[rc], errors='coerce').fillna(0)

    bwt = 'bowling_team' if 'bowling_team' in df.columns else bt

    # Batting
    if bat in df.columns and rc in df.columns:
        bat_grp = [k for k in ['match_id','innings_number', bt, bat] if k in df.columns]
        batting = df.groupby(bat_grp).agg(runs_scored=(rc,'sum'), balls_faced=(rc,'count')).reset_index()
        batting['strike_rate'] = (batting['runs_scored']/batting['balls_faced']*100).round(2)
        if 'is_boundary' in df.columns:
            fours = df[df['is_boundary']==1].groupby(bat_grp).size().reset_index(name='fours')
            batting = batting.merge(fours, on=bat_grp, how='left')
            batting['fours'] = batting['fours'].fillna(0).astype(int)
        else:
            batting['fours'] = 0
        sixes = df[df[rc]==6].groupby(bat_grp).size().reset_index(name='sixes')
        batting = batting.merge(sixes, on=bat_grp, how='left')
        batting['sixes'] = batting['sixes'].fillna(0).astype(int)
        dots = df[df[rc]==0].groupby(bat_grp).size().reset_index(name='dot_balls_faced')
        batting = batting.merge(dots, on=bat_grp, how='left')
        batting['dot_balls_faced'] = batting['dot_balls_faced'].fillna(0).astype(int)
        metrics['batting'] = batting

        # Milestones
        mil_keys = [k for k in ['match_id','innings_number', bat] if k in df.columns]
        mil = batting.groupby(mil_keys)['runs_scored'].sum().reset_index()
        mil['half_century'] = ((mil['runs_scored']>=50)&(mil['runs_scored']<100)).astype(int)
        mil['century'] = (mil['runs_scored']>=100).astype(int)
        metrics['milestones'] = mil

    # Bowling
    if bwl in df.columns and rc in df.columns:
        bowl_grp = [k for k in ['match_id','innings_number', bwt, bwl] if k in df.columns]
        bowling = df.groupby(bowl_grp).agg(balls_bowled=(rc,'count'), runs_conceded=(rc,'sum')).reset_index()
        bowling['economy_rate'] = (bowling['runs_conceded']/(bowling['balls_bowled']/6)).round(2)
        wk_col = 'dismissal_kind' if 'dismissal_kind' in df.columns else None
        if wk_col:
            wk_df = df[df[wk_col].notna() & (df[wk_col]!='') & (~df[wk_col].str.lower().str.contains('run out', na=False))]
            wk_counts = wk_df.groupby(bowl_grp).size().reset_index(name='wickets')
            bowling = bowling.merge(wk_counts, on=bowl_grp, how='left')
            bowling['wickets'] = bowling['wickets'].fillna(0).astype(int)
        dots_b = df[df[rc]==0].groupby(bowl_grp).size().reset_index(name='dot_balls_bowled')
        bowling = bowling.merge(dots_b, on=bowl_grp, how='left')
        bowling['dot_balls_bowled'] = bowling['dot_balls_bowled'].fillna(0).astype(int)
        bowling['maiden_overs'] = 0
        metrics['bowling'] = bowling

        # Bowler dismissals
        if wk_col and 'player_out' in df.columns:
            dism = df[df[wk_col].notna() & (df[wk_col]!='') &
                (~df[wk_col].str.lower().str.contains('run out', na=False)) &
                df['player_out'].notna() & (df['player_out']!='')]
            dism_keys = [k for k in ['match_id','innings_number', bwl, 'player_out', wk_col, bt] if k in dism.columns]
            metrics['bowler_dismissals'] = dism[dism_keys].reset_index(drop=True)

    # Team innings with phase run rates
    if bt in df.columns and 'over' in df.columns and rc in df.columns:
        inn_keys = [k for k in ['match_id','innings_number', bt] if k in df.columns]
        over_vals = df['over'].dropna()
        min_over = int(over_vals.min()) if len(over_vals) > 0 else 1
        pp_end = min_over + 5
        mid_end = min_over + 14

        team_inn = df.groupby(inn_keys).agg(total_runs=(rc,'sum'), total_balls=(rc,'count')).reset_index()
        team_inn['run_rate'] = (team_inn['total_runs']/(team_inn['total_balls']/6)).round(2)

        pp = df[df['over']<=pp_end].groupby(inn_keys).agg(pp_runs=(rc,'sum'), pp_balls=(rc,'count')).reset_index()
        pp['powerplay_run_rate'] = (pp['pp_runs']/(pp['pp_balls']/6)).round(2)

        mid = df[(df['over']>pp_end)&(df['over']<=mid_end)].groupby(inn_keys).agg(mid_runs=(rc,'sum'), mid_balls=(rc,'count')).reset_index()
        mid['middle_overs_run_rate'] = (mid['mid_runs']/(mid['mid_balls']/6)).round(2)

        death = df[df['over']>mid_end].groupby(inn_keys).agg(death_runs=(rc,'sum'), death_balls=(rc,'count')).reset_index()
        death['death_overs_run_rate'] = (death['death_runs']/(death['death_balls']/6)).round(2)

        team_inn = team_inn.merge(pp[inn_keys+['pp_runs','powerplay_run_rate']], on=inn_keys, how='left')
        team_inn = team_inn.merge(mid[inn_keys+['mid_runs','middle_overs_run_rate']], on=inn_keys, how='left')
        team_inn = team_inn.merge(death[inn_keys+['death_runs','death_overs_run_rate']], on=inn_keys, how='left')
        metrics['team_innings'] = team_inn

    # Win loss
    if 'winner' in df.columns and bt in df.columns and 'match_id' in df.columns:
        matches = df[['match_id', bt, 'winner']].drop_duplicates(subset=['match_id', bt])
        matches = matches.copy()
        matches['win'] = (matches[bt]==matches['winner']).astype(int)
        wl = matches.groupby(bt).agg(matches_played=('win','count'), wins=('win','sum')).reset_index()
        wl['losses'] = wl['matches_played'] - wl['wins']
        wl['win_pct'] = (wl['wins']/wl['matches_played']*100).round(1)
        metrics['win_loss'] = wl

    # Fielding: catches, run outs, wicketkeeper dismissals
    wk_col = 'dismissal_kind' if 'dismissal_kind' in df.columns else None
    fielders_col = 'fielders' if 'fielders' in df.columns else None
    if wk_col and bt in df.columns and 'match_id' in df.columns:
        match_keys = [k for k in ['match_id', bt] if k in df.columns]

        # Run outs per team per match
        run_outs_df = df[df[wk_col].str.lower().str.contains('run out', na=False)]
        if len(run_outs_df) > 0:
            run_outs = run_outs_df.groupby(match_keys).size().reset_index(name='run_outs')
            metrics['run_outs'] = run_outs

        # Catches - dismissal_kind == caught (but not caught and bowled, which belongs to bowler)
        caught_df = df[df[wk_col].str.lower().str.strip() == 'caught']
        if len(caught_df) > 0:
            catches = caught_df.groupby(match_keys).size().reset_index(name='catches')
            metrics['catches'] = catches

        # Wicketkeeper dismissals (caught + stumped where keeper involved)
        keeper_df = df[df[wk_col].str.lower().str.contains('stumped', na=False) |
                      (df[wk_col].str.lower().str.strip() == 'caught')]
        if len(keeper_df) > 0:
            keeper = keeper_df.groupby(match_keys).size().reset_index(name='wicketkeeper_dismissals')
            metrics['wicketkeeper_dismissals'] = keeper

    # Ducks
    if 'player_out' in df.columns and rc in df.columns:
        bat_grp = [k for k in ['match_id', 'innings_number', bt, bat] if k in df.columns]
        dismissed = df[df['player_out'].notna() & (df['player_out'] != '')]
        runs_when_out = dismissed.groupby(bat)[rc].sum().reset_index(name='runs_at_dismissal')
        ducks = runs_when_out[runs_when_out['runs_at_dismissal'] == 0]
        if len(ducks) > 0:
            metrics['ducks'] = ducks

    # Powerplay wickets lost per team per innings
    if 'over' in df.columns and wk_col and bt in df.columns:
        pp_df = df[df['over'] <= 6]
        pp_wk = pp_df[pp_df[wk_col].notna() & (pp_df[wk_col] != '')]
        inn_keys = [k for k in ['match_id', 'innings_number', bt] if k in df.columns]
        if len(pp_wk) > 0:
            pp_wickets = pp_wk.groupby(inn_keys).size().reset_index(name='powerplay_wickets_lost')
            metrics['powerplay_wickets'] = pp_wickets

    notes['hat_tricks'] = 'Skipped for large dataset.'
    notes['partnerships'] = 'Skipped for large dataset.'

    print(f"DEBUG: Fast metrics done: {list(metrics.keys())}")
    return metrics, notes


def fuzzy_find_player(query, player_list):
    """
    Find a player name from a list based on a natural language query.
    Handles:
    - Full name: "SM Curran" or "Sam Curran"
    - Surname only: "Curran"
    - Partial initials: "S Curran"
    - Common misspellings via partial matching
    Returns the best matching player name or None.
    """
    ql = query.lower().strip()
    player_list = list(player_list)

    # 1. Exact full name match
    for p in player_list:
        if p.lower() in ql:
            return p

    # 2. All parts of name appear in query (handles "SM Curran" -> "Sam Curran")
    for p in player_list:
        parts = [x for x in p.lower().split() if len(x) > 1]
        if len(parts) >= 2 and all(part in ql for part in parts):
            return p

    # 3. Surname only match (most reliable single-word match)
    for p in player_list:
        surname = p.split()[-1].lower()
        if len(surname) > 3 and surname in ql:
            return p

    # 4. Any significant part of name appears
    for p in player_list:
        parts = [x for x in p.lower().split() if len(x) > 3]
        if any(part in ql for part in parts):
            return p

    # 5. Fuzzy: query words appear in player name
    query_words = [w for w in ql.split() if len(w) > 3]
    best_match = None
    best_score = 0
    for p in player_list:
        p_lower = p.lower()
        score = sum(1 for w in query_words if w in p_lower)
        if score > best_score:
            best_score = score
            best_match = p

    return best_match if best_score > 0 else None


# Cricket terminology dictionary - terms relevant to data queries
# Sourced from Wikipedia Glossary of Cricket Terms and domain knowledge
CRICKET_TERMS = {
    # ---- BATTING MILESTONES ----
    'century': '100 runs in an innings',
    'centuries': '100+ runs in an innings',
    'hundred': '100 runs in an innings',
    'hundreds': '100+ runs in an innings',
    'ton': '100 runs in an innings',
    'tons': '100+ runs in an innings',
    'half century': '50 to 99 runs in an innings',
    'half centuries': '50 to 99 runs in an innings',
    'fifty': '50 to 99 runs in an innings',
    'fifties': '50 to 99 runs in an innings',
    'half ton': '50 to 99 runs in an innings',
    'duck': 'dismissed for 0 runs',
    'ducks': 'dismissed for 0 runs',
    'golden duck': 'dismissed first ball for 0 runs',
    'diamond duck': 'run out without facing a ball for 0',
    'blob': 'dismissed for 0 runs',  # slang for duck
    'audi': 'four consecutive ducks',
    'pair': 'dismissed for 0 in both innings of a match',

    # ---- BOWLING MILESTONES ----
    'hat trick': 'three wickets in three consecutive balls by same bowler',
    'hat-trick': 'three wickets in three consecutive balls by same bowler',
    'hattrick': 'three wickets in three consecutive balls by same bowler',
    'five wicket haul': '5 or more wickets in an innings',
    'five for': '5 or more wickets in an innings',
    'fifer': '5 or more wickets in an innings',
    'ten for': '10 wickets in a match',
    'four wicket haul': '4 wickets in an innings',
    'four fer': '4 wickets in an innings',

    # ---- BOWLING METRICS ----
    'maiden': 'over where no runs were conceded',
    'maiden over': 'over where no runs were conceded',
    'dot ball': 'delivery where no runs were scored',
    'dot balls': 'deliveries where no runs were scored',
    'economy': 'runs conceded per over bowled',
    'economy rate': 'runs conceded per over bowled',
    'bowling average': 'runs conceded per wicket taken',
    'strike rate': 'balls bowled per wicket (bowling) or runs per 100 balls (batting)',
    'bowling strike rate': 'balls bowled per wicket taken',
    'batting strike rate': 'runs scored per 100 balls faced',
    'analysis': 'bowling figures showing overs wickets and runs',
    'figures': 'bowling performance in format wickets for runs',
    'best bowling': 'best bowling figures in an innings or match',
    'beamer': 'illegal delivery above waist height without bouncing',
    'bouncer': 'short pitched delivery rising near batter head',
    'full toss': 'delivery reaching batter without bouncing',
    'yorker': 'delivery aimed at batter feet near crease',
    'wide': 'delivery too far from batter to be hit awarded 1 extra',
    'no ball': 'illegal delivery awarding 1 extra and a free hit in limited overs',
    'free hit': 'delivery following a no ball where batter cannot be dismissed',

    # ---- BATTING SHOTS AND EVENTS ----
    'six': '6 runs scored off bat over boundary without bouncing',
    'sixes': '6 runs scored off bat over boundary without bouncing',
    'maximum': '6 runs scored off bat',
    'maximums': '6 runs scored off bat',
    'four': '4 runs scored off bat reaching boundary along ground',
    'fours': '4 runs scored off bat reaching boundary along ground',
    'boundary': '4 or 6 runs scored when ball reaches perimeter',
    'boundaries': '4 or 6 runs scored when ball reaches perimeter',
    'overthrow': 'extra runs scored when fielder throws ball past stumps',
    'overthrows': 'extra runs scored when fielder throws ball past stumps',
    'bye': 'extra runs scored when ball passes wicketkeeper without touching bat',
    'byes': 'extra runs scored without touching bat or wicketkeeper',
    'leg bye': 'extra runs scored off batter body but not bat',
    'leg byes': 'extra runs scored off batter body but not bat',
    'extras': 'runs scored not off bat including wides no balls byes leg byes',
    'penalty runs': '5 extra runs awarded for fielding infringement',

    # ---- DISMISSAL TYPES ----
    'bowled': 'dismissed when ball hits stumps and removes bail',
    'caught': 'dismissed when fielder catches ball before it bounces',
    'caught behind': 'dismissed caught by wicket keeper',
    'caught and bowled': 'dismissed caught by the bowler themselves',
    'lbw': 'dismissed leg before wicket when ball would have hit stumps',
    'leg before wicket': 'dismissed when ball hits pad and would have hit stumps',
    'run out': 'dismissed when batter is outside crease when stumps are broken',
    'stumped': 'dismissed when wicketkeeper breaks stumps while batter is outside crease',
    'hit wicket': 'dismissed when batter breaks their own stumps',
    'obstructing the field': 'dismissed for deliberately blocking a fielder',
    'handled the ball': 'dismissed for touching ball with hand without consent',
    'timed out': 'dismissed for taking too long to come to the crease',
    'retired out': 'batter leaves without being dismissed counts as out',
    'retired hurt': 'batter leaves due to injury does not count as out',

    # ---- FIELDING POSITIONS AND EVENTS ----
    'keeper': 'wicket keeper',
    'wicket keeper': 'player who fields behind the stumps',
    'wicketkeeper': 'player who fields behind the stumps',
    'wk': 'wicket keeper',
    'keeper dismissals': 'dismissals involving the wicket keeper caught and stumped',
    'slip': 'fielder positioned behind batter on off side for catches',
    'gully': 'fielder position between slip and point',
    'point': 'fielder position square on the off side',
    'cover': 'fielder position in front of point on off side',
    'mid off': 'fielder position near the bowler on off side',
    'mid on': 'fielder position near the bowler on leg side',
    'mid wicket': 'fielder position square on the leg side',
    'square leg': 'fielder position square on the leg side',
    'fine leg': 'fielder position behind square on leg side',
    'third man': 'fielder position behind wicketkeeper on off side',
    'long on': 'fielder position deep on leg side near boundary',
    'long off': 'fielder position deep on off side near boundary',
    'cow corner': 'area between mid wicket and long on',

    # ---- MATCH PHASES ----
    'powerplay': 'first 6 overs with fielding restrictions',
    'power play': 'first 6 overs with fielding restrictions',
    'pp': 'powerplay first 6 overs',
    'middle overs': 'overs 7 to 15',
    'death overs': 'final overs 16 to 20',
    'death': 'final overs 16 to 20 of the innings',
    'slog overs': 'final overs where batters attack more',

    # ---- PLAYER ROLES ----
    'opener': 'batter who opens the innings batting first',
    'openers': 'two batters who start the innings',
    'top order': 'first three or four batters in the order',
    'middle order': 'batters usually coming in at positions 4 to 7',
    'lower order': 'batters usually coming in at positions 8 to 11',
    'tail': 'lower order batters who are primarily bowlers',
    'tailender': 'lower order batter who is primarily a bowler',
    'nightwatchman': 'lower order batter sent in late to protect top order batters',
    'pinch hitter': 'lower order batter promoted to score quickly',
    'all rounder': 'player who contributes significantly with both bat and bowl',
    'allrounder': 'player who contributes significantly with both bat and bowl',
    'pace bowler': 'bowler who bowls fast deliveries',
    'fast bowler': 'bowler who bowls at high speed',
    'seam bowler': 'bowler who uses seam of ball to move it off pitch',
    'swing bowler': 'bowler who moves ball through the air',
    'spin bowler': 'bowler who imparts spin on ball causing it to turn',
    'off spinner': 'right arm bowler whose ball turns from off to leg for right hand batter',
    'leg spinner': 'bowler whose ball turns from leg to off for right hand batter',
    'left arm spinner': 'left arm bowler whose ball turns away from right hand batter',

    # ---- TEAM AND MATCH TERMS ----
    'innings': 'period when one team bats',
    'over': '6 legal deliveries bowled by same bowler',
    'run rate': 'runs scored per over',
    'required run rate': 'runs per over needed to win',
    'asking rate': 'runs per over needed to win',
    'net run rate': 'difference between runs scored and conceded per over across tournament',
    'nrr': 'net run rate',
    'target': 'runs needed to win set by first batting team',
    'total': 'runs scored in an innings',
    'partnership': 'runs scored between two batters batting together',
    'stand': 'runs scored between two batters batting together',
    'century stand': '100 or more runs partnership',
    'century partnership': '100 or more runs scored together',
    'fifty stand': '50 or more runs partnership',
    'half century stand': '50 to 99 runs scored together',
    'batting collapse': 'multiple wickets falling quickly for few runs',
    'toss': 'coin flip deciding which team bats or fields first',
    'drs': 'decision review system for challenging umpire decisions',
    'review': 'challenge to umpire decision using technology',
    'super over': 'one over per side tie breaker in limited overs cricket',

    # ---- PITCH AND CONDITIONS ----
    'pitch': 'prepared strip of ground where bowling and batting takes place',
    'track': 'the pitch surface',
    'strip': 'the pitch surface',
    'crease': 'lines on pitch marking safe zones for batters and bowlers',
    'popping crease': 'line batter must be behind to avoid run out',
    'swing': 'movement of ball through air before bouncing',
    'seam': 'movement of ball off pitch after bouncing on seam',
    'reverse swing': 'swing in opposite direction achieved with old ball',
    'spin': 'rotation imparted on ball causing it to turn after bouncing',
    'turn': 'sideways movement of ball off pitch from spin',
    'bounce': 'height ball rises after pitching',
    'carry': 'distance ball travels in air after bouncing',
}

def normalise_cricket_query(query):
    """Replace cricket terminology with technical terms the AI understands."""
    normalised = query.lower()

    # Map cricket terms to technical equivalents for context
    # We keep the original query but add context
    term_mappings = []
    for term, meaning in CRICKET_TERMS.items():
        if term in normalised:
            term_mappings.append(f"{term} = {meaning}")

    return query, term_mappings


@app.route('/query', methods=['POST'])
def query():
    global current_df, dataset_info, computed_metrics, metric_notes
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded. Please upload a file first.'})
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        if not user_query:
            return jsonify({'success': False, 'error': 'No query provided'})

        # Normalise cricket terminology
        user_query_original, cricket_term_context = normalise_cricket_query(user_query)

        rc  = dataset_info.get('runs_col', 'runs_total')
        bt  = dataset_info.get('batting_team', 'batting_team')
        bwt = dataset_info.get('bowling_team', 'bowling_team')
        bat = dataset_info.get('batter_col', 'batter')
        bwl = dataset_info.get('bowler_col', 'bowler')
        ql  = user_query.lower()

        # Expand cricket terms in query for keyword matching
        ql_expanded = ql
        cricket_expansions = {
            'centur': '100', 'hundred': '100', 'ton ': '100 ',
            'half centur': '50', 'fifty': '50', 'fif': '50',
            'six': 'six', 'maximum': 'six', 'four': 'four',
            'boundary': 'four six', 'boundaries': 'four six',
            'maiden': 'maiden', 'dot ball': 'dot ball',
            'fifer': 'wicket', 'five for': 'wicket',
            'duck': 'duck', 'golden duck': 'duck',
        }
        for term, expansion in cricket_expansions.items():
            if term in ql_expanded:
                ql_expanded = ql_expanded.replace(term, expansion)

        # Direct over-by-over run lookup - bypass AI
        import re as _re
        over_run_match = _re.search(r'(\d+)\+?\s*runs?\s*(against|off|vs|from)\s+([a-z\s\.]+?)\s*(in a single over|in an over|per over|in one over)', ql)
        if over_run_match or ('single over' in ql and any(w in ql for w in ['runs', 'scored', 'hit'])):
            bwl_col = dataset_info.get('bowler_col', 'bowler')
            bat_col = dataset_info.get('batter_col', 'batter')
            rc = dataset_info.get('runs_col', 'runs_total')
            if bwl_col in current_df.columns and bat_col in current_df.columns and rc in current_df.columns and 'over' in current_df.columns:
                # Find bowler name in query
                bowlers = current_df[bwl_col].unique()
                matched_bowler = None
                matched_bowler = fuzzy_find_player(ql, bowlers)
                # Find run threshold
                nums = _re.findall(r'\d+', ql)
                threshold = int(nums[0]) if nums else 15
                if matched_bowler:
                    match_col = 'match_id' if 'match_id' in current_df.columns else None
                    innings_col = 'innings_number' if 'innings_number' in current_df.columns else None
                    group_keys = [k for k in [match_col, innings_col, bat_col, bwl_col, 'over'] if k]
                    bowler_data = current_df[current_df[bwl_col] == matched_bowler].copy()
                    bowler_data[rc] = pd.to_numeric(bowler_data[rc], errors='coerce').fillna(0)
                    over_runs = bowler_data.groupby(group_keys)[rc].sum().reset_index(name='over_runs')
                    high_scoring = over_runs[over_runs['over_runs'] >= threshold].sort_values('over_runs', ascending=False)
                    if len(high_scoring) > 0:
                        results = []
                        for _, row in high_scoring.head(20).iterrows():
                            results.append(f"{row[bat_col]} ({int(row['over_runs'])} runs, over {int(row['over'])})")
                        answer = f"{len(high_scoring)} instances of batsmen scoring {threshold}+ runs against {matched_bowler} in a single over. Top instances: {', '.join(results)}."
                    else:
                        answer = f"No batsman scored {threshold}+ runs against {matched_bowler} in a single over in this dataset."
                    return jsonify({'success': True, 'answer': answer, 'chart': None})

        # Detect ranking vs single-player queries
        is_ranking_query = any(w in ql for w in ['top ', 'most ', 'highest ', 'best ', 'ranking', 'leaderboard', 'list all'])

        # Handle ranking dismissal/wicket queries directly
        import re as _re
        ranking_dismissal = (
            any(w in ql for w in ['top ', 'most ', 'highest ', 'best ']) and
            any(w in ql for w in ['lbw', 'caught', 'bowled', 'stumped', 'run out', 'dismissal', 'wicket'])
        )
        if ranking_dismissal and 'bowler_dismissals' in computed_metrics:
            bd = computed_metrics['bowler_dismissals']
            bwl_col = dataset_info.get('bowler_col', 'bowler')
            bwt_col = dataset_info.get('bowling_team', 'bowling_team')
            wk_col = 'dismissal_kind' if 'dismissal_kind' in bd.columns else None

            # Find top N
            top_n_match = _re.search(r'top\s+(\d+)', ql)
            top_n = int(top_n_match.group(1)) if top_n_match else 10

            # Find dismissal type filter
            dismissal_map = {
                'lbw': 'lbw', 'leg before': 'lbw',
                'caught and bowled': 'caught and bowled',
                'caught': 'caught', 'bowled': 'bowled',
                'stumped': 'stumped', 'run out': 'run out',
                'hit wicket': 'hit wicket',
            }
            dismissal_filter = None
            for term, kind in dismissal_map.items():
                if term in ql:
                    dismissal_filter = kind
                    break

            # Check for single innings query
            single_innings = any(w in ql for w in ['single innings', 'in an innings', 'in one innings', 'in a match', 'per innings'])

            # Check for team filter (e.g. "English bowlers", "Australian bowlers")
            team_filter = None
            if 'bowling' in computed_metrics and bwt_col in computed_metrics['bowling'].columns:
                teams = computed_metrics['bowling'][bwt_col].unique()
                for team in teams:
                    team_adj = team.lower().replace(' ', '')
                    for adj in [team.lower(), team_adj,
                                team.lower().rstrip('s'),
                                team.lower().replace('united states of america', 'american')]:
                        if adj in ql or (len(adj) > 4 and adj[:5] in ql):
                            team_filter = team
                            break
                    if team_filter:
                        break
                # Also check common adjectives
                adj_map = {
                    'english': 'England', 'australian': 'Australia', 'indian': 'India',
                    'pakistani': 'Pakistan', 'south african': 'South Africa',
                    'new zealand': 'New Zealand', 'sri lankan': 'Sri Lanka',
                    'west indian': 'West Indies', 'bangladeshi': 'Bangladesh',
                    'irish': 'Ireland', 'afghan': 'Afghanistan', 'zimbabwean': 'Zimbabwe',
                }
                for adj, team in adj_map.items():
                    if adj in ql:
                        team_filter = team
                        break

            if single_innings and 'bowling' in computed_metrics:
                # Best performance in a single innings
                bowl_df = computed_metrics['bowling'].copy()
                if team_filter and bwt_col in bowl_df.columns:
                    bowl_df = bowl_df[bowl_df[bwt_col] == team_filter]
                if 'wickets' in bowl_df.columns:
                    best = bowl_df.sort_values('wickets', ascending=False).head(top_n)
                    results = []
                    for _, row in best.iterrows():
                        match_info = f"match {row.get('match_id','')}" if 'match_id' in row else ''
                        results.append(f"{row[bwl_col]} ({int(row['wickets'])} wkts for {int(row['runs_conceded'])} runs {match_info})")
                    team_str = f" ({team_filter})" if team_filter else ""
                    answer = f"Top {top_n} bowling performances in a single innings{team_str}: {', '.join(results)}."
                    return jsonify({'success': True, 'answer': answer, 'chart': None})
            else:
                # Career/total ranking
                filter_df = bd.copy()
                if team_filter and bwt_col in filter_df.columns:
                    filter_df = filter_df[filter_df[bwt_col] == team_filter]
                if wk_col and dismissal_filter:
                    filter_df = filter_df[filter_df[wk_col].str.lower().str.contains(dismissal_filter, na=False)]
                    label = f"{dismissal_filter.upper()} dismissals"
                else:
                    label = "total dismissals"
                ranking = filter_df.groupby(bwl_col).size().sort_values(ascending=False).head(top_n)
                if len(ranking) > 0:
                    team_str = f" ({team_filter})" if team_filter else ""
                    result_list = ', '.join([f"{k} ({v})" for k, v in ranking.items()])
                    answer = f"Top {top_n} bowlers{team_str} by {label}: {result_list}."
                    return jsonify({'success': True, 'answer': answer, 'chart': None})

        # Direct dismissal lookup - handles specific bowler dismissal queries
        dismissal_trigger = (
            not is_ranking_query and (
                any(w in ql for w in ['dismissed by', 'who did', 'victims of', 'wickets by', 'batsmen dismissed by']) or
                (any(w in ql for w in ['taken by', 'how many', 'number of']) and
                 any(w in ql for w in ['dismiss', 'lbw', 'caught', 'stumped', 'run out']))
            )
        )
        if dismissal_trigger and 'bowler_dismissals' in computed_metrics:
            bd = computed_metrics['bowler_dismissals']
            bwl_col = dataset_info.get('bowler_col', 'bowler')
            wk_col = 'dismissal_kind' if 'dismissal_kind' in bd.columns else None

            if bwl_col in bd.columns and 'player_out' in bd.columns:
                bowlers = bd[bwl_col].unique()
                matched = fuzzy_find_player(ql, bowlers)
                if matched:
                    bowler_name = matched
                    bowler_data = bd[bd[bwl_col] == bowler_name]

                    # Check for specific dismissal type in query
                    dismissal_map = {
                        'lbw': 'lbw',
                        'leg before': 'lbw',
                        'caught and bowled': 'caught and bowled',
                        'caught': 'caught',
                        'bowled': 'bowled',
                        'stumped': 'stumped',
                        'run out': 'run out',
                        'hit wicket': 'hit wicket',
                    }
                    dismissal_filter = None
                    for term, kind in dismissal_map.items():
                        if term in ql:
                            dismissal_filter = kind
                            break

                    if dismissal_filter and wk_col:
                        filtered = bowler_data[
                            bowler_data[wk_col].str.lower().str.contains(dismissal_filter, na=False)
                        ]
                        total = len(filtered)
                        if total > 0:
                            victim_list = ', '.join([f"{k} ({v})" for k, v in
                                filtered['player_out'].value_counts().items()])
                            answer = f"{bowler_name} took {total} {dismissal_filter.upper()} dismissal(s). Batsmen: {victim_list}."
                        else:
                            answer = f"{bowler_name} has no {dismissal_filter.upper()} dismissals recorded in this dataset."
                    else:
                        victims = bowler_data['player_out'].value_counts()
                        total = len(bowler_data)
                        unique_victims = len(victims)
                        # Show breakdown by dismissal type
                        breakdown = ""
                        if wk_col and total > 0:
                            type_counts = bowler_data[wk_col].value_counts()
                            breakdown = " Breakdown: " + ", ".join([f"{k} {v}" for k, v in type_counts.items()]) + "."
                        victim_list = ', '.join([f"{k} ({v})" for k, v in victims.items()])
                        answer = f"{bowler_name} dismissed {total} batsmen ({unique_victims} unique).{breakdown} Full list: {victim_list}."

                    return jsonify({'success': True, 'answer': answer, 'chart': None})

            return jsonify({'success': True,
                           'answer': 'Dismissal records not available. The dataset may be missing the player_out column.',
                           'chart': None})

        filtered_df, year_note = filter_by_year(current_df.copy(), user_query)
        if year_note and len(filtered_df) > 0:
            year_metrics, _ = compute_all_metrics(filtered_df)
            active_metrics = year_metrics
            print(f"DEBUG: Year filter applied: {year_note}, rows: {len(filtered_df)}")
        else:
            active_metrics = computed_metrics
            year_note = None

        metrics_context = build_context(user_query, active_metrics, metric_notes, dataset_info)
        if year_note:
            metrics_context = f"DATA FILTERED FOR: {year_note}\n\n" + metrics_context

        # ================================================================
        # NEW ARCHITECTURE: Intent → Data → Answer + Chart (single source)
        # Step 1: Claude identifies WHAT to show (intent only, no numbers)
        # Step 2: Python fetches EXACT numbers from data
        # Step 3: Answer and chart both built from the same Python data
        # This guarantees chart and answer always match exactly
        # ================================================================

        cricket_context = ""
        if cricket_term_context:
            cricket_context = "CRICKET TERMINOLOGY:\n" + "\n".join(cricket_term_context)

        intent_prompt = f"""You are a cricket analytics intent classifier. Given a user question about cricket statistics, identify WHAT data to fetch. Do NOT invent or guess numbers.

{cricket_context}

Available metrics: {list(active_metrics.keys())}
Available columns - batters: {bat}, bowlers: {bwl}, batting_team: {bt}, runs: {rc}

USER QUESTION: {user_query}

Return ONLY valid JSON with this exact structure:
{{
  "metric": "one of: runs | wickets | economy | strike_rate | batting_avg | sixes | fours | dot_balls | maidens | powerplay_rr | middle_rr | death_rr | overall_rr | win_loss | centuries | half_centuries | ducks | run_outs | catches | keeper | powerplay_wickets | extras | hat_tricks | partnerships | best_bowling | dismissal_type",
  "dimension": "one of: player | team | bowler",
  "top_n": 5,
  "ascending": false,
  "chart_type": "bar | pie | line | none",
  "chart_title": "short descriptive title",
  "team_filter": null or "team name if query mentions a specific batting team (e.g. England batsmen, for India etc)",
  "opponent_filter": null or "team name if query says against X or vs X or bowling team is X",
  "dismissal_filter": null or "lbw | caught | bowled | stumped | run out"
}}"""

        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        intent_msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=300,
            messages=[{'role': 'user', 'content': intent_prompt}]
        )

        intent_text = intent_msg.content[0].text.strip()
        if '```' in intent_text:
            parts = intent_text.split('```')
            intent_text = parts[1] if len(parts) > 1 else parts[0]
            if intent_text.startswith('json'):
                intent_text = intent_text[4:]
        intent = json.loads(intent_text.strip())

        metric     = intent.get('metric', 'runs')
        dimension  = intent.get('dimension', 'player')
        top_n      = min(int(intent.get('top_n') or 10), 20)
        ascending  = bool(intent.get('ascending', False))
        chart_type = intent.get('chart_type', 'bar')
        # Always show a bar chart for ranking metrics unless explicitly none
        if chart_type == 'none' and metric not in ['win_loss', 'extras', 'hat_tricks']:
            chart_type = 'bar' 
        chart_title = intent.get('chart_title', metric.replace('_', ' ').title())
        team_filter = intent.get('team_filter')
        opponent_filter = intent.get('opponent_filter')
        dismissal_filter = intent.get('dismissal_filter')

        # Also detect opponent from query directly as backup
        if not opponent_filter:
            opp_indicators = ['against ', 'vs ', 'versus ', 'facing ', 'bowling team ']
            for ind in opp_indicators:
                if ind in ql:
                    # Get text after the indicator
                    after = ql.split(ind)[-1].strip().split()[0].strip('?.,')
                    if len(after) > 2:
                        # Try to match to a real team
                        if 'bowling' in active_metrics:
                            bwt_col = dataset_info.get('bowling_team', 'bowling_team')
                            if bwt_col in active_metrics['bowling'].columns:
                                teams = active_metrics['bowling'][bwt_col].unique()
                                for team in teams:
                                    if after.lower() in team.lower() or team.lower() in after.lower():
                                        opponent_filter = team
                                        break
                    break

        # ================================================================
        # Step 2: Python fetches the EXACT data based on intent
        # ================================================================
        result_df = None
        label_col = None
        value_col = None
        value_label = metric.replace('_', ' ').title()

        def apply_team_filter(df, col):
            if team_filter and col in df.columns:
                return df[df[col].str.lower().str.contains(team_filter.lower(), na=False)]
            return df

        if metric == 'runs' and 'batting' in active_metrics:
            df_m = apply_team_filter(active_metrics['batting'], bt)
            result_df = df_m.groupby(bat)['runs_scored'].sum().reset_index(name='runs')
            label_col, value_col = bat, 'runs'
            value_label = 'Runs'

        elif metric == 'wickets' and 'bowling' in active_metrics:
            df_m = apply_team_filter(active_metrics['bowling'], 'bowling_team' if 'bowling_team' in active_metrics['bowling'].columns else bt)
            if 'wickets' in df_m.columns:
                result_df = df_m.groupby(bwl)['wickets'].sum().reset_index(name='wickets')
                label_col, value_col = bwl, 'wickets'
                value_label = 'Wickets'

        elif metric == 'economy' and 'bowling' in active_metrics:
            df_m = apply_team_filter(active_metrics['bowling'], bt)
            eco = df_m.groupby(bwl).agg(
                runs=('runs_conceded','sum'), balls=('balls_bowled','sum')).reset_index()
            eco = eco[eco['balls'] >= 18]
            eco['economy'] = (eco['runs'] / (eco['balls']/6)).round(2)
            result_df = eco[[bwl, 'economy']]
            label_col, value_col = bwl, 'economy'
            ascending = True
            value_label = 'Economy Rate'

        elif metric == 'strike_rate' and 'batting' in active_metrics:
            df_m = apply_team_filter(active_metrics['batting'], bt)
            sr = df_m.groupby(bat).agg(
                runs=('runs_scored','sum'), balls=('balls_faced','sum')).reset_index()
            sr = sr[sr['balls'] >= 60]
            sr['strike_rate'] = (sr['runs']/sr['balls']*100).round(2)
            result_df = sr[[bat, 'strike_rate']]
            label_col, value_col = bat, 'strike_rate'
            value_label = 'Strike Rate'

        elif metric == 'batting_avg' and 'batting' in active_metrics:
            df_m = apply_team_filter(active_metrics['batting'], bt)
            inn = df_m.groupby(bat).size().reset_index(name='innings')
            tot = df_m.groupby(bat)['runs_scored'].sum().reset_index()
            avg = tot.merge(inn, on=bat)
            avg = avg[avg['innings'] >= 5]
            avg['avg'] = (avg['runs_scored']/avg['innings']).round(2)
            result_df = avg[[bat, 'avg']]
            label_col, value_col = bat, 'avg'
            value_label = 'Batting Average'

        elif metric == 'sixes' and 'batting' in active_metrics:
            df_m = apply_team_filter(active_metrics['batting'], bt)
            result_df = df_m.groupby(bat)['sixes'].sum().reset_index(name='sixes')
            label_col, value_col = bat, 'sixes'
            value_label = 'Sixes'

        elif metric == 'fours' and 'batting' in active_metrics:
            df_m = apply_team_filter(active_metrics['batting'], bt)
            result_df = df_m.groupby(bat)['fours'].sum().reset_index(name='fours')
            label_col, value_col = bat, 'fours'
            value_label = 'Fours'

        elif metric == 'dot_balls' and 'bowling' in active_metrics:
            df_m = apply_team_filter(active_metrics['bowling'], bt)
            result_df = df_m.groupby(bwl)['dot_balls_bowled'].sum().reset_index(name='dot_balls')
            label_col, value_col = bwl, 'dot_balls'
            value_label = 'Dot Balls'

        elif metric == 'maidens' and 'bowling' in active_metrics:
            df_m = apply_team_filter(active_metrics['bowling'], bt)
            result_df = df_m.groupby(bwl)['maiden_overs'].sum().reset_index(name='maidens')
            label_col, value_col = bwl, 'maidens'
            value_label = 'Maiden Overs'

        elif metric == 'powerplay_rr' and 'team_innings' in active_metrics:
            df_m = apply_team_filter(active_metrics['team_innings'], bt)
            result_df = df_m.groupby(bt)['powerplay_run_rate'].mean().round(2).reset_index(name='pp_rr')
            label_col, value_col = bt, 'pp_rr'
            value_label = 'Powerplay Run Rate'

        elif metric == 'middle_rr' and 'team_innings' in active_metrics:
            df_m = apply_team_filter(active_metrics['team_innings'], bt)
            result_df = df_m.groupby(bt)['middle_overs_run_rate'].mean().round(2).reset_index(name='mid_rr')
            label_col, value_col = bt, 'mid_rr'
            value_label = 'Middle Overs Run Rate'

        elif metric == 'death_rr' and 'team_innings' in active_metrics:
            df_m = apply_team_filter(active_metrics['team_innings'], bt)
            result_df = df_m.groupby(bt)['death_overs_run_rate'].mean().round(2).reset_index(name='death_rr')
            label_col, value_col = bt, 'death_rr'
            value_label = 'Death Overs Run Rate'

        elif metric == 'overall_rr' and 'team_innings' in active_metrics:
            df_m = apply_team_filter(active_metrics['team_innings'], bt)
            result_df = df_m.groupby(bt)['run_rate'].mean().round(2).reset_index(name='rr')
            label_col, value_col = bt, 'rr'
            value_label = 'Overall Run Rate'

        elif metric == 'win_loss' and 'win_loss' in active_metrics:
            df_m = apply_team_filter(active_metrics['win_loss'], bt)
            result_df = df_m[[bt, 'wins', 'losses', 'win_pct']].copy()
            label_col, value_col = bt, 'wins'
            value_label = 'Wins'

        elif metric == 'centuries' and 'milestones' in active_metrics:
            df_m = apply_team_filter(active_metrics['milestones'], bt)
            result_df = df_m.groupby(bat)['century'].sum().reset_index(name='centuries')
            result_df = result_df[result_df['centuries'] > 0]
            label_col, value_col = bat, 'centuries'
            value_label = 'Centuries'

        elif metric == 'half_centuries' and 'milestones' in active_metrics:
            df_m = apply_team_filter(active_metrics['milestones'], bt)
            result_df = df_m.groupby(bat)['half_century'].sum().reset_index(name='fifties')
            result_df = result_df[result_df['fifties'] > 0]
            label_col, value_col = bat, 'fifties'
            value_label = 'Half Centuries'

        elif metric == 'ducks' and 'ducks' in active_metrics:
            df_m = active_metrics['ducks']
            result_df = df_m.groupby(bat).size().reset_index(name='ducks')
            label_col, value_col = bat, 'ducks'
            value_label = 'Ducks'

        elif metric == 'run_outs' and 'run_outs' in active_metrics:
            df_m = active_metrics['run_outs']
            grp = bt if bt in df_m.columns else df_m.columns[0]
            result_df = df_m.groupby(grp)['run_outs'].sum().reset_index(name='run_outs')
            label_col, value_col = grp, 'run_outs'
            value_label = 'Run Outs'

        elif metric == 'catches' and 'catches' in active_metrics:
            df_m = active_metrics['catches']
            grp = bt if bt in df_m.columns else df_m.columns[0]
            result_df = df_m.groupby(grp)['catches'].sum().reset_index(name='catches')
            label_col, value_col = grp, 'catches'
            value_label = 'Catches'

        elif metric == 'keeper' and 'wicketkeeper_dismissals' in active_metrics:
            df_m = active_metrics['wicketkeeper_dismissals']
            grp = bt if bt in df_m.columns else df_m.columns[0]
            result_df = df_m.groupby(grp)['wicketkeeper_dismissals'].sum().reset_index(name='keeper_dismissals')
            label_col, value_col = grp, 'keeper_dismissals'
            value_label = 'Keeper Dismissals'

        elif metric == 'powerplay_wickets' and 'powerplay_wickets' in active_metrics:
            df_m = apply_team_filter(active_metrics['powerplay_wickets'], bt)
            grp_col = 'powerplay_wickets_lost' if 'powerplay_wickets_lost' in df_m.columns else df_m.columns[-1]
            result_df = df_m.groupby(bt)[grp_col].sum().reset_index(name='pp_wickets')
            label_col, value_col = bt, 'pp_wickets'
            value_label = 'Powerplay Wickets Lost'

        elif metric == 'hat_tricks' and 'hat_tricks' in active_metrics:
            ht = active_metrics['hat_tricks']
            if len(ht) > 0 and bwl in ht.columns:
                result_df = ht[[bwl]].copy()
                result_df['hat_tricks'] = 1
                result_df = result_df.groupby(bwl)['hat_tricks'].sum().reset_index()
                label_col, value_col = bwl, 'hat_tricks'
                value_label = 'Hat Tricks'

        elif metric == 'dismissal_type' and 'bowler_dismissals' in active_metrics:
            bd = active_metrics['bowler_dismissals']
            wk_col = 'dismissal_kind' if 'dismissal_kind' in bd.columns else None
            df_m = apply_team_filter(bd, bt)
            if dismissal_filter and wk_col:
                df_m = df_m[df_m[wk_col].str.lower().str.contains(dismissal_filter, na=False)]
                value_label = f'{dismissal_filter.upper()} Dismissals'
            if wk_col:
                result_df = df_m.groupby(bwl).size().reset_index(name='dismissals')
                label_col, value_col = bwl, 'dismissals'
            else:
                result_df = df_m.groupby(bwl).size().reset_index(name='dismissals')
                label_col, value_col = bwl, 'dismissals'

        elif metric == 'best_bowling' and 'bowling' in active_metrics:
            df_m = apply_team_filter(active_metrics['bowling'], bt)
            if 'wickets' in df_m.columns:
                result_df = df_m.sort_values(['wickets','runs_conceded'], ascending=[False, True]).head(top_n)
                result_df = result_df[[bwl, 'wickets', 'runs_conceded']].copy()
                result_df['figures'] = result_df['wickets'].astype(str) + '/' + result_df['runs_conceded'].astype(int).astype(str)
                label_col, value_col = bwl, 'wickets'
                value_label = 'Wickets (Best Bowling)'

        # Fallback to runs if metric not matched
        if result_df is None and 'batting' in active_metrics:
            result_df = active_metrics['batting'].groupby(bat)['runs_scored'].sum().reset_index(name='runs')
            label_col, value_col = bat, 'runs'
            value_label = 'Runs'

        # ================================================================
        # Step 3: Sort, select top N, build answer AND chart from same data
        # ================================================================
        chart_file = None
        answer = ''

        if result_df is not None and label_col and value_col and value_col in result_df.columns:
            result_df = result_df.copy()
            result_df[value_col] = pd.to_numeric(result_df[value_col], errors='coerce').fillna(0)
            result_df = result_df.dropna(subset=[label_col, value_col])
            result_df = result_df.sort_values(value_col, ascending=ascending).head(top_n)

            # Build answer directly from this data (no AI number generation)
            rows = []
            for _, row in result_df.iterrows():
                label = str(row[label_col])
                val = row[value_col]
                val_str = f"{val:.2f}" if isinstance(val, float) and val != int(val) else str(int(val)) if val == int(val) else str(val)
                rows.append(f"{label} ({val_str} {value_label.lower()})")

            team_str = f" for {team_filter}" if team_filter else ""
            opp_str = f" against {opponent_filter}" if opponent_filter else ""
            dismissal_str = f" ({dismissal_filter} only)" if dismissal_filter else ""
            answer = f"Top {len(result_df)} {value_label}{team_str}{opp_str}{dismissal_str}: {', '.join(rows)}."

            # Build chart from IDENTICAL data slice
            if chart_type != 'none' and len(result_df) > 0:
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.set_facecolor('#0d0d1f')

                    colors = ['#00d4ff','#00ff88','#ffaa00','#ff4466','#aa88ff',
                              '#ff88cc','#00ccaa','#ff8800','#88aaff','#ffff44']

                    if chart_type == 'bar':
                        bars = ax.bar(result_df[label_col].astype(str),
                                     result_df[value_col],
                                     color=[colors[i % len(colors)] for i in range(len(result_df))],
                                     edgecolor='#2a2a4e', linewidth=0.5)
                        # Add value labels on bars
                        for bar in bars:
                            h = bar.get_height()
                            val_str = f"{h:.1f}" if h != int(h) else str(int(h))
                            ax.text(bar.get_x() + bar.get_width()/2, h,
                                   val_str, ha='center', va='bottom',
                                   color='white', fontsize=8, fontweight='bold')

                    elif chart_type == 'line':
                        ax.plot(result_df[label_col].astype(str),
                               result_df[value_col],
                               color='#00d4ff', linewidth=2, marker='o', markersize=6)
                        for i, row in result_df.iterrows():
                            val_str = f"{row[value_col]:.1f}" if row[value_col] != int(row[value_col]) else str(int(row[value_col]))
                            ax.annotate(val_str,
                                       (str(row[label_col]), row[value_col]),
                                       textcoords='offset points', xytext=(0,8),
                                       ha='center', color='white', fontsize=8)

                    elif chart_type == 'pie':
                        wedge_colors = colors[:len(result_df)]
                        ax.pie(result_df[value_col],
                              labels=result_df[label_col].astype(str),
                              autopct='%1.1f%%',
                              colors=wedge_colors,
                              startangle=90)
                        ax.axis('equal')

                    ax.set_title(chart_title, color='white', fontsize=13, pad=15)
                    ax.tick_params(colors='#aaaaaa', labelsize=9)
                    plt.xticks(rotation=30, ha='right', color='#aaaaaa')
                    if chart_type != 'pie':
                        ax.set_ylabel(value_label, color='#aaaaaa', fontsize=10)
                        ax.spines['bottom'].set_color('#2a2a4e')
                        ax.spines['left'].set_color('#2a2a4e')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    chart_file = f'chart_{uuid.uuid4().hex[:8]}.png'
                    plt.savefig(os.path.join(STATIC_FOLDER, chart_file),
                               facecolor='#1a1a2e', dpi=120)
                    plt.close()
                except Exception as chart_err:
                    print(f"DEBUG: Chart error: {chart_err}")
                    chart_file = None
        else:
            answer = f"The requested statistic ({metric}) is not available in the current dataset."

        return jsonify({'success': True, 'answer': answer, 'chart': chart_file})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/dashboard-data')
def dashboard_data():
    global current_df, computed_metrics, dataset_info, metric_notes
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded. Please upload a dataset first.'})
    try:
        bat = dataset_info.get('batter_col', 'batter')
        bwl = dataset_info.get('bowler_col', 'bowler')
        bt  = dataset_info.get('batting_team', 'batting_team')
        bwt = dataset_info.get('bowling_team', 'bowling_team')
        rc  = dataset_info.get('runs_col', 'runs_total')
        m   = computed_metrics

        result = {'success': True, 'dataset_name': f"{len(current_df):,} rows loaded"}

        # KPIs
        kpi = {}
        kpi['matches'] = current_df['match_id'].nunique() if 'match_id' in current_df.columns else '—'
        kpi['total_runs'] = int(current_df[rc].sum()) if rc in current_df.columns else 0
        kpi['teams'] = current_df[bt].nunique() if bt in current_df.columns else '—'
        kpi['batsmen'] = current_df[bat].nunique() if bat in current_df.columns else '—'
        kpi['bowlers'] = current_df[bwl].nunique() if bwl in current_df.columns else '—'

        if 'bowling' in m and 'wickets' in m['bowling'].columns:
            kpi['total_wickets'] = int(m['bowling']['wickets'].sum())
            top_b = m['bowling'].groupby(bwl)['wickets'].sum().sort_values(ascending=False)
            kpi['top_bowler_name'] = top_b.index[0].split()[-1] if len(top_b) > 0 else '—'
            kpi['top_bowler_wickets'] = int(top_b.iloc[0]) if len(top_b) > 0 else 0
        else:
            kpi['total_wickets'] = 0
            kpi['top_bowler_name'] = '—'
            kpi['top_bowler_wickets'] = 0

        if 'batting' in m:
            top_bat_runs = m['batting'].groupby(bat)['runs_scored'].sum().sort_values(ascending=False)
            kpi['top_scorer_name'] = top_bat_runs.index[0].split()[-1] if len(top_bat_runs) > 0 else '—'
            kpi['top_scorer_runs'] = int(top_bat_runs.iloc[0]) if len(top_bat_runs) > 0 else 0
            kpi['total_sixes'] = int(m['batting']['sixes'].sum())
            kpi['total_fours'] = int(m['batting']['fours'].sum())
        else:
            kpi['top_scorer_name'] = '—'
            kpi['top_scorer_runs'] = 0
            kpi['total_sixes'] = 0
            kpi['total_fours'] = 0

        if 'milestones' in m:
            kpi['centuries'] = int(m['milestones']['century'].sum())
            kpi['half_centuries'] = int(m['milestones']['half_century'].sum())
        else:
            kpi['centuries'] = 0
            kpi['half_centuries'] = 0

        result['kpi'] = kpi

        # Top batsmen
        if 'batting' in m:
            top10 = m['batting'].groupby(bat)['runs_scored'].sum().sort_values(ascending=False).head(10)
            result['top_batsmen'] = [{'name': k, 'runs': int(v)} for k, v in top10.items()]

            # Strike rate
            sr = m['batting'].groupby(bat).agg(
                runs=('runs_scored','sum'), balls=('balls_faced','sum')).reset_index()
            sr['sr'] = (sr['runs'] / sr['balls'] * 100).round(1)
            sr = sr[sr['balls'] >= 60].sort_values('sr', ascending=False).head(10)
            result['top_sr'] = [{'name': r[bat], 'sr': r['sr']} for _, r in sr.iterrows()]

            # Sixes and fours
            top_six = m['batting'].groupby(bat)['sixes'].sum().sort_values(ascending=False).head(10)
            result['top_sixes'] = [{'name': k, 'sixes': int(v)} for k, v in top_six.items()]

            top_four = m['batting'].groupby(bat)['fours'].sum().sort_values(ascending=False).head(10)
            result['top_fours'] = [{'name': k, 'fours': int(v)} for k, v in top_four.items()]

            # Batting average
            inn_c = m['batting'].groupby(bat).size().reset_index(name='innings')
            tot_r = m['batting'].groupby(bat)['runs_scored'].sum().reset_index()
            avg_df = tot_r.merge(inn_c, on=bat)
            avg_df['avg'] = (avg_df['runs_scored'] / avg_df['innings']).round(2)
            avg_df = avg_df[avg_df['innings'] >= 10].sort_values('avg', ascending=False).head(10)
            result['batting_avg'] = [{'name': r[bat], 'avg': r['avg']} for _, r in avg_df.iterrows()]

        # Top bowlers
        if 'bowling' in m and 'wickets' in m['bowling'].columns:
            top10b = m['bowling'].groupby(bwl)['wickets'].sum().sort_values(ascending=False).head(10)
            result['top_bowlers'] = [{'name': k, 'wickets': int(v)} for k, v in top10b.items()]

            # Economy
            eco = m['bowling'].groupby(bwl).agg(
                tr=('runs_conceded','sum'), tb=('balls_bowled','sum'),
                dots=('dot_balls_bowled','sum'), maidens=('maiden_overs','sum')
            ).reset_index()
            eco['eco'] = (eco['tr'] / (eco['tb'] / 6)).round(2)
            eco = eco[eco['tb'] >= 18].sort_values('eco').head(10)
            result['top_eco'] = [{'name': r[bwl], 'eco': r['eco']} for _, r in eco.iterrows()]

            # Dot balls
            top_dots = eco.sort_values('dots', ascending=False).head(10)
            result['top_dots'] = [{'name': r[bwl], 'dots': int(r['dots'])} for _, r in top_dots.iterrows()]

            # Maidens
            top_maidens = eco.sort_values('maidens', ascending=False).head(10)
            result['top_maidens'] = [{'name': r[bwl], 'maidens': int(r['maidens'])} for _, r in top_maidens.iterrows()]

        # Team runs and wickets
        if rc in current_df.columns and bt in current_df.columns:
            team_runs = current_df.groupby(bt)[rc].sum().sort_values(ascending=False)
            result['team_runs'] = [{'team': k, 'runs': int(v)} for k, v in team_runs.items()]

        if 'bowling' in m and 'wickets' in m['bowling'].columns and bwt in m['bowling'].columns:
            team_wkts = m['bowling'].groupby(bwt)['wickets'].sum().sort_values(ascending=False)
            result['team_wickets'] = [{'team': k, 'wickets': int(v)} for k, v in team_wkts.items()]

        # Phase run rates
        if 'team_innings' in m:
            ti = m['team_innings']
            phase = ti.groupby(bt).agg(
                pp=('powerplay_run_rate','mean'),
                mid=('middle_overs_run_rate','mean'),
                death=('death_overs_run_rate','mean')
            ).reset_index()
            phase = phase.sort_values('pp', ascending=False)
            result['phase_rr'] = [{'team': r[bt], 'pp': round(r['pp'],2), 'mid': round(r['mid'],2), 'death': round(r['death'],2)}
                                   for _, r in phase.iterrows()]

        # Win loss
        if 'win_loss' in m:
            wl = m['win_loss'].sort_values('wins', ascending=False)
            result['win_loss'] = [{'team': r[bt], 'wins': int(r['wins']), 'losses': int(r['losses'])}
                                   for _, r in wl.iterrows()]

        # Dismissal types
        if 'dismissal_kind' in current_df.columns:
            dt = current_df['dismissal_kind'].dropna()
            dt = dt[dt != ''].value_counts().head(8)
            result['dismissal_types'] = [{'type': k, 'count': int(v)} for k, v in dt.items()]

        # Powerplay wickets
        if 'powerplay_wickets' in m and bt in m['powerplay_wickets'].columns:
            pp_wk = m['powerplay_wickets'].groupby(bt)['powerplay_wickets_lost'].sum().sort_values(ascending=False)
            result['pp_wickets'] = [{'team': k, 'wickets': int(v)} for k, v in pp_wk.items()]

        # Clean NaN values before sending to JavaScript
        import math
        def clean_nan(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_nan(i) for i in obj]
            return obj

        result = clean_nan(result)
        return jsonify(result)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/lookup', methods=['POST'])
def lookup():
    global current_df, computed_metrics, dataset_info
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    try:
        data = request.get_json()
        query = data.get('query', '').lower()
        bwl = dataset_info.get('bowler_col', 'bowler')
        bat = dataset_info.get('batter_col', 'batter')

        # Direct lookup: who did bowler X dismiss?
        if 'bowler_dismissals' in computed_metrics:
            bd = computed_metrics['bowler_dismissals']
            # Find bowler name mentioned in query
            if bwl in bd.columns and 'player_out' in bd.columns:
                bowlers = bd[bwl].unique()
                matched = [b for b in bowlers if b.lower() in query or
                          any(part in query for part in b.lower().split())]
                if matched:
                    bowler_name = matched[0]
                    victims = bd[bd[bwl] == bowler_name]['player_out'].value_counts().head(20)
                    return jsonify({
                        'success': True,
                        'bowler': bowler_name,
                        'total_wickets': len(bd[bd[bwl] == bowler_name]),
                        'victims': victims.to_dict()
                    })
        return jsonify({'success': False, 'error': 'Could not find matching bowler'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
