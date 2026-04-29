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
    key_cols = [c for c in ['match_id', 'innings_number', 'over', 'ball_str'] if c in df.columns]
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
                               'detail': f'{has_spaces} cells have leading/trailing spaces'})
    if not issues:
        issues.append({'type': 'Data looks clean!',
                       'detail': 'No significant issues found. Ready to query.'})
    return issues


def apply_cleaning(df):
    cleaned = df.copy()
    if 'date' in cleaned.columns:
        cleaned['year'] = pd.to_datetime(cleaned['date'], errors='coerce').dt.year.astype('Int64')
    key_cols = [c for c in ['match_id', 'innings_number', 'over', 'ball_str'] if c in cleaned.columns]
    if len(key_cols) >= 3:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=key_cols, keep='first')
        print(f"DEBUG: Removed {before - len(cleaned)} duplicate rows")
    else:
        cleaned = cleaned.drop_duplicates()
    str_cols = cleaned.select_dtypes(include=['object']).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    num_cols = cleaned.select_dtypes(include=['number']).columns
    for col in num_cols:
        if cleaned[col].isnull().sum() > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
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

    if any(w in ql for w in ['century', 'hundred', 'half century', 'fifty', '50', '100']):
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
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': 'Only CSV and Excel files are supported'})
        if 'date' in df.columns:
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year.astype('Int64')
        current_df = df.copy()
        original_df = df.copy()
        dataset_info = analyse_dataset(df)
        computed_metrics, metric_notes = compute_all_metrics(df)
        issues = detect_issues(df)
        preview = df.head(5).fillna('').to_dict(orient='records')
        summary = get_metric_summary(computed_metrics, metric_notes)
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'issues': issues,
            'preview': preview,
            'metric_summary': summary,
            'unavailable_stats': list(metric_notes.keys())
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/clean', methods=['POST'])
def clean():
    global current_df, computed_metrics, metric_notes
    if current_df is None:
        return jsonify({'success': False, 'error': 'No data loaded'})
    try:
        data = request.get_json()
        if data.get('action') == 'apply':
            current_df = apply_cleaning(current_df)
            computed_metrics, metric_notes = compute_all_metrics(current_df)
            preview = current_df.head(5).fillna('').to_dict(orient='records')
            return jsonify({'success': True, 'preview': preview})
        else:
            return jsonify({'success': False, 'error': 'Unknown action'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


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

        rc  = dataset_info.get('runs_col', 'runs_total')
        bt  = dataset_info.get('batting_team', 'batting_team')
        bwt = dataset_info.get('bowling_team', 'bowling_team')
        bat = dataset_info.get('batter_col', 'batter')
        bwl = dataset_info.get('bowler_col', 'bowler')
        ql  = user_query.lower()

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
                for b in bowlers:
                    parts = [p for p in b.lower().split() if len(p) > 1]
                    if all(p in ql for p in parts):
                        matched_bowler = b
                        break
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

        # Direct dismissal lookup - bypass AI for this query type
        if any(w in ql for w in ['dismissed by', 'who did', 'victims of', 'wickets by', 'batsmen dismissed by']):
            if 'bowler_dismissals' in computed_metrics:
                bd = computed_metrics['bowler_dismissals']
                bwl_col = dataset_info.get('bowler_col', 'bowler')
                if bwl_col in bd.columns and 'player_out' in bd.columns:
                    bowlers = bd[bwl_col].unique()
                    # Prefer exact full name match first, then partial
                    exact = [b for b in bowlers if b.lower() == ql.split('by ')[-1].strip().rstrip('?')]
                    partial = [b for b in bowlers if b.lower() in ql]
                    # Match initials like "sm curran" -> "SM Curran"
                    initials = [b for b in bowlers if all(
                        part in b.lower() for part in ql.split() if len(part) > 1
                    )]
                    matched = exact or partial or initials
                    if matched:
                        bowler_name = matched[0]
                        bowler_data = bd[bd[bwl_col] == bowler_name]
                        victims = bowler_data['player_out'].value_counts()
                        total = len(bowler_data)
                        unique_victims = len(victims)
                        victim_list = ', '.join([f"{k} ({v})" for k, v in victims.items()])
                        answer = f"{bowler_name} dismissed {total} batsmen ({unique_victims} unique) in this dataset. Full list: {victim_list}."
                        return jsonify({'success': True, 'answer': answer, 'chart': None})
            return jsonify({'success': True,
                           'answer': 'Dismissal records are not available. The dataset may be missing the player_out column.',
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

        chart_df = None
        chart_x = None
        chart_y = None

        if any(w in ql for w in ['economy']):
            if 'bowling' in active_metrics:
                eco_chart = active_metrics['bowling'].groupby(bwl).agg(
                    total_runs=('runs_conceded', 'sum'), total_balls=('balls_bowled', 'sum')
                ).reset_index()
                eco_chart['economy_rate'] = (eco_chart['total_runs'] / (eco_chart['total_balls'] / 6)).round(2)
                chart_df = eco_chart[eco_chart['total_balls'] >= 18]
                chart_x, chart_y = bwl, 'economy_rate'
        elif any(w in ql for w in ['dot ball']):
            if 'bowling' in active_metrics:
                chart_df = active_metrics['bowling'].groupby(bwl).agg(dot_balls_bowled=('dot_balls_bowled', 'sum')).reset_index()
                chart_x, chart_y = bwl, 'dot_balls_bowled'
        elif any(w in ql for w in ['maiden']):
            if 'bowling' in active_metrics:
                chart_df = active_metrics['bowling'].groupby(bwl).agg(maiden_overs=('maiden_overs', 'sum')).reset_index()
                chart_x, chart_y = bwl, 'maiden_overs'
        elif any(w in ql for w in ['wicket', 'bowler', 'bowling']):
            if 'bowling' in active_metrics:
                chart_df = active_metrics['bowling'].groupby(bwl)['wickets'].sum().reset_index()
                chart_x, chart_y = bwl, 'wickets'
        elif any(w in ql for w in ['powerplay']):
            if 'team_innings' in active_metrics:
                chart_df = active_metrics['team_innings'].groupby(bt)['powerplay_run_rate'].mean().reset_index()
                chart_x, chart_y = bt, 'powerplay_run_rate'
        elif any(w in ql for w in ['middle over']):
            if 'team_innings' in active_metrics:
                chart_df = active_metrics['team_innings'].groupby(bt)['middle_overs_run_rate'].mean().reset_index()
                chart_x, chart_y = bt, 'middle_overs_run_rate'
        elif any(w in ql for w in ['death over']):
            if 'team_innings' in active_metrics:
                chart_df = active_metrics['team_innings'].groupby(bt)['death_overs_run_rate'].mean().reset_index()
                chart_x, chart_y = bt, 'death_overs_run_rate'
        elif any(w in ql for w in ['run rate']):
            if 'team_innings' in active_metrics:
                chart_df = active_metrics['team_innings'].groupby(bt)['run_rate'].mean().reset_index()
                chart_x, chart_y = bt, 'run_rate'
        elif any(w in ql for w in ['six']):
            if 'batting' in active_metrics:
                chart_df = active_metrics['batting'].groupby(bat)['sixes'].sum().reset_index()
                chart_x, chart_y = bat, 'sixes'
        elif any(w in ql for w in ['four', 'boundary']):
            if 'batting' in active_metrics:
                chart_df = active_metrics['batting'].groupby(bat)['fours'].sum().reset_index()
                chart_x, chart_y = bat, 'fours'
        elif any(w in ql for w in ['strike rate']):
            if 'batting' in active_metrics:
                chart_df = active_metrics['batting'].groupby(bat)['strike_rate'].mean().reset_index()
                chart_x, chart_y = bat, 'strike_rate'
        elif any(w in ql for w in ['average', 'per innings', 'innings average']):
            if 'batting' in active_metrics:
                innings_c = active_metrics['batting'].groupby(bat).size().reset_index(name='innings')
                total_r = active_metrics['batting'].groupby(bat)['runs_scored'].sum().reset_index()
                avg_chart = total_r.merge(innings_c, on=bat)
                avg_chart['avg_per_innings'] = (avg_chart['runs_scored'] / avg_chart['innings']).round(2)
                chart_df = avg_chart[avg_chart['innings'] >= 5]
                chart_x, chart_y = bat, 'avg_per_innings'
        elif any(w in ql for w in ['win', 'loss']):
            if 'win_loss' in active_metrics:
                chart_df = active_metrics['win_loss']
                chart_x, chart_y = bt, 'wins'
        else:
            if 'batting' in active_metrics:
                chart_df = active_metrics['batting'].groupby(bat)['runs_scored'].sum().reset_index()
                chart_x, chart_y = bat, 'runs_scored'

        prompt = f"""You are a cricket analyst AI with access to pre-computed match statistics from ball-by-ball data.

{metrics_context}

IMPORTANT NOTES:
- Batting metrics ARE calculated per inning per match. You CAN answer per-innings questions.
- For hat tricks AGAINST a team, batting_team = that team (they were batting when the hat trick was taken).
- If a stat is unavailable, say so clearly with the reason.

USER QUESTION: {user_query}

Return ONLY valid JSON:
{{
  "answer": "specific 2-3 sentence answer with real names and numbers",
  "chart_type": "bar" or "line" or "pie" or "none",
  "chart_title": "descriptive chart title",
  "top_n": 10
}}"""

        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=400,
            messages=[{'role': 'user', 'content': prompt}]
        )

        response_text = message.content[0].text.strip()
        if '```' in response_text:
            parts = response_text.split('```')
            response_text = parts[1] if len(parts) > 1 else parts[0]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()

        instructions = json.loads(response_text)
        answer = instructions.get('answer', 'Here is your result.')
        chart_type = instructions.get('chart_type', 'none')
        top_n = min(int(instructions.get('top_n') or 10), 15)
        chart_file = None

        if chart_type != 'none' and chart_df is not None and chart_x and chart_y:
            if chart_x in chart_df.columns and chart_y in chart_df.columns:
                chart_df = chart_df.copy()
                chart_df[chart_y] = pd.to_numeric(chart_df[chart_y], errors='coerce').fillna(0)
                ascending = True if chart_y == 'economy_rate' else False
                chart_df = chart_df.sort_values(chart_y, ascending=ascending).head(top_n)

                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#1a1a2e')
                ax.set_facecolor('#0d0d1f')

                if chart_type == 'bar':
                    ax.bar(chart_df[chart_x].astype(str), chart_df[chart_y], color='#00d4ff', edgecolor='#2a2a4e')
                elif chart_type == 'line':
                    ax.plot(chart_df[chart_x].astype(str), chart_df[chart_y], color='#00d4ff', linewidth=2, marker='o')
                elif chart_type == 'pie':
                    ax.pie(chart_df[chart_y], labels=chart_df[chart_x].astype(str), autopct='%1.1f%%',
                           colors=['#00d4ff', '#00ff88', '#ffaa00', '#ff4444', '#aa88ff',
                                   '#ff88aa', '#88ffaa', '#aaaaff', '#ffff00', '#ff8800'])
                    ax.axis('equal')

                ax.set_title(instructions.get('chart_title', 'Chart'), color='white', fontsize=13, pad=15)
                ax.tick_params(colors='#aaaaaa', labelsize=9)
                plt.xticks(rotation=30, ha='right', color='#aaaaaa')
                ax.spines['bottom'].set_color('#2a2a4e')
                ax.spines['left'].set_color('#2a2a4e')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                chart_file = f'chart_{uuid.uuid4().hex[:8]}.png'
                plt.savefig(os.path.join(STATIC_FOLDER, chart_file), facecolor='#1a1a2e', dpi=120)
                plt.close()

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
