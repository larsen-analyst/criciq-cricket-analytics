"""
CricIQ YAML to CSV Converter
Supports both old (pre-2017) and new (post-2017) Cricsheet YAML formats.
"""
import yaml
import pandas as pd
import zipfile
import io
import os
import re


def parse_single_yaml(yaml_content, match_id=None):
    """Parse a Cricsheet YAML file - handles both old and new formats."""
    try:
        try:
            from yaml import CLoader as Loader
            data = yaml.load(yaml_content, Loader=Loader)
        except ImportError:
            data = yaml.safe_load(yaml_content)
    except Exception as e:
        return [], f"YAML parse error: {e}"

    if not data or 'info' not in data or 'innings' not in data:
        return [], "Invalid YAML: missing info or innings"

    info = data['info']
    rows = []

    # Extract match info
    dates = info.get('dates', [])
    match_date = str(dates[0]) if dates else ''
    city = info.get('city', '')
    venue = info.get('venue', '')
    match_type = info.get('match_type', '')
    gender = info.get('gender', 'male')
    teams = info.get('teams', [])
    team1 = teams[0] if len(teams) > 0 else ''
    team2 = teams[1] if len(teams) > 1 else ''

    outcome = info.get('outcome', {})
    winner = outcome.get('winner', outcome.get('result', ''))

    toss = info.get('toss', {})
    toss_winner = toss.get('winner', '')
    toss_decision = toss.get('decision', '')

    pom = info.get('player_of_match', [])
    player_of_match = ', '.join(pom) if isinstance(pom, list) else str(pom or '')

    if not match_id:
        match_id = f"{match_date}_{team1}_vs_{team2}".replace(' ', '_')

    # Detect format
    innings_list = data.get('innings', [])
    is_new_format = _is_new_format(innings_list)

    for innings_num, innings_data in enumerate(innings_list, 1):
        if is_new_format:
            rows += _parse_new_innings(
                innings_data, innings_num, match_id, match_date,
                city, venue, match_type, gender, team1, team2,
                winner, toss_winner, toss_decision, player_of_match
            )
        else:
            rows += _parse_old_innings(
                innings_data, innings_num, match_id, match_date,
                city, venue, match_type, gender, team1, team2,
                winner, toss_winner, toss_decision, player_of_match
            )

    return rows, None


def _is_new_format(innings_list):
    """Detect if this is new format (has 'team' key) or old format (has '1st innings' key)."""
    if not innings_list:
        return True
    first = innings_list[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        return 'team' in keys or 'overs' in keys
    return True


def _parse_new_innings(innings_data, innings_num, match_id, match_date,
                        city, venue, match_type, gender, team1, team2,
                        winner, toss_winner, toss_decision, player_of_match):
    """Parse new Cricsheet format (post-2017)."""
    rows = []
    batting_team = innings_data.get('team', '')
    bowling_team = team2 if batting_team == team1 else team1

    for over_data in innings_data.get('overs', []):
        over_num = over_data.get('over', 0)
        over_1indexed = over_num + 1

        for ball_num, delivery in enumerate(over_data.get('deliveries', []), 1):
            row = _build_row(
                delivery, match_id, match_date, city, venue, match_type,
                gender, team1, team2, winner, toss_winner, toss_decision,
                player_of_match, innings_num, batting_team, bowling_team,
                over_1indexed, ball_num,
                batter_key='batter', runs_batter_key='batter'
            )
            rows.append(row)

    return rows


def _parse_old_innings(innings_data, innings_num, match_id, match_date,
                        city, venue, match_type, gender, team1, team2,
                        winner, toss_winner, toss_decision, player_of_match):
    """Parse old Cricsheet format (pre-2017). Keys like '1st innings:', deliveries as 0.1, 0.2 etc."""
    rows = []

    # Get the innings content - key is like '1st innings' or '2nd innings'
    innings_content = None
    for key, val in innings_data.items():
        if isinstance(val, dict) and ('team' in val or 'deliveries' in val):
            innings_content = val
            break

    if innings_content is None:
        return rows

    batting_team = innings_content.get('team', '')
    bowling_team = team2 if batting_team == team1 else team1

    deliveries = innings_content.get('deliveries', [])

    for delivery_item in deliveries:
        if not isinstance(delivery_item, dict):
            continue

        for ball_key, delivery in delivery_item.items():
            if not isinstance(delivery, dict):
                continue

            # Parse ball key like "0.1", "1.3" -> over and ball number
            try:
                ball_float = float(str(ball_key))
                over_1indexed = int(ball_float) + 1
                ball_num = round((ball_float - int(ball_float)) * 10)
                if ball_num == 0:
                    ball_num = 1
            except (ValueError, TypeError):
                over_1indexed = 1
                ball_num = 1

            row = _build_row(
                delivery, match_id, match_date, city, venue, match_type,
                gender, team1, team2, winner, toss_winner, toss_decision,
                player_of_match, innings_num, batting_team, bowling_team,
                over_1indexed, ball_num,
                batter_key='batsman', runs_batter_key='batsman'
            )
            rows.append(row)

    return rows


def _build_row(delivery, match_id, match_date, city, venue, match_type,
               gender, team1, team2, winner, toss_winner, toss_decision,
               player_of_match, innings_num, batting_team, bowling_team,
               over_1indexed, ball_num, batter_key='batter', runs_batter_key='batter'):
    """Build a single delivery row dict."""

    batter = delivery.get(batter_key, delivery.get('batter', ''))
    bowler = delivery.get('bowler', '')
    non_striker = delivery.get('non_striker', '')

    runs = delivery.get('runs', {})
    runs_batter = runs.get(runs_batter_key, runs.get('batter', 0))
    runs_extras = runs.get('extras', 0)
    runs_total = runs.get('total', 0)

    extras = delivery.get('extras', {})
    wides = extras.get('wides', 0)
    noballs = extras.get('noballs', 0)
    byes = extras.get('byes', 0)
    legbyes = extras.get('legbyes', 0)
    extras_detail = str(extras) if extras else ''

    # Wickets - handle both old and new format
    wickets = delivery.get('wickets', [])
    # Old format sometimes uses 'wicket' singular
    if not wickets and 'wicket' in delivery:
        wickets = [delivery['wicket']]

    dismissal_kind = ''
    player_out = ''
    fielders = ''
    wicket_flag = 0

    if wickets:
        wicket_flag = 1
        w = wickets[0] if isinstance(wickets, list) else wickets
        if isinstance(w, dict):
            dismissal_kind = w.get('kind', '')
            player_out = w.get('player_out', '')
            fielder_list = w.get('fielders', [])
            if fielder_list:
                fielders = ', '.join([
                    f.get('name', str(f)) if isinstance(f, dict) else str(f)
                    for f in fielder_list
                ])

    is_legal_ball = 1 if (wides == 0 and noballs == 0) else 0
    is_boundary = 1 if runs_batter in [4, 6] else 0
    is_dot_ball = 1 if (runs_total == 0 and is_legal_ball) else 0
    ball_str = round((over_1indexed - 1) + ball_num / 10, 1)

    if over_1indexed <= 6:
        phase = 'powerplay'
    elif over_1indexed <= 15:
        phase = 'middle'
    else:
        phase = 'death'

    return {
        'match_id': match_id,
        'date': match_date,
        'city': city,
        'venue': venue,
        'match_type': match_type,
        'gender': gender,
        'team1': team1,
        'team2': team2,
        'teams': f"{team1} vs {team2}",
        'winner': winner,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'player_of_match': player_of_match,
        'innings_number': innings_num,
        'innings_name': f"{'1st' if innings_num == 1 else '2nd'} innings",
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'over': over_1indexed,
        'ball_in_over': ball_num,
        'ball_str': ball_str,
        'batter': batter,
        'bowler': bowler,
        'non_striker': non_striker,
        'runs_total': runs_total,
        'runs_batter': runs_batter,
        'runs_extras': runs_extras,
        'extras_detail': extras_detail,
        'wides': wides,
        'noballs': noballs,
        'byes': byes,
        'legbyes': legbyes,
        'is_legal_ball': is_legal_ball,
        'is_boundary': is_boundary,
        'is_dot_ball': is_dot_ball,
        'wicket_flag': wicket_flag,
        'dismissal_kind': dismissal_kind,
        'player_out': player_out,
        'fielders': fielders,
        'phase': phase,
    }


def _quick_match_filter(yaml_content, team_filter, year_from, year_to):
    """Quick pre-filter without full YAML parse."""
    try:
        lines = yaml_content[:2000]
        if team_filter:
            if team_filter.lower() not in lines.lower():
                return False
        if year_from or year_to:
            dates = re.findall(r'(\d{4})-\d{2}-\d{2}', lines)
            if dates:
                year = int(dates[0])
                if year_from and year < year_from:
                    return False
                if year_to and year > year_to:
                    return False
        return True
    except Exception:
        return True


def convert_yaml_to_df(file_content, filename, team_filter=None, year_from=None, year_to=None):
    """
    Convert YAML or ZIP of YAMLs to clean DataFrame.
    Supports both old and new Cricsheet formats.
    """
    all_rows = []
    errors = []
    match_count = 0
    skipped = 0

    team_filter = team_filter.strip().lower() if team_filter else None
    year_from = int(year_from) if year_from else None
    year_to = int(year_to) if year_to else None

    filename_lower = filename.lower()

    if filename_lower.endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
                yaml_files = [f for f in zf.namelist()
                             if f.lower().endswith('.yaml') or f.lower().endswith('.yml')]

                if not yaml_files:
                    return None, "No YAML files found in ZIP", 0

                print(f"DEBUG: Found {len(yaml_files)} YAML files in ZIP")

                for yaml_file in yaml_files:
                    try:
                        match_id = os.path.basename(yaml_file).replace('.yaml','').replace('.yml','')
                        yaml_content = zf.read(yaml_file).decode('utf-8')

                        if team_filter or year_from or year_to:
                            if not _quick_match_filter(yaml_content, team_filter, year_from, year_to):
                                skipped += 1
                                continue

                        rows, err = parse_single_yaml(yaml_content, match_id)
                        if err:
                            errors.append(f"{yaml_file}: {err}")
                        else:
                            all_rows.extend(rows)
                            match_count += 1
                    except Exception as e:
                        errors.append(f"{yaml_file}: {e}")

                print(f"DEBUG: Processed {match_count} matches, skipped {skipped}, errors {len(errors)}")

        except zipfile.BadZipFile:
            return None, "Invalid ZIP file", 0

    elif filename_lower.endswith('.yaml') or filename_lower.endswith('.yml'):
        try:
            yaml_content = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
            match_id = filename.replace('.yaml','').replace('.yml','')
            rows, err = parse_single_yaml(yaml_content, match_id)
            if err:
                return None, err, 0
            all_rows.extend(rows)
            match_count = 1
        except Exception as e:
            return None, str(e), 0
    else:
        return None, "Unsupported file type", 0

    if not all_rows:
        msg = "No delivery data found in YAML files"
        if errors:
            msg += f". First error: {errors[0]}"
        return None, msg, 0

    df = pd.DataFrame(all_rows)

    int_cols = ['innings_number', 'over', 'ball_in_over', 'runs_total',
                'runs_batter', 'runs_extras', 'wides', 'noballs', 'byes',
                'legbyes', 'is_legal_ball', 'is_boundary', 'is_dot_ball', 'wicket_flag']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    print(f"DEBUG: Final dataset: {match_count} matches, {len(df)} rows")
    return df, None, match_count


def get_metric_summary(metrics, notes):
    parts = []
    if 'batting' in metrics:
        parts.append(f"{len(metrics['batting'])} batting records")
    if 'bowling' in metrics:
        parts.append(f"{len(metrics['bowling'])} bowling records")
    if 'team_innings' in metrics:
        parts.append(f"{len(metrics['team_innings'])} team innings")
    if 'hat_tricks' in metrics and len(metrics['hat_tricks']) > 0:
        parts.append(f"{len(metrics['hat_tricks'])} hat tricks")
    if 'win_loss' in metrics:
        parts.append(f"{len(metrics['win_loss'])} teams in win/loss table")
    if notes:
        parts.append(f"{len(notes)} stats unavailable due to missing columns")
    return ', '.join(parts)
