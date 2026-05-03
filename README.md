# CricIQ - AI-Powered Cricket Analytics

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat&logo=flask&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude-API-orange?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=flat&logo=pandas&logoColor=white)

A natural language cricket analytics web application built as part of an MSc dissertation at the University of Essex. CricIQ allows coaches and athletes to upload ball-by-ball cricket datasets and ask plain English questions to receive instant statistical insights and visualisations.

---

## Features

### Data Upload
- Single file upload: CSV, Excel, YAML, YML
- Multiple ZIP upload: upload Cricsheet ZIP files one at a time, CricIQ merges automatically
- Supports both old (pre-2017) and new (post-2017) Cricsheet YAML formats
- Auto-generates clean CSV from YAML files for future use
- Session persistence: data stays loaded when navigating between pages

### Data Cleaning
- Detects and reports duplicate deliveries, missing values, and whitespace issues
- Cricket-rules-accurate run validation:
  - 0 to 7: Valid (maximum is no ball + six)
  - 8 to 9: Invalid, flagged and capped to 7 on Apply All Fixes
  - 10: Rare but valid (wide + helmet penalty + boundary four), flagged for coach confirmation
  - 11+: Invalid, flagged as definite error
- Detects corrupted ball numbering (ball_in_over stuck at 1)
- Detects wrong runs column (mean below 0.5 or above 3.0)

### Query Engine (Intent-Based Architecture)
- Claude API classifies query intent only, Python fetches exact numbers from data
- Answer and chart guaranteed to match as they come from the same data source
- 120+ cricket terms in terminology dictionary sourced from Wikipedia
- Fuzzy player name matching: handles surname only, initials, partial names
- Player name autocomplete search above query box
- Year range filtering: "since 2020", "2017 to 2022", "before 2018"
- Opponent filtering: "top 5 centuries against India"
- Team filtering using adjectives: "English bowlers", "Australian batsmen"
- Single innings detection: shows best per-inning figures not career totals
- Dismissal type filtering: "LBW dismissals by SM Curran"
- Top N ranking with correct data source for charts

### Metrics Computed (20+ Statistics)

**Batting:** runs scored, balls faced, strike rate, fours, sixes, dot balls faced, batting average, centuries, half centuries, ducks, top scorer per innings

**Bowling:** wickets, runs conceded, economy rate, dot balls bowled, maiden overs, best bowling per innings, hat tricks, dismissal records by type

**Fielding:** run outs, catches, wicketkeeper dismissals per team per match

**Team:** overall run rate, powerplay run rate (overs 1 to 6), middle overs run rate (overs 7 to 15), death overs run rate (overs 16 to 20), powerplay wickets lost, extras, win/loss record, toss analysis

**Large Dataset Mode:** For datasets over 100,000 rows, fast metrics mode skips expensive hat trick and partnership computation to maintain performance

### Dashboard
Professional analytics dashboard at `/dashboard` with:
- 12 KPI cards: matches, runs, wickets, teams, batsmen, bowlers, top scorer, top wickets, sixes, fours, centuries, half centuries
- Top 10 run scorers and wicket takers bar charts
- Top 10 strike rates and economy rates
- Run rate by phase grouped bar chart
- Win/loss doughnut chart
- Runs and wickets by team
- Top six hitters and boundary hitters inline bars
- Batting average bar chart
- Dismissal types pie chart
- Top dot ball and maiden over bowlers
- Powerplay wickets by team

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.14, Flask 3.1 |
| AI | Anthropic Claude API (claude-haiku-4-5) |
| Data Processing | Pandas 3.0, NumPy |
| Visualisation | Matplotlib |
| YAML Parsing | PyYAML with CLoader |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
criciq/
├── app.py               # Flask application, routes, intent-based query engine
├── metrics.py           # Cricket statistics computation engine
├── yaml_converter.py    # Cricsheet YAML to CSV converter (old and new formats)
├── templates/
│   ├── index.html       # Main query interface
│   └── dashboard.html   # Analytics dashboard
├── static/              # Generated chart images (auto-created)
├── uploads/             # Temporary file uploads (auto-created)
├── .env                 # API key (not included in repository)
└── requirements.txt     # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (get one at https://console.anthropic.com)

### Installation

1. Clone the repository
```bash
git clone https://github.com/larsen-analyst/criciq-cricket-analytics.git
cd criciq-cricket-analytics
```

2. Install dependencies
```bash
pip install flask flask-cors pandas matplotlib anthropic python-dotenv openpyxl pyyaml
```

3. Create a `.env` file in the root folder
```
ANTHROPIC_API_KEY=your_api_key_here
```

4. Create required folders
```bash
mkdir static uploads
```

5. Run the application
```bash
python app.py
```

6. Open your browser at `http://127.0.0.1:5000`

---

## Usage

### Uploading Data

**Single File:** Click the upload area and select a CSV, Excel or YAML file

**Multiple ZIPs:** Switch to Multi ZIP mode, upload Cricsheet ZIP files one at a time, click Finalise and Analyse when done, then download the merged CSV for future use

### Example Queries

```
Top 5 batsmen by total runs since 2020
Who took the most wickets?
Economy rate of the top 5 bowlers
Show me powerplay run rate by team
Top 10 centuries scored against India
List batsmen dismissed by AU Rashid
How many LBW dismissals by SM Curran
Top 5 bowlers with most LBW dismissals
Batsmen who scored 15+ runs against JF Archer in a single over
Top 5 English bowlers in a single innings
Were there any hat tricks?
Show me run rate in death overs by team
Most sixes in the dataset
Top batsmen by strike rate since 2018
```

---

## Data Format

CricIQ works with ball-by-ball cricket datasets. The application auto-detects column names and adapts to the dataset structure. Recommended columns:

`match_id`, `date`, `innings_number`, `batting_team`, `bowling_team`, `batter`, `bowler`, `non_striker`, `over`, `ball_in_over`, `runs_total`, `runs_batter`, `runs_extras`, `dismissal_kind`, `player_out`, `is_boundary`, `is_dot_ball`

Data sourced from [Cricsheet](https://cricsheet.org) works directly with this application.

---

## Architecture

### Query Pipeline (Intent-Based)

```
User Query
    ↓
Cricket Terminology Normalisation (120+ terms)
    ↓
Direct Lookup (dismissals, over runs, rankings)
    OR
Claude API: classifies intent only
    ↓
Python fetches exact numbers from computed metrics
    ↓
Answer built from Python data (guaranteed correct)
    ↓
Chart built from identical Python data (guaranteed matches answer)
```

This architecture ensures every number in every answer comes directly from the data. Claude never writes numbers, only decides what to show.

### Data Flow

```
Upload → Detect format → Parse/clean → Compute metrics → Store in memory
    ↓
Query → Classify intent → Fetch from metrics → Build answer + chart
    ↓
Session persists across page navigation (no re-upload needed)
```

---

## Dissertation Context

This project was developed as part of an MSc Sport and Exercise Science (Performance Analysis) dissertation at the University of Essex. The research investigates how natural language processing and AI can make cricket performance data accessible to coaches and athletes without requiring technical data analysis skills.

**Supervisor:** Simon Quick
**Institution:** University of Essex
**Expected Completion:** 2026

---

## Known Limitations

- Hat tricks and partnerships skipped for datasets over 100,000 rows
- Spin vs fast bowler classification requires a `bowler_type` column
- Cloud deployment not yet configured (runs locally only)

---

## Author

**Larsen** - MSc Sport and Exercise Science (Performance Analysis)
University of Essex, Colchester, UK
[GitHub](https://github.com/larsen-analyst) | [LinkedIn](https://www.linkedin.com/in/analystlarsen)
