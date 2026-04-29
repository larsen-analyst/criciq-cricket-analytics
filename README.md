# CricIQ - AI-Powered Cricket Analytics

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat&logo=flask&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude-API-orange?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=flat&logo=pandas&logoColor=white)

A natural language cricket analytics web application built as part of an MSc dissertation at the University of Essex. CricIQ allows coaches and athletes to upload ball-by-ball cricket datasets and ask plain English questions to receive instant statistical insights and visualisations.

---

## Features

- **Natural Language Queries** - Ask questions in plain English such as "Top 5 batsmen by total runs since 2020" or "What is the economy rate of the top bowlers?"
- **Automated Data Cleaning** - Detects and reports duplicate deliveries, missing values, and whitespace issues before analysis
- **20+ Pre-Computed Metrics** - Automatically calculates batting averages, bowling economy, run rates by phase, partnerships, hat tricks, dismissal records, and more
- **Year Range Filtering** - Filter all statistics by date range using natural language, e.g. "since 2020" or "2017 to 2022"
- **Direct Data Lookups** - Bypass AI for precise queries such as "Who did SM Curran dismiss?" or "Batsmen who scored 15+ runs against JF Archer in a single over"
- **Interactive Charts** - Bar, line and pie charts generated dynamically from query results
- **Phase Analysis** - Run rates broken down by powerplay (1-6), middle overs (7-15), and death overs (16-20)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.14, Flask 3.1 |
| AI | Anthropic Claude API (claude-haiku-4-5) |
| Data Processing | Pandas 3.0, NumPy |
| Visualisation | Matplotlib |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
criciq/
├── app.py              # Flask application, routes, query engine
├── metrics.py          # Cricket statistics computation engine
├── templates/
│   └── index.html      # Frontend interface
├── static/             # Generated chart images (auto-created)
├── uploads/            # Temporary file uploads (auto-created)
├── .env                # API key (not included in repository)
└── requirements.txt    # Python dependencies
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
pip install flask flask-cors pandas matplotlib anthropic python-dotenv openpyxl
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

1. **Upload Data** - Click the upload area and select a CSV or Excel ball-by-ball cricket dataset
2. **Review Cleaning Report** - The app automatically detects data quality issues and reports them
3. **Ask Questions** - Type any cricket analytics question in plain English and click Ask

### Example Queries

```
Top 5 batsmen by total runs since 2020
Who took the most wickets?
What is the economy rate of the top 5 bowlers?
Show me powerplay run rate by team
List batsmen dismissed by AU Rashid
Top 10 batsmen who scored 15+ runs against JF Archer in a single over
Were there any hat tricks?
Show me run rate in death overs by team
Top 5 batsmen by average runs per innings since 2020
```

---

## Data Format

CricIQ works with ball-by-ball cricket datasets in CSV or Excel format. The application auto-detects column names and adapts to the dataset structure. Recommended columns include:

- `match_id`, `date`, `innings_number`
- `batting_team`, `bowling_team`
- `batter`, `bowler`, `non_striker`
- `over`, `ball_in_over`
- `runs_total`, `runs_batter`, `runs_extras`
- `dismissal_kind`, `player_out`
- `is_boundary`, `is_dot_ball`

Data sourced from [Cricsheet](https://cricsheet.org) works well with this application.

---

## Computed Statistics

### Batting
- Total runs, balls faced, strike rate per batter per inning
- Boundaries (4s) and sixes per batter per inning
- Dot balls faced, batting average across innings
- Centuries (100+) and half centuries (50-99)
- Duck counts (dismissed for 0)
- Partnership runs, century and half century partnerships

### Bowling
- Wickets, runs conceded, economy rate per bowler per inning
- Dot balls bowled, maiden overs
- Best bowling figures per inning
- Hat trick detection with match and team details
- Individual dismissal records per bowler

### Fielding
- Run outs per team per match
- Catches per team per match
- Wicketkeeper dismissals

### Team
- Overall run rate per inning
- Powerplay run rate (overs 1-6)
- Middle overs run rate (overs 7-15)
- Death overs run rate (overs 16-20)
- Powerplay wickets lost
- Win/loss record and toss analysis
- Extras per inning

---

## Dissertation Context

This project was developed as part of an MSc Sport and Exercise Science (Performance Analysis) dissertation at the University of Essex. The research investigates how natural language processing and AI can make cricket performance data accessible to coaches and athletes without requiring technical data analysis skills.

**Supervisor:** Simon Quick  
**Institution:** University of Essex  
**Expected Completion:** 2026

---

## Known Limitations

- Data cleaning deduplication for ball-by-ball datasets is under active development
- Spin vs fast bowler classification requires a `bowler_type` column in the dataset
- Calculated metrics such as win percentage require a `winner` column to be present

---

## Author

**Larsen** - MSc Sport and Exercise Science (Performance Analysis)  
University of Essex, Colchester, UK  
[GitHub](https://github.com/larsen-analyst) | [LinkedIn](https://www.linkedin.com/in/analystlarsen)
