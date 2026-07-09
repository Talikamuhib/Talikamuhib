"""
Generate a realistic (synthetic) ADHD Patient Pathway dataset for Tableau practice.

This produces one row per referral with the fields a real patient-pathway
analyst would work with: referral source, dates, waiting times, appointment
outcomes (attended / DNA / cancelled), triage priority, age band, and outcome.

The data is fully synthetic and safe to share in a portfolio.
Run:  python scripts/generate_adhd_pathway_data.py
Output: data/adhd_pathway_data.csv
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)  # reproducible

N_REFERRALS = 1200
START = date(2024, 1, 1)
END = date(2025, 12, 31)

REFERRAL_SOURCES = ["GP", "School", "Self-referral", "Secondary Care", "Social Services"]
SOURCE_WEIGHTS =   [0.52,   0.18,      0.14,        0.11,             0.05]

CLINICS = ["North Hub", "South Hub", "East Community", "West Community", "Central"]
AGE_BANDS = ["6-11", "12-17", "18-24", "25-39", "40+"]
AGE_WEIGHTS = [0.22, 0.28, 0.20, 0.22, 0.08]
GENDERS = ["Female", "Male", "Other/Not stated"]
GENDER_WEIGHTS = [0.42, 0.55, 0.03]
PRIORITY = ["Routine", "Urgent"]
PRIORITY_WEIGHTS = [0.82, 0.18]

# Appointment stages a patient can progress through
STAGES = ["Triage", "Assessment", "Diagnosis", "Treatment/Follow-up"]


def rand_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def choose(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def wait_days(priority, source):
    """Simulate referral-to-first-appointment waiting time (skewed)."""
    base = 28 if priority == "Urgent" else 84
    spread = 21 if priority == "Urgent" else 70
    w = int(random.gauss(base, spread))
    if source == "Self-referral":
        w += random.randint(0, 21)  # self-referrals often wait a bit longer
    return max(3, w)


def attendance_outcome(priority, wait, age_band, source):
    """Did-not-attend (DNA) risk driven by realistic, learnable factors.

    Real no-show research consistently links non-attendance to longer waits,
    lower-urgency (routine) referrals, younger adults/adolescents, and
    self-referral routes. We encode those as modest, additive effects so a
    model can recover a believable signal (overall DNA rate ~16-18%).
    """
    dna_risk = 0.06

    # Priority: routine referrals no-show more than urgent
    if priority == "Routine":
        dna_risk += 0.05

    # Waiting time: the longer the wait, the higher the risk (dose-response)
    if wait > 140:
        dna_risk += 0.14
    elif wait > 112:
        dna_risk += 0.10
    elif wait > 84:
        dna_risk += 0.06
    elif wait > 56:
        dna_risk += 0.03

    # Age: adolescents and younger adults have higher DNA rates
    if age_band in ("12-17", "18-24"):
        dna_risk += 0.05
    elif age_band == "25-39":
        dna_risk += 0.02

    # Referral route: self-referrals no-show slightly more
    if source == "Self-referral":
        dna_risk += 0.03

    dna_risk = min(dna_risk, 0.55)

    r = random.random()
    if r < dna_risk:
        return "DNA"
    if r < dna_risk + 0.06:
        return "Cancelled"
    return "Attended"


def main():
    rows = []
    for i in range(1, N_REFERRALS + 1):
        referral_date = rand_date(START, END)
        source = choose(REFERRAL_SOURCES, SOURCE_WEIGHTS)
        priority = choose(PRIORITY, PRIORITY_WEIGHTS)
        clinic = random.choice(CLINICS)
        age_band = choose(AGE_BANDS, AGE_WEIGHTS)
        gender = choose(GENDERS, GENDER_WEIGHTS)

        wait = wait_days(priority, source)
        first_appt_date = referral_date + timedelta(days=wait)
        outcome = attendance_outcome(priority, wait, age_band, source)

        # Progression through the pathway (only if they attended)
        if outcome == "Attended":
            # How far along the pathway did they get?
            reached = random.choices(
                [1, 2, 3, 4], weights=[0.12, 0.30, 0.28, 0.30], k=1
            )[0]
            stage = STAGES[reached - 1]
            diagnosed = "Yes" if reached >= 3 and random.random() < 0.72 else "No"
        else:
            stage = "Triage"
            diagnosed = "No"

        # Simple patient engagement score (0-100) influenced by attendance & wait
        engagement = 70
        if outcome == "Attended":
            engagement += 15
        elif outcome == "DNA":
            engagement -= 25
        engagement -= min(20, wait // 12)
        engagement = max(0, min(100, engagement + random.randint(-8, 8)))

        rows.append(
            {
                "referral_id": f"REF-{i:05d}",
                "referral_date": referral_date.isoformat(),
                "referral_month": referral_date.strftime("%Y-%m"),
                "referral_source": source,
                "clinic": clinic,
                "priority": priority,
                "age_band": age_band,
                "gender": gender,
                "wait_days": wait,
                "first_appointment_date": first_appt_date.isoformat(),
                "appointment_outcome": outcome,
                "pathway_stage": stage,
                "diagnosed": diagnosed,
                "engagement_score": engagement,
                "meets_18wk_target": "Yes" if wait <= 126 else "No",
            }
        )

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "adhd_pathway_data.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
