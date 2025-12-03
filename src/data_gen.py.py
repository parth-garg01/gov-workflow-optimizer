import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
import random

def random_choice_with_weights(choices, weights, size):
    return np.random.choice(choices, size=size, p=np.array(weights)/sum(weights))

def generate_synthetic_dataset(n_rows=10000, seed=42, save_path="../data/synthetic_files.csv"):
    np.random.seed(seed)
    random.seed(seed)

    departments = ["Revenue", "Transport", "Health", "Education", "Welfare", "UrbanDev"]
    dept_weights = [0.2, 0.15, 0.15, 0.2, 0.2, 0.1]

    file_types = ["application", "permit", "appeal", "report"]
    file_type_weights = [0.45, 0.2, 0.15, 0.2]

    priorities = ["Low", "Medium", "High"]
    priority_weights = [0.6, 0.3, 0.1]

    base_date = datetime.now() - timedelta(days=365)

    rows = []
    for i in range(n_rows):
        file_id = str(uuid.uuid4())
        dept = random_choice_with_weights(departments, dept_weights, 1)[0]
        ftype = random_choice_with_weights(file_types, file_type_weights, 1)[0]
        priority = random_choice_with_weights(priorities, priority_weights, 1)[0]
        submission_date = base_date + timedelta(days=int(np.random.exponential(scale=120)))

        complexity_score = np.clip(np.random.beta(2, 5), 0, 1)
        num_pages = int(np.clip(np.random.normal(loc=5 + complexity_score*10, scale=3), 1, 200))
        required_approvals = int(np.clip(np.random.poisson(lam=1 + complexity_score*2), 1, 6))

        assigned_officer_id = f"OFF{np.random.randint(1, 200):03d}"
        officer_experience_years = round(max(0.5, np.random.normal(loc=3 - 0.5*(priority=="High"), scale=2)), 1)
        current_backlog_officer = int(np.clip(np.random.poisson(lam=5 + complexity_score*10), 0, 200))

        sla_map = {
            "application": {"Low": 7, "Medium": 5, "High": 3},
            "permit": {"Low": 14, "Medium": 10, "High": 5},
            "appeal": {"Low": 21, "Medium": 14, "High": 7},
            "report": {"Low": 10, "Medium": 7, "High": 4}
        }
        sla_days = sla_map[ftype][priority]

        stages = ["Clerk", "Officer", "SectionHead", "Finance", "Director"]
        routing_path = "->".join(stages[:min(len(stages), 1 + required_approvals)])

        base_hours = 2 + 0.5 * num_pages + 10 * complexity_score + 8 * (required_approvals - 1)
        backlog_penalty = 0.3 * current_backlog_officer
        experience_reduction = max(0, (officer_experience_years - 1) * -1.5)
        priority_adj = {"Low": 1.0, "Medium": 0.9, "High": 0.7}[priority]

        dept_factor = {"Revenue": 1.05, "Transport": 1.0, "Health": 1.1, "Education": 0.95, "Welfare": 1.08, "UrbanDev": 1.12}[dept]

        noise = np.random.normal(0, 12)
        processing_time_hours = max(
            1.0,
            (base_hours + backlog_penalty + experience_reduction) * priority_adj * dept_factor + noise
        )

        sla_hours = sla_days * 24
        delay_ratio = processing_time_hours / max(1, sla_hours)

        prob_delay = 1 / (1 + np.exp(-6 * (delay_ratio - 0.6)))
        prob_delay = np.clip(prob_delay + 0.15 * complexity_score + 0.001 * current_backlog_officer, 0, 0.99)
        delayed = np.random.rand() < prob_delay

        rows.append({
            "file_id": file_id,
            "department": dept,
            "file_type": ftype,
            "priority": priority,
            "submission_date": submission_date,
            "complexity_score": round(float(complexity_score), 3),
            "num_pages": num_pages,
            "required_approvals": required_approvals,
            "assigned_officer_id": assigned_officer_id,
            "officer_experience_years": officer_experience_years,
            "current_backlog_officer": current_backlog_officer,
            "routing_path": routing_path,
            "sla_days": sla_days,
            "processing_time_hours": round(float(processing_time_hours), 2),
            "delayed": delayed,
            "delay_ratio": round(float(delay_ratio), 3)
        })

    df = pd.DataFrame(rows)
    df["processing_time_days"] = (df["processing_time_hours"] / 24).round(3)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to: {save_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_dataset()
