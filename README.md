# Government File Processing System

A machine learning-based system for predicting and optimizing government file processing workflows. This project combines data generation, predictive modeling, and interactive dashboards to help government departments manage their document processing efficiently.

## Overview

This system helps government departments:
- **Predict Processing Times**: Use machine learning to estimate how long a government file will take to process
- **Identify Delay Risks**: Classify files at risk of exceeding their Service Level Agreements (SLAs)
- **Visualize Workflows**: Interactive dashboards to monitor department performance and file status
- **Optimize Resource Allocation**: Better understand workload distribution across departments and officers

## Project Structure

```
.
├── data/
│   ├── new_files.csv              # Input files for processing
│   ├── predictions.csv            # Model predictions on files
│   └── synthetic_files.csv        # Synthetic training dataset
├── notebooks/                      # Jupyter notebooks for exploration
├── src/
│   ├── data_gen.py                # Synthetic dataset generation
│   ├── models/
│   │   └── train_models.py        # ML model training pipeline
│   ├── api/                       # API endpoints
│   └── dashboard/
│       └── app.py                 # Streamlit dashboard application
└── README.md
```

## Features

### 1. Data Generation (`src/data_gen.py`)
- Generates realistic synthetic government file datasets
- Simulates files from multiple departments (Revenue, Transport, Health, Education, Welfare, Urban Development)
- Creates varied file types: applications, permits, appeals, and reports
- Includes realistic attributes like priority levels, complexity scores, and SLA requirements

### 2. Machine Learning Models (`src/models/train_models.py`)
- **Processing Time Regression**: Predicts how many hours a file will take to process
- **Delay Risk Classification**: Predicts if a file will exceed its SLA deadline
- Uses ensemble methods (Random Forest, LightGBM) for robust predictions
- Preprocessing pipeline with one-hot encoding for categorical features
- Model evaluation metrics: MSE, R², accuracy, and detailed classification reports

### 3. Interactive Dashboard (`src/dashboard/app.py`)
- Built with **Streamlit** for real-time monitoring
- **Visualizations**:
  - Processing time trends by department
  - Delay risk distribution
  - Officer workload analysis
  - File priority and complexity analysis
- **Predictions**: Submit new files and get instant processing time and delay risk predictions
- **Data Export**: Save predictions to CSV for further analysis

## Installation

### Requirements
- Python 3.8+
- pandas, numpy
- scikit-learn, LightGBM
- Streamlit, Plotly
- joblib

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/parth-garg01/gov-workflow-optimizer.git
cd gov-workflow-optimizer
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data
```bash
python src/data_gen.py
```
This creates `data/synthetic_files.csv` with 10,000 sample government files.

### 2. Train Models
```bash
python src/models/train_models.py
```
This trains and saves:
- `processing_time_model.pkl`: Regression model for time prediction
- `delay_risk_model.pkl`: Classification model for delay detection

### 3. Run Dashboard
```bash
streamlit run src/dashboard/app.py
```
Access the dashboard at `http://localhost:8501`

## Data Schema

### File Record Attributes
| Field | Type | Description |
|-------|------|-------------|
| file_id | UUID | Unique file identifier |
| department | String | Assigned government department |
| file_type | String | Type: application, permit, appeal, or report |
| priority | String | Priority level: Low, Medium, High |
| submission_date | DateTime | When file was submitted |
| complexity_score | Float | 0-1 score of file complexity |
| num_pages | Integer | Number of pages in the file |
| required_approvals | Integer | Number of approvals needed |
| assigned_officer_id | String | Officer processing the file |
| officer_experience_years | Float | Years of experience |
| current_backlog_officer | Integer | Current workload for officer |
| sla_days | Integer | Service level agreement deadline (days) |
| routing_path | String | Processing stages (Clerk→Officer→...→Director) |
| processing_time_hours | Float | **Target: Actual processing time** |
| processing_time_days | Float | Processing time in days |
| delayed | Boolean | **Target: Whether SLA was missed** |

## Model Performance

- **Processing Time Regression**: Trained to minimize MSE across multiple file types and priorities
- **Delay Risk Classification**: Achieves high accuracy in identifying at-risk files before they exceed SLA
- Models are saved as pickle files for quick loading and inference

## Dashboard Features

### Main Sections
1. **Overview**: Key metrics and summary statistics
2. **Department Analysis**: Performance by department
3. **File Status**: Monitor individual files and their progress
4. **Predictions**: Make predictions on new files
5. **Officer Workload**: View officer assignment and backlog

### Interactive Controls
- Filter by department, priority, and date range
- Export predictions and analysis results
- Real-time model inference on new submissions

## API Endpoints

(If applicable to your setup)
- `/predict`: Submit a new file for prediction
- `/files`: Query file status and history
- `/metrics`: Get department and system-wide metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - feel free to use this project for educational and commercial purposes.

## Contact

For questions or support, please contact the project maintainers.

## Acknowledgments

- Built with Streamlit, scikit-learn, and LightGBM
- Designed to optimize government file processing workflows
