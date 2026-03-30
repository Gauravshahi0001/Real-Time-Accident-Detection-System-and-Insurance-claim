 Interactive dashboard for accident monitoring

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the ML model:
```bash
python train_model.py
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## System Components

- Flask web application
- SQLite database for accident records
- Random Forest Classifier for severity prediction
- Bootstrap-based responsive UI
- Real-time updates using JavaScript

## Usage

1. Access the web interface
2. Fill out the accident report form
3. Submit the report to get instant severity prediction
4. Monitor real-time updates on the dashboard
5. View historical accident data and analysis

## Technical Stack

- Python 3.8+
- Flask
- SQLAlchemy
- scikit-learn
- Bootstrap 5
- SQLite
