# House Price Prediction App 🏡💰

This project predicts house sale prices based on various features using XGBoost and provides a Gradio interface for easy interaction.

## Features
- Predict house sale price based on key features:
  - Lot Area
  - Overall Quality & Condition
  - Living Area (Gr Liv Area)
  - Total Basement Area
  - Garage Cars
  - Bedrooms and Bathrooms
  - Kitchen Quality
  - Year Built
- Interactive web interface using Gradio.

## Technologies Used
- Python 3.12
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Gradio

## Installation

1. Clone this repository:
```bash
git clone https://github.com/0mohamed123/house-price-prediction.git
cd house-price-prediction
```
2. Install dependencies:
```bash
pip install NumPy Pandas XGBoost Scikit-learn Gradio
```
3. Run the Gradio interface:
```bash
python app.py
```

## Model
- Model Type: XGBoost Regressor
- Performance: MAE ≈ $17,000
- Model saved as: models/house_price_model.pkl

## Example Predictions

| Lot Area | Overall Quality | Gr Liv Area | Bedrooms | Kitchen Quality | Year Built | Predicted Price ($) |
|----------|----------------|------------|----------|----------------|------------|-------------------|
| 7000     | 6              | 1500       | 3        | TA             | 2012       | 194,760           |
| 12000    | 9              | 3000       | 4        | Gd             | 2018       | 412,085           |
| 5000     | 4              | 1000       | 2        | Fa             | 1980       | 96,705            |

## Project Structure

house-price-prediction/

├─ train_model.py

├─ predict_ui.py

├─ models/house_price_model.pkl

├─ data/

└─ README.md

## License
This project is for educational and portfolio purposes. Feel free to use and adapt it for learning or demonstrations.
