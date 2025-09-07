import gradio as gr
import joblib

# load model
model = joblib.load("models/house_price_model.pkl")

# predict function
def predict_price(lot_area, overall_qual, overall_cond, gr_liv_area,
                  total_bsmt_sf, garage_cars, full_bath, bedroom_abvgr,
                  kitchen_qual, year_built):
    
    # calculate HouseAge
    house_age = 2025 - year_built
    
    # map kitchen quality to numeric
    kitchen_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    kitchen_qual_num = kitchen_mapping.get(kitchen_qual, 3)  # default TA
    
    data = [[lot_area, overall_qual, overall_cond, gr_liv_area,
             total_bsmt_sf, garage_cars, full_bath, bedroom_abvgr,
             kitchen_qual_num, house_age]]
    
    prediction = model.predict(data)[0]
    return f"💰 Predicted Sale Price: ${prediction:,.2f}"

# Gradio UI
inputs = [
    gr.Number(label="Lot Area", value=7000),
    gr.Slider(1, 10, step=1, label="Overall Quality", value=6),
    gr.Slider(1, 10, step=1, label="Overall Condition", value=6),
    gr.Number(label="Living Area (Gr Liv Area)", value=1500),
    gr.Number(label="Total Basement SF", value=800),
    gr.Number(label="Garage Cars", value=2),
    gr.Number(label="Full Bath", value=2),
    gr.Number(label="Bedrooms Above Ground", value=3),
    gr.Dropdown(["Ex", "Gd", "TA", "Fa", "Po"], label="Kitchen Quality", value="TA"),
    gr.Number(label="Year Built", value=2012)
]

output = gr.Textbox(label="Prediction")

app = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=output,
    title="🏡 House Price Prediction App (Improved)",
    description="Predict house prices using an XGBoost model with engineered features.",
)

if __name__ == "__main__":
    app.launch()