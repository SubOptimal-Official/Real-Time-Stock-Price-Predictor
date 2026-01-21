from flask import Flask, render_template, request
from stock_predictor import StockPredictor
import yfinance as yf

app = Flask(__name__)
predictor = StockPredictor()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    chart_data = None

    if request.method == "POST":
        ticker = request.form.get("ticker").upper()

        result = predictor.run(ticker)

        stock = yf.Ticker(ticker)
        hist = stock.history(period="3y")

        chart_data = {
            "dates": hist.index.strftime("%Y-%m-%d").tolist(),
            "prices": hist["Close"].round(2).tolist()
        }

    return render_template(
        "index.html",
        result=result,
        chart_data=chart_data
    )

if __name__ == "__main__":
    app.run(debug=True)
