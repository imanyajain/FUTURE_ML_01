"""
forecast_prophet.py
Aggregates the Superstore file to monthly sales, fits a Prophet model,
forecasts next N months, saves forecast CSV and displays plots.

Usage:
  python forecast_prophet.py --input "data/Sample - Superstore.csv" --output_dir "reports" --months 12
"""

import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def prepare_monthly(input_csv):
    df = pd.read_csv(input_csv, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    monthly = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    monthly = monthly.rename(columns={'Order Date':'ds', 'Sales':'y'})
    return monthly

def train_forecast(df_monthly, periods=12, freq='M'):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_monthly)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future)
    return m, fc

def plot_and_save(df_monthly, model, fc, out_dir, periods):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save forecast table (last `periods` rows)
    forecast_out = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(periods).rename(columns={'ds':'date'})
    forecast_csv = out_dir / 'forecast_{}_months.csv'.format(periods)
    forecast_out.to_csv(forecast_csv, index=False)
    print("Saved forecast CSV:", forecast_csv)

    # Plot actual vs forecast
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_monthly['ds'], df_monthly['y'], label='Actual')
    ax.plot(fc['ds'], fc['yhat'], label='Forecast')
    ax.fill_between(np.array(fc['ds'].dt.to_pydatetime()), fc['yhat_lower'], fc['yhat_upper'], alpha=0.2)
    ax.set_title(f'Monthly Sales — Actual vs Forecast ({periods} months ahead)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    fig_path = out_dir / 'actual_vs_forecast.png'
    fig.tight_layout()
    fig.savefig(fig_path)
    print("Saved plot:", fig_path)

    # Components plot
    comp = model.plot_components(fc)
    comp_fig_path = out_dir / 'forecast_components.png'
    comp.tight_layout()
    comp_fig_path = out_dir / 'forecast_components.png'
    comp.savefig(comp_fig_path)
    print("Saved components plot:", comp_fig_path)

    return forecast_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to Superstore CSV')
    parser.add_argument('--output_dir', type=str, default='reports', help='Output directory')
    parser.add_argument('--months', type=int, default=12, help='Forecast horizon in months')
    args = parser.parse_args()

    monthly = prepare_monthly(args.input)
    model, fc = train_forecast(monthly, periods=args.months)
    forecast_csv = plot_and_save(monthly, model, fc, args.output_dir, args.months)
    print("Done — forecast CSV:", forecast_csv)

if __name__ == '__main__':
    main()
