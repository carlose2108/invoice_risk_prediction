import pandas as pd

from typing import Dict, Any
from app.api import InvoiceRisk, InvoicePayload

from fastapi import status

from app.api import init_app

app = init_app()


@app.post("/invoice_risk", response_model=InvoiceRisk, status_code=status.HTTP_200_OK)
def predict(payload: InvoicePayload) -> Dict[str, Any]:
    # Read data
    df = pd.read_csv("data/dataTest.csv")
    invoices_to_predict = df[df["invoiceId"].isin(payload.invoice_id)].drop(columns=["Unnamed: 0", "invoiceId"])

    # Make predictions
    preds = app.model.predict(invoices_to_predict)
    predictions = InvoiceRisk(invoice_risk_predictions=preds)
    predictions = predictions.model_dump()
    return predictions


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> Dict[str, str]:
    return {"healthcheck": "Everything OK!"}
