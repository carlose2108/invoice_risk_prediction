from pydantic import BaseModel, field_validator

import pandas as pd
from typing import List


class InvoicePayload(BaseModel):
    invoice_id: List[int]
    country: str

    @field_validator("invoice_id")
    def validate_invoice_id(cls, value):

        invoice_ids = pd.read_csv("../data/dataTest.csv")
        valid_invoice_ids = invoice_ids["invoiceId"].tolist()

        if not all(item in valid_invoice_ids for item in value):
            raise ValueError(f"Invalid invoice id. Must be one of: {valid_invoice_ids}")
        return value

    @field_validator("country")
    def validate_country(cls, value):
        if value not in ["CL", "MX"]:
            raise ValueError("Invalid country. Must be 'CL' or 'MX'.")
        return value


# Example Usage
try:
    person = InvoicePayload(invoice_id=["4919", "10423", 123], country="CL")
except ValueError as e:
    print(e)  # Output: Value error, Age must be positive
