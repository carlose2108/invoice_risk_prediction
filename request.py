import requests
import json

url = "https://xepelin-invoice-risk-2utpj2tmea-uc.a.run.app/invoice_risk"

payload = json.dumps({
  "invoice_id": [
    4919,
    10423
  ],
  "country": "CL"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
