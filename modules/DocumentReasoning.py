import google.generativeai as genai
import json
import os

class DocumentReasoningAgent:
    
    def __init__(self,api_key:str,model_name:str = "gemini-2.0-flash"):

        os.environ['GEMINI_API_KEY'] = api_key
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        self.model = genai.GenerativeModel(model_name)

    def build_prompt(self,extracted_json:dict,ocr_text:str):
        
        prompt = f"""
        You are a document reasoning agent, specialized in receipts.
        You will be given:
        1. Extracted structured json from a receipt.
        2. The OCR text of the receipt.

        Tasks:
        - Detect any missing or inconsistent fields/data in the structured json.
        - Identify possible vendor names from context if extraction failed or missing from json.
        - Suggest corrections or likely values to fill the missing fields from the extracted OCR text.
        - Provide reasoning for your suggestions.
        - Respond ONLY in JSON format with these keys:
        {{
            "observations": [
                {{
                    "field": "field_name",
                    "suggestion": "suggested_value",
                    "reasoning": "explanation for the suggestion"
                }}
            ]
            "overall_summary": "brief summary of the document's content and any other relevant observations"
            "confidence_score": "0-100, indicating the confidence in the suggested values"
        }}
        Example Input:
        
        "json": {
            "Company": "",
            "Date": "2020-01-02",
            "Address": "Shop 12, Main Street",
            "Total": ""
        },
        "ocr_text = "McDonald's\n02/10/2020\nShop12,Main street\nTotal: 15.90""

        Example Output:

        {{
            "observations": [
                {{
                    "field": "Company",
                    "suggestion": "McDonald's",
                    "reasoning": "The vendor name is clearly visible in the OCR text."
                }},
                {{
                    "field": "Total",
                    "suggestion": "15.90",
                    "reasoning": "The total amount is not present in the json, but it is clearly visible in the OCR text."
                }}
            ],
            "overall_summary": "The vendor name is clearly visible in the OCR text and the total amount is not present in the json, but it is clearly visible in the OCR text.",
            "confidence_score": 100
            ]
        }}

        Extracted JSON: {json.dumps(extracted_json,indent=2)}
        OCR Text: {ocr_text}
        """
        return prompt
    
    def infer(self,extracted_json:dict,ocr_text:str):
        
        prompt = self.build_prompt(extracted_json,ocr_text)
        response = self.model.generate_content(prompt)
        raw_output = response.text.strip()

        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            result = {"error":"Invalid Json response","raw_output":raw_output}

        return result