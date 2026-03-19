import json
import pandas as pd
from src.literature.info_extractor import InfoExtractor
from src.data_pipeline.user_pdf_pipeline import extract_text_from_pdf

class SurveyGenerator:
    def __init__(self):
        self.extractor = InfoExtractor()

    def process_papers(self, files):
        rows = []

        for file in files:
            text = extract_text_from_pdf(file)
            

            try:
                info = self.extractor.extract(text)
                data = json.loads(info)
                rows.append(data)
            except:
                continue

        df = pd.DataFrame(rows)
        df.fillna("Not specified", inplace=True)
        return df