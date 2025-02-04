from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

load_dotenv()

class ChatPDF():
    def __init__(self):
        api_key = os.getenv("OPENAI_KEY")

        self.pdf_loc = os.getenv("FILE_LOC")
        self.client = OpenAI(api_key=api_key)

    def call_chatgpt(self, system_prompt, user_prompt, output_response=True):
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=256
        )
        response = response.choices[0].message.content

        if output_response:
            print(response)

        return response

    def __call__(self, thing_to_predict:str):
        pdf_list = self.read_pdfs()
        seperator = '\n-----\n'
        question = f"Take these sets of texts (sperated by {seperator}) and find key predictors (if any) of {thing_to_predict}"

        format = 'Summarise your answer as a table, formatted with the column names "Paper Name", "Why it is helpful", "Predictive Feature". Also provide the same features under the header "FEATURES:"'

        prompt = f'{question} + \n\n'
        for key, value in pdf_list.items():
            prompt += value
            prompt += seperator

        prompt += format

        gpt_response = self.call_chatgpt(system_prompt=format, user_prompt=prompt)
        response, factors_extended = gpt_response.split('FEATURES:')
        
        features_reduced = self.call_chatgpt("You summarise text into between 1 and 3 keywords. You provide the keywords and nothing more.",
                                             factors_extended)
        
        print(response)
        print(features_reduced)
    
    def extract_pdf(self, file):
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        if 'incorrect startxref pointer(1)' in text:
            text = 'Error reading PDF - try different file'
        return text

    def read_pdfs(self, output_files=False):
        # get PDF text
        pdf_dir = os.path.join('pdfs', self.pdf_loc)
        pdf_files = {}

        for i, file in enumerate(os.listdir(pdf_dir)):
            tmp_file = os.path.join(pdf_dir, file)
            pdf_files[file] = self.extract_pdf(tmp_file)
            if output_files:
                print(f' {i} - {tmp_file}')

        return pdf_files
    

gpt = ChatPDF()
gpt('Ecosystem health')