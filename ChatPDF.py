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
        self.thing_to_predict = ''

    def call_chatgpt(self, system_prompt, user_prompt, output_response=False):
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
        self.thing_to_predict = thing_to_predict
        seperator = '\n-----\n'
        question = f"Take these sets of texts (sperated by {seperator}) and find key predictors (if any) of {thing_to_predict}"

        format = 'You summarise your answers as comma-deliminated tables fit for a csv (not a terminal), with no "|" values, with the column names "Paper Name", "Why it is helpful", "Predictive Feature".'

        prompt = f'{question} + \n\n'
        for key, value in pdf_list.items():
            prompt += value
            prompt += seperator

        prompt += format

        gpt_response = self.call_chatgpt(system_prompt=format, user_prompt=prompt)

        with open('file.csv', 'w') as f:
            f.write(gpt_response)

        features_reduced = self.call_chatgpt(
            "You summarise text into between 1 and 3 keywords. You provide the keywords and nothing more.",
            'Sumarise the features column of this: ' + gpt_response)
        
        print(features_reduced)
        return gpt_response
    
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
    
    def call_dalle3(self, given_table):
        assert self.thing_to_predict is not None, 'call ChatPDF and set a thing to predict first!'
        
        prompt = self.call_chatgpt(system_prompt="you provide a summary of these features in less than 400 characters (no table). Create it as a prompt for dalle, where the aim is to get a nice image or cartoon of the thing I am trying to predict. ", user_prompt=given_table)
        prompt += 'This image has no words or letters. It is artistic. It is in the style of jackson polloc'
        response = self.client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="512x512",
            quality="standard",
            n=1,
        )

        print(response.data[0].url)
    

gpt = ChatPDF()
table = gpt('Ecosystem health')
# gpt.call_dalle3(table)