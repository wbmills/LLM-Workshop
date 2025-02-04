from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import pandas as pd  # Import pandas for handling tabular data

load_dotenv()

class ChatPDF():
    def __init__(self):
        api_key = os.getenv("OPENAI_KEY")
        self.pdf_loc = os.getenv("FILE_LOC")
        self.client = OpenAI(api_key=api_key)
        self.results = []  # To store results from each PDF

    def call_chatgpt(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256
        )
        return response.choices[0].message.content

    def __call__(self, thing_to_predict: str):
        pdf_dir = os.path.join('pdfs', self.pdf_loc)
        for file_name in os.listdir(pdf_dir):
            pdf_path = os.path.join(pdf_dir, file_name)
            pdf_text = self.extract_pdf(pdf_path)
            print(f"Processing: {file_name}")  # Print out the title for each PDF
            self.process_pdf_content(pdf_text, thing_to_predict, file_name)
        
        # After processing all PDFs, generate the final output
        self.generate_final_output()

    def process_pdf_content(self, pdf_text, thing_to_predict, file_name):
        # Prepare the prompt to extract relevant information
        question = f"Based on the following text, identify the key predictors (if any) of {thing_to_predict} and extract the author list and published year, identify the category of paper, including research paper or review paper."

        format = ('Please return the output in the following format:\n'
                'Paper Title: [Title]\n'
                'Author list: [Authors]\n'
                'Published year: [Year]\n'
                'Summary: [Summary]\n'
                'Category: [Category]\n'
                'Predictive Feature for host health: [Features]')

        prompt = f'{question}\n\n{pdf_text}\n{format}'

        gpt_response = self.call_chatgpt(system_prompt=format, user_prompt=prompt)

        # Check for presence of the expected delimiter and handle response accordingly
        if 'FEATURES:' in gpt_response:
            response, factors_extended = gpt_response.split('FEATURES:')
        else:
            # Handle the case where 'FEATURES:' is not found
            response = gpt_response
            factors_extended = ""  # Set as empty or handle differently as needed

        #print("GPT Response:")
        #print(response)

        # Extracting additional details from the response
        lines = response.split('\n')
        details = {line.split(':')[0]: line.split(':')[1].strip() for line in lines if ':' in line}

        # Append results to the results list
        self.results.append({
            "Paper Title": details.get("Paper Title"),
            "Author list": details.get("Author list"),
            "Published year": details.get("Published year"),
            "Summary": details.get("Summary"),
            "Category": details.get("Category"),
            "Predictive Feature for host health": details.get("Predictive Feature for host health")
        })

    def generate_final_output(self):
        # Convert results to a DataFrame and print
        df = pd.DataFrame(self.results)
        #print("\nFinal Output Table:")
        #print(df)

        # Export to a .txt file with tab-separated values
        output_file = "gut_output.txt"
        df.to_csv(output_file, sep='\t', index=False)

        print(f"Results exported to {output_file}")

    def extract_pdf(self, file):
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        if 'incorrect startxref pointer(1)' in text:
            text = 'Error reading PDF - try different file'
        return text

# Example of creating an instance and calling it
gpt = ChatPDF()
gpt('Gut_microbiome')

