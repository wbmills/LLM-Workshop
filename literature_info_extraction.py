# Import necessary libraries
from openai import OpenAI  # OpenAI API client for making GPT calls
from dotenv import load_dotenv  # Loads environment variables from .env file
from PyPDF2 import PdfReader  # Library to read and extract text from PDFs
import os  # For accessing environment variables and file paths

# Load environment variables from a .env file
load_dotenv()

# Define a class ChatPDF to handle PDF reading and GPT-based information extraction
class ChatPDF():
    def __init__(self):
        """
        Initializes the ChatPDF class.
        - Loads the OpenAI API key from environment variables.
        - Loads the location of PDF files.
        - Creates an OpenAI client instance.
        """
        api_key = os.getenv("OPENAI_KEY")  # Fetch API key from environment
        self.pdf_loc = os.getenv("FILE_LOC")  # Fetch the folder location where PDFs are stored
        self.client = OpenAI(api_key=api_key)  # Initialize the OpenAI client

    def call_chatgpt(self, system_prompt, user_prompt, output_response=True):
        """
        Calls the OpenAI GPT model with system and user prompts.
        - Uses the 'gpt-4o-mini' model to process the request.
        - Takes system and user prompts as input.
        - Returns the response from GPT.

        Parameters:
        - system_prompt (str): Instruction to GPT on how to behave.
        - user_prompt (str): User's query to GPT.
        - output_response (bool): If True, prints the response.

        Returns:
        - str: GPT-generated response.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Specifies which GPT model to use
            messages=[
                {"role": "system", "content": system_prompt},  # System prompt sets context
                {"role": "user", "content": user_prompt},  # User prompt asks the question
            ],
            max_tokens=256  # Limits the response length to 256 tokens
        )
        response = response.choices[0].message.content  # Extracts the text response

        if output_response:
            print(response)  # Prints the response if required

        return response

    def __call__(self, thing_to_predict: str):
        """
        Calls the class instance to analyze PDFs and extract key predictive features using GPT.

        Parameters:
        - thing_to_predict (str): The subject for which key predictors are sought.

        Steps:
        1. Reads PDFs and extracts text.
        2. Constructs a GPT prompt to find key predictors from the text.
        3. Calls GPT and extracts key features.
        4. Summarizes extracted features into keywords.
        5. Prints extracted insights.
        """
        pdf_list = self.read_pdfs()  # Reads all PDFs and extracts text
        seperator = '\n-----\n'  # Defines separator between different papers in the prompt

        # Constructs the question for GPT
        question = f"Take these sets of texts (separated by {seperator}) and find key predictors (if any) of {thing_to_predict}"

        # Specifies the response format for GPT
        format = 'Summarise your answer as a table, formatted with the column names "Paper Name", "Why it is helpful", "Predictive Feature". Also provide the same features under the header "FEATURES:"'

        # Builds the complete prompt by appending extracted text
        prompt = f'{question} + \n\n'
        for key, value in pdf_list.items():
            prompt += value
            prompt += seperator

        prompt += format  # Adds the formatting instructions to the prompt

        # Calls GPT to extract insights
        gpt_response = self.call_chatgpt(system_prompt=format, user_prompt=prompt)

        # Splits the response to separate the main table from features
        response, factors_extended = gpt_response.split('FEATURES:')
        
        # Calls GPT again to extract concise feature keywords
        features_reduced = self.call_chatgpt(
            "You summarise text into between 1 and 3 keywords. You provide the keywords and nothing more.",
            factors_extended
        )
        
        # Prints extracted insights
        print(response)
        print(features_reduced)
    
    def extract_pdf(self, file):
        """
        Extracts text from a given PDF file.

        Parameters:
        - file (str): Path to the PDF file.

        Returns:
        - str: Extracted text from the PDF.
        """
        reader = PdfReader(file)  # Initialize PDF reader
        # Extract text from all pages in the PDF
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

        # If there's an issue with reading the PDF, return an error message
        if 'incorrect startxref pointer(1)' in text:
            text = 'Error reading PDF - try different file'
        return text

    def read_pdfs(self, output_files=False):
        """
        Reads all PDFs from the specified folder, extracts text, and stores it in a dictionary.

        Parameters:
        - output_files (bool): If True, prints the list of processed files.

        Returns:
        - dict: Dictionary with file names as keys and extracted text as values.
        """
        pdf_dir = os.path.join('pdfs', self.pdf_loc)  # Constructs full path to the PDF directory
        pdf_files = {}

        # Iterate through all files in the directory
        for i, file in enumerate(os.listdir(pdf_dir)):
            tmp_file = os.path.join(pdf_dir, file)  # Get full file path
            pdf_files[file] = self.extract_pdf(tmp_file)  # Extract text and store it
            if output_files:
                print(f' {i} - {tmp_file}')  # Print the processed file names

        return pdf_files  # Return extracted text from all PDFs
    

# Create an instance of ChatPDF
gpt = ChatPDF()

# Call the instance with 'Ecosystem health' as the topic to predict
gpt('Ecosystem health')
