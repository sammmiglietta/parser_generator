import warnings
import os
import getpass
from typing import List
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

warnings.filterwarnings("ignore", message = "divide by zero encountered in divide")
warnings.simplefilter(action = 'ignore', category = FutureWarning)

def set_if_undefined(var):
    """Set environment variable (API KEY)."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

def initialize_llm():
    """Initialize a hosted model from Hugging Face (< 10 GB)."""
    # model_id = 'codellama/CodeLlama-34b-Instruct-hf'
    # https://evalplus.github.io/leaderboard.html
    # model_id = 'microsoft/Phi-3-mini-4k-instruct' # 56esimo
    # model_id = 'meta-llama/Meta-Llama-3-8B-Instruct' # 60eismo
    model_id = 'bigcode/starcoder2-15b' # 79eismo

    llm = HuggingFaceEndpoint(huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
                              repo_id=model_id,
                              task="text-generation",
                              temperature = 0.5, 
                              repetition_penalty = 1.3,
                              do_sample = True,
                              top_p = 0.9,
                              max_new_tokens = 2048  # increase for longer parser implementations
                              )
    return llm

def create_prompt_template():
    """Create prompt template for generating parser functions in C."""

    template = """You are a specialized C programming agent that creates parser functions following strict requirements.
Each parser you create must have all of the following characteristics:

1. Input Handling: The code deals with a pointer to a buffer of bytes or a file descriptor for reading unstructured data.

2. Internal State Management:
   - code must maintain internal state in memory during parsing
   - state should track parsing progress and context
   - state should not be part of the output

3. Decision-Making: the code takes decisions based on the input and the internal state.

4. Data Structure Creation: the function must build appropriate data structures for parsed content or execute specific actions based on parsed content

5. Parser Outcome: the code returns either a boolean value or a data structure built from the parsed data indicating the outcome of the recognition.

6. Parser Composition: the code behavior is defined as a composition of other parsers.

Task: {input}

Generate a complete C implementation that strictly follows all these requirements.
Include all necessary header files, structures, and supporting functions.

Code:
"""
    return PromptTemplate.from_template(template)

def run_llm(input):
    ### set up API keys
    set_if_undefined("HUGGING_FACE_API_KEY")

    ### initialize model and prompt
    llm = initialize_llm()
    prompt = create_prompt_template()

    ### runnable sequence
    chain = prompt | llm

    try:
        ### run the chain
        response = chain.invoke({
            "input": input
        })
        print(response)
        
    except Exception as e:
        print(f"Error running chain: {str(e)}")


request = """Create a parser for JSON files."""
# request = """Create a parser for CSV files that extracts numeric columns."""
run_llm(request)