from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import tiktoken
import json
import os
import urllib.parse  # Used for parsing the URL
import ast
import datetime

with open("openai_config.json") as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key["openai_key"]

def num_tokens_from_string(string: str, 
                           encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_into_chunks(text, 
                      chunk_size=8000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def json_to_linkedin_post(data):
    # Start building the LinkedIn post content
    linkedin_content = []
    
    # Title with relevant emoji
    linkedin_content.append(f"✨ {data['Title']} ✨\n")
    
    # Introduction with 'bulb' emoji
    linkedin_content.append(f"💡 Introduction:\n{data['Introduction']}\n")
    
    # Main Features with 'gear' emoji
    linkedin_content.append(f"⚙️ Main Features:\n{data['Main Features']}\n")
    
    # Case Study or Example with 'book' emoji
    linkedin_content.append(f"📖 Case Study or Example:\n{data['Case Study or Example']}\n")
    
    # Importance and Benefits with 'heart' emoji
    linkedin_content.append(f"❤️ Importance and Benefits:\n{data['Importance and Benefits']}\n")
    
    # Future Directions with 'rocket' emoji
    linkedin_content.append(f"🚀 Future Directions:\n{data['Future Directions']}\n")
    
    # Call to Action with 'loudspeaker' emoji
    linkedin_content.append(f"📢 Call to Action:\n{data['Call to Action']}\n")
    
    # Hashtags
    linkedin_content.append(f"{data['Hashtags']}\n")
    
    # Join all parts of the LinkedIn post content into a single string
    final_linkedin = '\n'.join(linkedin_content)
    
    return final_linkedin


def save_markdown_file(markdown_content, url, folder_path):
    # Parse the URL and extract the base file name
    parsed_url = urllib.parse.urlparse(url)
    base_name = os.path.basename(parsed_url.path)
    if not base_name:  # In case the URL ends with a slash and has no base name
        base_name = "default"
    
    # Create a valid file name by appending '.md' and replacing invalid characters
    file_name = base_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(".json","") + ".md"
    
    # Join folder path with file name
    file_path = os.path.join(folder_path, file_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Write Markdown content to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(markdown_content)

    # Return the path for confirmation or further use
    return file_path

response_schemas = [
    ResponseSchema(
        name="Title",
        description="Replace [Main Subject] with the main theme or innovation of the content."
    ),
    ResponseSchema(
        name="Introduction",
        description="Fill out [Feature Name], [System/Service/Technology], [Primary Purpose], [Key Components], and [Main Benefit or Outcome] based on your content."
    ),
    ResponseSchema(
        name="Main Features",
        description="List and describe the main features or components from your JSON. Replace [Feature 1 Name], [Feature 2 Name], [Feature 3 Name], etc., with actual feature names and descriptions."
    ),
    ResponseSchema(
        name="Case Study or Example",
        description="Describe a real-world application or case study where this feature or system was implemented, including the outcomes."
    ),
    ResponseSchema(
        name="Importance and Benefits",
        description="Explain why this innovation is important and list the benefits it provides."
    ),
    ResponseSchema(
        name="Future Directions",
        description="Provide insights into the future plans or directions for this feature or technology."
    ),
    ResponseSchema(
        name="Call to Action",
        description="End with a call to action, encouraging readers to learn more, with a link to additional resources or a full article."
    ),
    ResponseSchema(
        name="Hashtags",
        description=" Replace [YourHashtag1], [YourHashtag2], [YourHashtag3] with relevant hashtags for your post"
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

chat_model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5, openai_api_key=key["openai_key"])

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""
            As an expert in distilling complex scientific research into digestible content, you are tasked with crafting a LinkedIn post that makes the key points from a research paper engaging and accessible to a mixed audience including industry professionals, academics, and laypeople. The summary should adhere to the following guidelines:

            - **Headline**: Create a compelling headline that encapsulates the essence and novelty of the research.

            - **Introduction**: Provide a brief overview of the research paper and its significance.

            - **Algorithmic Innovations**:
            - Insights on novel algorithms, optimizations, regularization methods, loss functions, and architectural modifications.

            - **Empirical Results**:
            - Insights on performance metrics, benchmark comparisons, analysis of results, and visualization of model outputs.
            
            - **Theoretical Analyses**:
            - Insights on convergence proofs, generalization bounds, complexity analysis, and theoretical guarantees.

            - **Applications and Use Cases**:
            - Insights on real-world applications, case studies, domain-specific challenges, and opportunities.
            
            - **Datasets and Data Preprocessing**:
            - Insights on dataset descriptions, data preprocessing techniques, data augmentation, and handling of imbalanced or noisy data.
            
            - **Ethical and Societal Implications**:
            - Insights on fairness, bias, discrimination, privacy concerns, and broader societal impacts.

            - **Future Directions and Open Problems**:
            - Insights on current limitations, future research directions, unresolved issues, and suggestions for improvement.
            
            - **Engagement**: End with a thought-provoking question, a suggestion for further discussion, or a call to action to encourage dialogue.

            Provide this engaging summary based on the following details and formatting instructions:

            Research Paper Summary: {research_summary}
            URL: {pdf_link}

            Remember, the ultimate goal is to make the research understandable and compelling to a wide range of readers. Your summary should invite curiosity and conversation, making complex ideas approachable and intriguing. 
            Only use the provided context to generate the summary. Strictly adhere to the output format.
            
            Use the format instructions below to ensure your post is consistent, clear, and captivating:
            
            {format_instructions}
            
            Remember, the ultimate goal is to make the research understandable and compelling to a wide range of readers. Your summary should invite curiosity and conversation, making complex ideas approachable and intriguing. 
            Only use the provided context to generate the summary. Strictly adhere to the output format. Ensure the generated content sticks within 3000 chracters. 
        """)
    ],
    input_variables=["research_summary", "pdf_link"],
    partial_variables={"format_instructions": format_instructions}
)

val_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
val_format_instructions = val_output_parser.get_format_instructions()

validation_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""
        Ensure the LinkedIn post follows the prescribed structure and meets quality standards. Apply these validation steps:

        1. Confirm the output is structured as a dictionary with keys corresponding to each section of the post: 'headline', 'introduction', 'algorithmic_innovations', 'empirical_results', 'theoretical_analyses', 'applications_and_use_cases', 'datasets_and_data_preprocessing', 'ethical_and_societal_implications', 'future_directions_and_open_problems', and 'engagement'. Each key should map to a non-empty string value.
        2. Ensure the 'headline' is compelling and encapsulates the research's main message.
        3. Verify the 'introduction' offers a succinct yet informative overview of the research's significance.
        4. Check that each section ('algorithmic_innovations', 'empirical_results', etc.) contains relevant and insightful information.
        5. Make sure the 'engagement' section includes a thought-provoking question or a call to action to prompt reader interaction.
        6. Review the output for syntax errors or trailing commas that could interfere with JSON parsing. Ensure proper use of quotes, brackets, and commas.
        7. Attempt to parse the output as JSON (or as a Python dictionary). Confirm there are no parsing errors, indicating that the structure conforms to JSON/dictionary syntax standards.
        8. If all conditions are met, the output is correctly formatted and ready for use.

        Apply the above checks to the following output, ensuring it adheres to the provided structure and can be effectively parsed as JSON or a Python dictionary:

        {answer}

        Use these formatting guidelines to ensure the post is consistent, clear, and engaging:

        {val_format_instructions}
        """)
    ],
    input_variables=["answer"],
    partial_variables={"format_instructions": val_format_instructions}
)

date_str = datetime.datetime.today().strftime("%Y-%m-%d")
#date_str = "custom"
folder_path = f"linkedin_posts/{date_str}"  # Adjust the folder path as needed

#os.makedirs(f"linkedin_post_md/{date_str}")

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Load JSON content from the file
        with open(file_path, "r") as f:
            inference = json.load(f)
        
        # Extract research summary and PDF link
        research_summary = str(inference.pop("Summary", None))
        pdf = filename.split("_")[-1]
        link = f"https://arxiv.org/pdf/{pdf}.pdf"
        print(link)
        
        # Format the prompt
        _input = prompt.format_prompt(research_summary=research_summary, pdf_link=link)
        
        # Generate output from chat model
        output = chat_model(_input.to_messages())
        
        # Convert dictionary to Markdown content
        formatted_markdown = json_to_linkedin_post(ast.literal_eval(output.content.split("json\n")[1].replace("\n```","")))
        
        # Save the Markdown content into a file
        save_markdown_file(formatted_markdown, filename, f"linkedin_post_md\\{date_str}")
        
        print(f"Markdown file saved successfully for {filename}.")
