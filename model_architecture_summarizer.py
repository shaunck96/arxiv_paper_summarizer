from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import tiktoken
import json
import os
import ast

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

response_schemas = [
    ResponseSchema(
        name="Layer Architecture",
        description="Detailed insights on the arrangement and types of layers within the deep learning model, including input, hidden, and output layers. Discuss the specific roles and configurations of convolutional, pooling, recurrent, normalization, and fully connected layers."
    ),
    ResponseSchema(
        name="Activation Functions",
        description="Comprehensive details on the types of activation functions used across different layers of the network. Explain the rationale behind choosing specific activation functions and their impact on the learning process and network behavior."
    ),
    ResponseSchema(
        name="Optimization Techniques",
        description="In-depth information on optimization strategies, including gradient descent variations, learning rate schedules, and other algorithmic enhancements aimed at improving model training efficiency and convergence rate."
    ),
    ResponseSchema(
        name="Loss Functions",
        description="Exhaustive descriptions of the loss functions applied during the model's training phase. Include discussions on why particular loss functions were chosen and their contributions to the model's learning objectives."
    ),
    ResponseSchema(
        name="Regularization Methods",
        description="Detailed insights on regularization techniques implemented to reduce overfitting and improve model generalization. This could include methods like dropout, L1/L2 regularization, and data augmentation strategies."
    ),
    ResponseSchema(
        name="Model Innovations",
        description="Elaborate on unique, innovative aspects of the model's architecture. This could encompass custom layers, novel data processing techniques, or distinctive architectural adjustments that set this model apart from conventional approaches."
    ),
    ResponseSchema(
        name="Model Evaluation Metrics",
        description="Detailed explanations of the metrics and methodologies used to assess the model's performance. Discuss the relevance of chosen metrics in the context of the model's intended application and the interpretation of these metrics in evaluating model success."
    ),
    ResponseSchema(
        name="Architectural Diagrams and Visualizations",
        description="Insights on any available architectural diagrams, flowcharts, or other visualizations that help in understanding the model's structure and data flow."
    ),
    ResponseSchema(
        name="Implementation Details",
        description="Details on the implementation aspects of the model, including software frameworks, hardware requirements, and execution environments."
    ),
    ResponseSchema(
        name="Challenges and Limitations",
        description="Discussion on the encountered challenges, limitations, and trade-offs during the model's design and training processes."
    ),
    ResponseSchema(
        name="Comparative Analysis",
        description="Insights on how this model's architecture compares to other state-of-the-art solutions. Highlight the advantages and disadvantages in terms of performance, efficiency, and applicability."
    ),
    ResponseSchema(
        name="Future Directions",
        description="Ideas and suggestions for future improvements, potential research directions, and unexplored avenues within the model's architectural framework."
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

chat_model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5, openai_api_key=key["openai_key"])

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""
        Craft a LinkedIn post summarizing the deep learning architecture from a research paper, tailored for a mixed audience. The summary should adhere to the following structure and guidelines:
        
        - **Headline**: Summarize the essence and innovation of the model architecture.
        - **Introduction**: Give an overview of the importance of the model architecture.
        - **Layer Architecture**:
            - Insights on types, configurations, and sequencing of layers.
        - **Activation Functions**:
            - Details on the activation functions used.
        - **Optimization Techniques**:
            - Information on optimization algorithms and parameters.
        - **Loss Functions**:
            - Descriptions of the loss functions utilized.
        - **Regularization Methods**:
            - Techniques to prevent overfitting.
        - **Model Innovations**:
            - Unique or novel architectural features.
        - **Model Evaluation Metrics**:
            - Metrics and validation techniques for performance evaluation.
        - **Engagement**: Conclude with a question or call to action for dialogue.
        
        Provide this engaging summary in at least 300 words based on the following details and formatting instructions:
        
        Research Paper Summary: {hist}
        URL: {link}
        
        Use the format instructions below to ensure your post is consistent, clear, and captivating:
        
        {format_instructions}
        
        Remember, the ultimate goal is to make the research understandable and compelling to a wide range of readers. Your summary should invite curiosity and conversation, making complex ideas approachable and intriguing. 
            Only use the provided context to generate the summary. Strictly adhere to the output format.
        """)
    ],
    input_variables=["hist", "link"],
    partial_variables={"format_instructions": format_instructions}
)

val_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
val_format_instructions = val_output_parser.get_format_instructions()

validation_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("""
        Validate the LinkedIn post's structure and quality, ensuring it is focused on deep learning architecture and adheres to these validation steps:

        - Check for structured output as a dictionary with specific keys: 'headline', 'introduction', 'layer_architecture', 'activation_functions', 'optimization_techniques', 'loss_functions', 'regularization_methods', 'model_innovations', 'model_evaluation_metrics', 'engagement'.
        - Confirm the 'headline' succinctly encapsulates the model's architectural novelty.
        - Ensure 'introduction' provides an informative overview of the architecture's significance.
        - Review each section for relevant, deep learning-specific information.
        - Verify the 'engagement' prompts reader interaction.
        - Check syntax and JSON/dictionary structure for parsing accuracy.
        - If all conditions are met, the post is well-structured and focused on architecture.

        Apply the above checks to the following output, ensuring it adheres to the provided structure and can be effectively parsed as JSON or a Python dictionary:

        {answer}

        Use these formatting guidelines to ensure the post is consistent, clear, and engaging:

        {val_format_instructions}
        """)
    ],
    input_variables=["answer"],
    partial_variables={"format_instructions": val_format_instructions}
)

date_str = "2024-03-20"
with open("C:\\Users\\307164\\Desktop\\Huggingface_Paper_Extractor\\inference\\model_architecture\\{}\\inference.json".format(date_str), "r") as f:
    inference = json.load(f)

for index in range(len(list(inference.keys()))):
    link = list(inference.keys())[index]
    input_to_llm = inference[link]

    #if num_tokens_from_string(input_to_llm, "gpt-3.5-turbo-16k") < 16000:
    _input = prompt.format_prompt(hist=input_to_llm, 
                                    link=link)
    output = chat_model(_input.to_messages())
    # Attempt to parse the output; handle parsing errors.
    try:
        # Attempt to convert the string representation of a dictionary back into a dictionary.
        output_dict = ast.literal_eval(output.content.replace("```json\n", "").replace("\n```", ""))
    except ValueError as e:
        # If there's a parsing error, log it and try the alternative method.
        try:
            _input = validation_prompt.format_prompt(answer=output)
            output = chat_model(_input.to_messages())
            output_dict = ast.literal_eval(output.content.replace("```json\n", "").replace("\n```", ""))
        except ValueError as e:
            # If there's a second parsing error, return an error message.
            output_dict = {"Headline":"Content couldn't be parsed into a dictionary."}

    output_dict['Summary'] = input_to_llm
    print(link)
    print("\n\n")
    # Create directory if it does not exist
    directory_path = f"linkedin_posts/model_architecture/{date_str}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Define the filename based on the link
    filename = link.split("\\")[-1]  # Changed from backslash to forward slash for URL
    file_path = f"{directory_path}/post_{filename}.json"  # Correct the file path

    # Write the output in JSON format
    with open(file_path, "w") as f:
        json.dump(output_dict, f)  # Use json.dump() to write the dict directly into the file

    print(f"File saved: {file_path}")

