


def news_entities_extractor(keyword):
    response_schemas = [
        ResponseSchema(name="Positive News", description="List of Positive News Articles about **{}**, if any else return no news".format(keyword)),
        ResponseSchema(name="Positive Aspect", description="List of Topics discussed by the Positive News Articles about **{}**, if any else return no news".format(keyword)),
        ResponseSchema(name="Negative News", description="List of Negative News Articles about **{}**, if any else return no news".format(keyword)),
        ResponseSchema(name="Negative Aspect", description="List of Topics discussed by the Negative News Articles about **{}**, if any else return no news".format(keyword)),
        ResponseSchema(name="Neutral News", description="List of Neutral News Articles about **{}**, if any else return no news".format(keyword)),
        ResponseSchema(name="Neutral Aspect", description="List of Topics discussed by the Neutral News Articles about **{}**, if any else return no news".format(keyword)),
        ResponseSchema(name="Publish Date", description="List of dates the articles was published, if any else return no news"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5, openai_api_key = "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")

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""
            You are a helpful assistant who evaluates news articles for the day split by "\n\n", identifies positive and negative news about **{}** and summarizes it in a concise format.
            {format_instructions}
            News Articles: {question}""")
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    topic_dict = {}
    token_check = 0
    clean_transcript = input_to_llm_two[:16000]
    transcription_token_length = num_tokens_from_string(clean_transcript, "gpt-3.5-turbo-16k")
    if transcription_token_length<16000:
      _input = prompt.format_prompt(question=clean_transcript)
      output = chat_model(_input.to_messages())
      return(output_parser.parse(output.content))
    else:
      print("Token Limit Exceeded. Summarizing and evaluating")
      complete_content_chunks = split_into_chunks(clean_transcript,16000)
      summarized_transcription = []
      for chunk in complete_content_chunks:
        doc =  Document(page_content=chunk, metadata={"source": "transcription"})
        summ_chain = load_summarize_chain(chat_model, chain_type="stuff")
        transcription = summ_chain.run([doc])
        summarized_transcription.append(transcription)
      summarized_transcription = ' '.join(summarized_transcription)
      summ_chain = load_summarize_chain(chat_model, chain_type="stuff")
      doc_summarized =  Document(page_content=summarized_transcription, metadata={"source": "summarized_transcription"})
      summarized_transcription_updated = summ_chain.run([doc_summarized])
      _input = prompt.format_prompt(question=summarized_transcription_updated+'\nURL: '+str(list(self.transcriptions.keys())[0]))
      output = chat_model(_input.to_messages())
      topic_dict[output_parser.parse(output.content)['URL']] = output_parser.parse(output.content)
      return(output_parser.parse(output.content))

def split_into_chunks(text, chunk_size=4000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
