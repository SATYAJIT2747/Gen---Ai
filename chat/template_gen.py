from langchain_core.prompts import PromptTemplate
template  = PromptTemplate.from_template(
    "Summarize the paper {paper} in a {summary_type} with a {style} style and a length of {length}."  , validate_template=True
) 
template.save("template.json")
