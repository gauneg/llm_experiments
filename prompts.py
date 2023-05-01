prompt_dict = {
    
    'instructabsa_prompt-1': """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
Now complete the following example-
input: """,
    
    'self_for_flan-t5': """Extract aspect terms from the following input. input: 
    """,
    'aspect_sentiment': lambda aspect_term, sentence: f"""Given the aspect term and the sentence. Predict if the aspect term in the sentence has a positive, negative, neutral or conflict sentiment expressed on it.
aspect term: {aspect_term}
sentence: {sentence} """

}
