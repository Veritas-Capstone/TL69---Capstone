# Set this into class when doing integration with other modules
# For now, we can run outside class for testing functionality
# Also might need to add Ollama to our requirements.txt file, although it could use a general update lol
# but why does Madhav have a setup env file for his ollama environment


import ollama


# class ClaimExtractor:
#     def __init__(self, model="mistral"):
#         self.model = model

### Variations of the prompt to test with text: The Earth revolves around the Sun.
# Extract atomic, self-contained, independently verifiable claims.
# Split compound statements into separate claims.
# Return JSON list.
# Returns: {"Claim 1": "The Earth revolves", "Claim 2": "around the Sun"}

# Extract atomic, independently verifiable claims.
# Split compound statements into separate claims.
# Return JSON list.
# Returns: {"Claim 1": "The Earth revolves", "Claim 2": "around the Sun"}

# Extract atomic, verifiable claims.
# Split compound statements into separate claims.
# { "Atomic Claims": ["The Earth revolves around the Sun."],
#   "Compound Statements Split": [
#     "The Earth revolves around its axis.",
#     "The Earth orbits the Sun."
#   ]
# }

### Parsing Text
# In 1969, NASA successfully landed the Apollo 11 spacecraft on the Moon. 
# Neil Armstrong became the first human to walk on the lunar surface, and Buzz Aldrin joined him shortly after. 
# The mission was launched from Kennedy Space Center in Florida on July 16, 1969. 
# The Saturn V rocket used for the launch remains one of the most powerful rockets ever built. 
# The Moon landing was broadcast live to an estimated 600 million people worldwide. 

# Some critics argue that the Moon landing was staged, but no credible evidence has been found to support this claim. 
# The United States spent approximately $25.4 billion on the Apollo program between 1961 and 1973. 
# The Apollo program resulted in significant technological advancements, including improvements in computing and materials science. 
# The Soviet Union never successfully landed a cosmonaut on the Moon. 
# Today, NASA is developing the Artemis program, which aims to return humans to the Moon by the late 2020s.

passage_options = [
"The Earth revolves around the Sun.",
"""In 1969, NASA successfully landed the Apollo 11 spacecraft on the Moon. 
Neil Armstrong became the first human to walk on the lunar surface, and Buzz Aldrin joined him shortly after. 
The mission was launched from Kennedy Space Center in Florida on July 16, 1969. 
The Saturn V rocket used for the launch remains one of the most powerful rockets ever built. 
The Moon landing was broadcast live to an estimated 600 million people worldwide. 

Some critics argue that the Moon landing was staged, but no credible evidence has been found to support this claim. 
The United States spent approximately $25.4 billion on the Apollo program between 1961 and 1973. 
The Apollo program resulted in significant technological advancements, including improvements in computing and materials science. 
The Soviet Union never successfully landed a cosmonaut on the Moon. 
Today, NASA is developing the Artemis program, which aims to return humans to the Moon by the late 2020s."""
]

parsing_prompt_options = [
"""Extract atomic, self-contained, independently verifiable claims.
Split compound statements into separate claims.
Return JSON list.""",
]

prompt = f"""
{parsing_prompt_options[0]}

Passage:
{passage_options[1]}
"""


print(prompt)
# List of models: [mistral, mixtral, gemma:7b, llama3, phi3]
response = ollama.chat(
    model='mixtral',
    messages=[
        {
            'role': 'user', 
            'content': prompt
        }
    ],
    options={
        "temperature": 0
    },
    format='json'
)

print(response['message']['content'])



# Metrics
# time
# accuracy
# other precision/recall metrics
# going to have to eyeball test this
# replicability is a huge one, if I run the same prompt does it give me the same claims, becausae if so I can fine tune, if there's variability that might be an issue



# If running locally, not the following ps commands
# ollama stop mistral # stops it in case you're moving onto some other task
# ollama ps   # checks if the ollama process is still running




# Some other models I might want to try out later
# ollama pull llama3
# ollama pull gemma:7b
# ollama pull phi3
# ollama pull mixtral                       the problem I have with this one is unless we can hook it up to cloud, it requires more ram which we might not be able to demo on a laptop