# Tutorial: Extracting Information from Text Using OpenAI's API


This tutorial demonstrates how to use OpenAI's API to extract information from a text input. We will go through the steps of setting up the OpenAI client, making requests to the API, and processing the responses. The example involves summarizing a news article and extracting structured information from it.

## Installation and Imports
First, we need to install the required library and import necessary modules.
!pip install openai -q
# Import required libraries
from openai import OpenAI
#from google.colab import userdata
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import textwrap

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
## Setting Up the OpenAI Client

We will set up the OpenAI client using a custom API key and base URL.  `userdata.get('TOGETHER_API_KEY')` is used to securely access your API key stored in Google Colab's user secrets.  This avoids hardcoding your API key directly in the notebook.
# Setup OpenAI client with custom API key and base URL
#TOGETHER_API_KEY = userdata.get('TOGETHER_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
### Summarizing Text

We will call the language model to summarize a given text into a single sentence.
# Create client
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
We will use a French news article as our first example input. This article discusses political reactions to the Ukraine war.
text_1 = """
Vous pouvez partager un article en cliquant sur les icônes de partage en haut à droite de celui-ci.
La reproduction totale ou partielle d’un article, sans l’autorisation écrite et préalable du Monde, est strictement interdite.
Pour plus d’informations, consultez nos conditions générales de vente.
Pour toute demande d’autorisation, contactez syndication@lemonde.fr.
En tant qu’abonné, vous pouvez offrir jusqu’à cinq articles par mois à l’un de vos proches grâce à la fonctionnalité « Offrir un article ».

https://www.lemonde.fr/international/live/2025/03/03/en-direct-guerre-en-ukraine-pour-donald-trump-les-etats-unis-ont-des-problemes-plus-urgents-que-de-s-inquieter-de-poutine_6572748_3210.html

L’altercation entre Volodymyr Zelensky et Donald Trump a été délibérément provoquée par les Etats-Unis, selon Friedrich Merz

Lors d’une conférence de presse, lundi, à Hambourg, Friedrich Merz, le candidat de l’alliance CDU/CSU à la chancellerie, a déclaré, après des consultations avec les instances dirigeantes de la CDU à Berlin, qu’il avait regardé la scène de l’altercation entre Volodymyr Zelensky et Donald Trump. « A mon avis, il ne s’agit pas d’une réaction spontanée aux interventions de Zelensky, mais manifestement d’une escalade délibérément provoquée lors de cette rencontre dans le bureau Ovale. »

« Il y a une certaine continuité dans ce que nous voyons actuellement de Washington dans la série d’événements des dernières semaines et des derniers mois, y compris la présence de la délégation américaine à Munich à la conférence sur la sécurité », a-t-il poursuivi. « Je plaide pour que nous nous préparions au fait que nous devrons faire beaucoup, beaucoup plus pour notre propre sécurité dans les années et les décennies à venir », a ajouté le futur chancelier.

Néanmoins, il souhaite que « tout soit mis en œuvre afin de maintenir les Américains en Europe », dans un contexte de spéculations selon lesquelles Trump pourrait retirer une partie des troupes américaines d’Allemagne. Le futur chancelier a précisé qu’il n’avait pas l’intention de se rendre aux Etats-Unis pour l’instant et qu’il ne le ferait qu’après une éventuelle élection en tant que chancelier par le Bundestag.

Par ailleurs, il a défendu le chancelier Olaf Scholz (SPD) contre les critiques concernant son rôle lors du sommet des dirigeants occidentaux à Londres. « Il n’est pas surprenant que l’Allemagne ne soit pas pleinement perçue et prise au sérieux sur la scène internationale en ce moment, a-t-il déclaré. Tout autre chancelier dans sa situation – ayant perdu sa majorité parlementaire et étant en transition vers un nouveau gouvernement – connaîtrait la même difficulté. »

Il a souligné que lui et Olaf Scholz s’efforcent d’« introduire la position allemande dans les négociations internationales et européennes en étroite coordination ». Toutefois, il estime qu’il « serait souhaitable que l’Allemagne participe bientôt à ces discussions avec un chef de gouvernement élu et disposant d’une majorité au Bundestag ».

3/3 2025
"""
Here, we call the LLM to summarize the French text in one sentence, requesting the output in English. We are using the `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` model for this task.
# Call the LLM with the JSON schema
chat_completion = client.chat.completions.create(
    #model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",

    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Summarize the following French text in one sentence in English: " + text_1 ,
        },
    ],
)

output = chat_completion.choices[0].message.content
The `textwrap.fill` function enhances readability by wrapping the output summary to a specified width (80 characters).
print(textwrap.fill(output, width=80))
## Creating a User Object

Let's start with a simple example to understand structured output. We will create a `User` object using a predefined schema.

We define a schema using `pydantic` `BaseModel`. `BaseModel` allows us to define data structures with type validation. `Field` provides metadata and descriptions for each field, aiding the language model in understanding the desired output format. `model_json_schema()` automatically generates a JSON schema from the Pydantic model, which is used to instruct the LLM about the expected JSON structure.
# Define the schema for the User object.
class User(BaseModel):
    name: str = Field(description="user name")
    address: str = Field(description="address")
We call the LLM to create a user object in JSON format based on the `User` schema.  `response_format={"type": "json_object", "schema": User.model_json_schema()}` tells the API to expect a JSON object as a response, structured according to the `User` schema.
# Call the LLM to create a User object in JSON format
chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    response_format={"type": "json_object", "schema": User.model_json_schema()},
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that answers in JSON.",
        },
        {
            "role": "user",
            "content": "Create a user named Alice, who lives in 42, Wonderland Avenue. Output in JSON.",
        },
    ],
)

created_user = json.loads(chat_completion.choices[0].message.content)
print(json.dumps(created_user, indent=2))
## Extracting Article Details - Using a Generalized Schema

Now we move to a more complex task: extracting information from news articles. We will use a generalized schema `ExtractScheme` designed to capture key details from various news articles about different events. This schema aims for flexibility to handle diverse news content.
class ExtractScheme(BaseModel):
    title: str = Field(description="Title of the news article")
    publication_date: str = Field(description="Date when the article was published. If not explicitly mentioned, infer from article content if possible.")
    main_event: str = Field(description="Primary event or topic discussed in the article")
    event_summary: str = Field(description="A brief summary of the event or article's main points")
    entities_involved: List[str] = Field(description="Organizations, countries, or key entities involved in the event")
    key_people: List[str] = Field(description="Key people or figures mentioned in relation to the event")
    relevant_locations: Optional[List[str]] = Field(description="Locations that are central to the event, if any")
    key_developments: Optional[List[str]] = Field(description="Key developments or actions that have occurred or are expected")
    potential_impact: Optional[List[str]] = Field(description="Potential impacts or consequences of the event")
    keywords: List[str] = Field(description="Key terms or phrases that are central to the article")
We will first use our French text (`text_1`) about the Ukraine war and political reactions. The system message instructs the LLM to act as an AI for structured information extraction from news articles, following the `ExtractScheme`. The user message provides `text_1` and requests information extraction in English JSON format.
# Call the LLM to extract information from text_1 using ExtractScheme
chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    response_format={"type": "json_object", "schema": ExtractScheme.model_json_schema()},
    messages=[
        {
            "role": "system",
            "content": "You are an AI model tasked with extracting structured information from a news article. Follow the schema provided below to extract the relevant details. You do not invent information that is not in the provided text. Output in English JSON format.",
        },
        {
            "role": "user",
            "content": "Extract article information from the following French text and output in English JSON format: " + text_1,
        },
    ],
)
The extracted output is loaded from the JSON response and printed in a user-friendly format. `ensure_ascii=False` ensures correct display of non-ASCII characters if present.
extracted_output = json.loads(chat_completion.choices[0].message.content)
print(json.dumps(extracted_output, ensure_ascii=False, indent=2))
You can inspect the generated JSON schema using `ExtractScheme.model_json_schema()` or its string representation. This schema guides the LLM's output formatting.
ExtractScheme.model_json_schema()
json_schema = str(ExtractScheme.model_json_schema())
In this example, we explicitly pass the JSON schema as a string in the user prompt. This is equivalent to using `response_format` and can be helpful for debugging or more direct prompt control.  The system message reinforces the need for JSON output only.
# Call the LLM to extract information from text_1 using schema string in prompt
chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    #response_format={"type": "json_object", "schema": CaseDetails.model_json_schema()}, # alternative method
    messages=[
        {
            "role": "system",
            "content": "You are an AI model tasked with extracting structured information from a news article. Follow the schema provided below to extract the relevant details. You do not invent information that is not in the provided text. You output JSON only in English. Nothing else.",
        },
        {
            "role": "user",
            "content": "Extract article information from the following French text and output in English JSON format: " + text_1 + " Use following JSON schema:" + json_schema,
        },
    ],
)
extracted_output = json.loads(chat_completion.choices[0].message.content)
print(json.dumps(extracted_output, ensure_ascii=False, indent=2))
## Extracting Article Details - Second Example (Technology News)

Let's use a second text example, this time an English article about Meta's AI chatbot app launch, to demonstrate the versatility of our `ExtractScheme`.
text_2 = """
Meta’s AI chatbot will soon have a standalone app
​
 Summarise
​
Emma RothFeb 28, 2025 at 12:05 AM GMT+1
STK043_VRG_Illo_N_Barclay_6_Meta
Meta is planning to launch a dedicated app for its AI chatbot, according to a report from CNBC. The Verge can also confirm that Meta is working on the standalone app. The new app could launch in the second quarter of this year, CNBC says, joining the growing number of standalone AI apps, including OpenAI’s ChatGPT, Google Gemini, and Microsoft Copilot.

Meta has already brought its AI chatbot across Facebook, Instagram, Messenger, and WhatsApp, but launching a standalone app could help the company reach people who don’t already use those platforms. Similar to rival chatbots, Meta AI can answer questions, generate images, edit photos, and more. It recently gained the ability to use its “memory” to provide better recommendations.

In a response to CNBC’s report, OpenAI CEO Sam Altman joked, “ok fine maybe we’ll do a social app.” Meta declined to comment.

Meta has ramped up its efforts to compete in the AI industry in recent months, with CEO Mark Zuckerberg announcing plans to invest up to $65 billion to further the company’s AI ambitions. The company also plans on holding an event dedicated to AI on April 29th.

Additional reporting by Alex Heath.

5 Comments5 New
"""
We use the same generalized `ExtractScheme` and prompt structure for `text_2`.  We also include an `assistant` example message to demonstrate the expected JSON output format to the LLM. This example is based on the *first* text example (`text_1`) and is crucial for guiding the LLM towards the correct output structure.
# Call the LLM to extract information from text_2 using ExtractScheme and example assistant message
chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    #response_format={"type": "json_object", "schema": CaseDetails.model_json_schema()}, # alternative method
    messages=[
        {
            "role": "system",
            "content": "You are an AI model tasked with extracting structured information from a news article. Follow the schema provided below to extract the relevant details. You do not invent information that is not in the provided text. You output JSON only in English. Nothing else.",
        },
        {
            "role": "user",
            "content": "Extract article information from the following text and output in English JSON format: " + text_2 + " Use following JSON schema:" + json_schema,
        },
                {
            "role": "assistant",
            "content": """{
  "title": "L’altercation entre Volodymyr Zelensky et Donald Trump a été délibérément provoquée par les Etats-Unis, selon Friedrich Merz",
  "publication_date": "March 3, 2025",
  "main_event": "Political reactions to an altercation between Volodymyr Zelensky and Donald Trump",
  "event_summary": "Friedrich Merz claims that the altercation between Zelensky and Trump was deliberately provoked by the U.S. and expresses concerns about US commitment to European security.",
  "entities_involved": [
    "United States",
    "Ukraine",
    "Germany",
    "CDU/CSU alliance"
  ],
  "key_people": [
    "Friedrich Merz",
    "Volodymyr Zelensky",
    "Donald Trump",
    "Olaf Scholz"
  ],
  "relevant_locations": [
    "Hambourg",
    "Berlin",
    "Munich",
    "Washington",
    "London"
  ],
  "key_developments": [
    "Friedrich Merz's press conference in Hambourg",
    "Consultations with CDU leadership in Berlin",
    "Merz's statement on US-Europe relations and German security",
    "Defense of Olaf Scholz's role at a summit in London"
  ],
  "potential_impact": [
    "Potential shift in US foreign policy under Trump",
    "Increased pressure on Europe to ensure its own security",
    "Speculation about US troop withdrawal from Germany",
    "Impact on German political landscape and leadership"
  ],
  "keywords": [
    "Ukraine",
    "Donald Trump",
    "Volodymyr Zelensky",
    "Friedrich Merz",
    "US foreign policy",
    "European security",
    "German politics"
  ]
}""",
        },
                {
            "role": "user",
            "content": "Extract article information from the following text and output in English JSON format: " + text_2 + " Use following JSON schema:" + json_schema,
        },
    ],
)
extracted_output = json.loads(chat_completion.choices[0].message.content)
print(json.dumps(extracted_output, ensure_ascii=False, indent=2))
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Define the extraction schema (same as in the notebook)
class ExtractScheme(BaseModel):
    #title: str = Field(description="Title of the news article")
    #publication_date: str = Field(description="Date when the article was published. If not explicitly mentioned, infer from article content if possible.")
    real_article: str = Field(description="Real article or scraping problem/artifact/copyright issue? - Select YES/NO only.")
    main_event: str = Field(description="Primary event or topic discussed in the article")
    event_summary: str = Field(description="A brief summary of the event or article's main points")
    entities_involved: List[str] = Field(description="Organizations, countries, or key entities involved in the event")
    key_people: List[str] = Field(description="Key people or figures mentioned in relation to the event")
    relevant_locations: Optional[List[str]] = Field(description="Locations that are central to the event, if any")
    key_developments: Optional[List[str]] = Field(description="Key developments or actions that have occurred or are expected")
    potential_impact: Optional[List[str]] = Field(description="Potential impacts or consequences of the event")
    keywords: List[str] = Field(description="Key terms or phrases that are central to the article")

# Setup OpenAI client
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# Load articles from local jsonl file
def load_articles_from_jsonl(file_path):
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            article_json = json.loads(line.strip())
            articles.append(article_json)
    return articles

# Path to your local jsonl file
jsonl_file_path = 'paraphrased_articles.jsonl' # Replace with your actual file path
articles_data = load_articles_from_jsonl(jsonl_file_path)

# Filter articles based on text presence and length
filtered_articles_data = []
for article in articles_data:
    if 'text' in article and isinstance(article['text'], str) and len(article['text']) >= 100:
        filtered_articles_data.append(article)

articles_data = filtered_articles_data # Replace original with filtered data
print(f"Number of articles after filtering: {len(articles_data)}")


extracted_data_table = []
json_schema = str(ExtractScheme.model_json_schema()) # Get JSON schema string

# Import necessary additional libraries
import pandas as pd
from tqdm.notebook import tqdm

# Iterate over articles and perform extraction with tqdm progress bar
for article in tqdm(articles_data[:10], desc="Processing Articles"): # Limiting to first 10 articles as requested
    article_text = article['text']
    original_title = article['title']
    original_date = article['date']

    try:
        chat_completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI model tasked with extracting structured information from a news article. Follow the schema provided below to extract the relevant details. You do not invent information that is not in the provided text. You output JSON only in English. Nothing else.",
                },
                {
                    "role": "user",
                    "content": f"Extract article information from the following text and output in English JSON format: {article_text} Use following JSON schema:" + json_schema,
                },
            ],
            response_format={"type": "json_object", "schema": ExtractScheme.model_json_schema()}, # enforce JSON output and schema
        )

        extracted_content_json = json.loads(chat_completion.choices[0].message.content)
        extracted_content = ExtractScheme(**extracted_content_json).dict() # Validate and convert to dict

        # Add original title and date to the extracted data for the table
        extracted_content['original_title'] = original_title
        extracted_content['original_date'] = original_date
        extracted_data_table.append(extracted_content)

        print(f"Extracted information for: {original_title}")

    except Exception as e:
        print(f"Error processing article: {original_title}. Error: {e}")
        extracted_data_table.append({'original_title': original_title, 'original_date': original_date, 'error': str(e)}) # Store error info

# Convert to pandas DataFrame
df = pd.DataFrame(extracted_data_table)

# Flatten list columns to make viewing easier
def flatten_list_columns(df):
    flattened_df = df.copy()
    list_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]

    for col in list_columns:
        # Convert lists to comma-separated strings
        flattened_df[col] = flattened_df[col].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and x else '')

    return flattened_df

# Flatten the dataframe and display the head
flattened_df = flatten_list_columns(df)
print("\nExtracted Data Table (Flattened):")
display(flattened_df.head())

# Optional: Save to CSV
flattened_df.to_csv('extracted_news_data_flattened.csv', index=False)

# Original output format (JSON)
print("\nExtracted Data Table (Original):")
for row in extracted_data_table:
    print(json.dumps(row, ensure_ascii=False, indent=2))
