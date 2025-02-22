# food-insecurity-risk-mining
Automatic named entity recognition pipeline to identify possible drivers of food insecurity in news articles written in French language 🇫🇷. The project aims to support the event extraction (EE) task using sentiment analysis of relevant sentences and link the TIME and LOCATION entities to each event's mention.

Test our app in Google Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tetis-nlp/food-insecurity-risk-mining/blob/main/TETIS_foodinsecurity_analyzer.ipynb)
# Intro: Analyzer of text related to Food Security

🎯 **Goal**: Analyze the input text to identify the food insecurity risk factors, the geographical scope, and relevant named entities.

✅ **Tasks**:
* NER on locations, time, and organizations.
* NER on risk factors from expert's lexicon.
* Sentiment analysis of neutral terms related to prices, food production, and farming materials.

⭕ **Pending tasks**
* Entity linking "Risk factor" - DATE - DURATION - PLACE

🐱‍👤 **GitHub repository**: https://github.com/tetis-nlp/food-insecurity-risk-mining/

📄 **Dataverse with output data and thematic lexicon**: https://doi.org/10.57745/1PISWK

**Team members**:

| Member      | Affiliation            | Role |
|-------------|------------------------|-------------|
| [Nelson JAIMES-QUINTERO](https://github.com/NelsonJQ/)    | INRAE / TETIS - Univ. de Strasbourg      |Author|
| [Maguelonne TEISSEIRE](https://umr-tetis.fr/index.php/fr/equipe-misca/maguelone-teisseire) | INRAE / TETIS            |Supervisor|
| [Sarah VALENTIN](https://www.scopus.com/authid/detail.uri?authorId=57203356039)  | CIRAD / TETIS |Supervisor|

# Citation
```
@data{1PISWK_2024,
author = {Jaimes-Quintero, Nelson and Teisseire, Maguelonne and Valentin, Sarah},
publisher = {Recherche Data Gouv},
title = {{Corpus de journaux en français sur la sécurité alimentaire au Burkina Faso et Sénégal annotés en entités nommées et analyse de sentiment}},
UNF = {UNF:6:bIwyM4z47x2MBnhmAR+pHw==},
year = {2024},
version = {V1},
doi = {10.57745/1PISWK},
url = {https://doi.org/10.57745/1PISWK}
}
```
**Publication date**: 07-2024

**Research lab**: TETIS - [Maison de la télédétection](https://www.teledetection.fr/index.php/en/organizations/umr-tetis) (Montpellier, FRANCE)

# 💡 1. Basic concepts
## 1.1 What is food security?
Is food... 📦available? 💸🚚accessible? 🤢safe? 🏜️sustainable?

A more technical definition is proposed by the High Level Panel of Experts on Food Security and Nutrition (HLPE-FN) of the Committee on World Food Security ([Rome, 2020](https://www.fao.org/cfs/cfs-hlpe/publications/hlpe-15/en)) :

![image](https://github.com/tetis-nlp/food-insecurity-risk-mining/assets/73116221/08d3e730-28c6-4b71-bba6-25731b5d82fb)

## 1.2 What causes food insecurity?
Many factors of different dimensions such as crop yields, weather conditions, food prices, farming material's prices and availability, invasive species, irrational land use, terrorism, etc. ([See a bibliographical review of the many causes and how they are measured](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10161169/))
![image](https://github.com/tetis-nlp/food-insecurity-risk-mining/assets/73116221/c28771e7-8f32-4a5f-92cc-de5215e77532)

## 1.3 How can we use NLP to detect risk factors?
We can analyze press articles to monitor the apparition or evolution of events linked to possible causes of food insecurity. This will help to explain why a specific region is vulnerable to food insecurity.

Some possible causes are easy to detect, for instance:
* "inflation", "war", "bad harvest", "earthquake";
* but some other possible causes might be expressed in many different ways "the crops of **{CEREAL: sorgho|millet|oat}** were **{MODIFIER: severely|incredibly}** **{negative VERB: affected|destroyed}** by the **{AGENT: rain|ProperNounOfCriminals}**", even with idioms.


Given this complexity in how media talk about events, we propose:
* to identify words with a high probability of being a direct or indirect cause of food insecurity (war, terrorism, inflation, natural disasters, etc.), and
* to identify sentences with possible but not probable causes of food insecurity: ("harvest" -> Is something negative happening to the harvests?)

![image](https://github.com/tetis-nlp/food-insecurity-risk-mining/assets/73116221/b3bebd8f-6988-42fa-829d-03bcf866b57a)


# 2. Parameters of our main function

Our main function `analyze_food_sentiment()` has the following parameters:

| Parameter            | Description                                                                                                                                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_text`         | A string containing the text to be analyzed for food sentiment.                                                                                                                                                                 |
| `spacy_model`        | The spaCy model to be used for language processing. The default is 'fr_core_news_lg', which is the French large model. In the future, you'll be able to choose your own spaCy model even for other languages.                      |
| `polarity_calculator`| default = "Ollama", choose the model that will analyze the polarity of the relevant sentences. The options are "Ollama", 'vader', 'transformers', 'isdm': 1️⃣⭐ The option **Ollama** uses open-source models that can be loaded using a local Ollama server [installable here](https://youtu.be/IxyoNpK3oYI) (this server doesn't work on Google Colab). 2️⃣⭐ The option **vader** is a rule-based polarity calculator that is already installed in the requirements on this notebook (it works on GColab). 3️⃣⭐ The option **transformers** are BERT-type pre-trained language models that can be automatically downloaded and loaded from HuggingFace by setting its name in the variable `transf_model`. 4️⃣⭐ The option **isdm** is restricted to users that have an API token of the ISDM-Chat that uses a remote server to query a `mixtral:8x7b-instruct-v0.1-q5_0` model (this option can be adapted to [groq tokens available here](https://www.analyticsvidhya.com/blog/2024/05/how-to-instantly-access-llama-on-groq/)). |
| `only_negative`      |  Set it as True to return only sentences tagged as negative. Set False to show all sentences (positive and neutral included). default = False                                                                                        |
| `theme_clustering`   | Set to `True` to find the theme of the extracted concept, helping visualize six big themes related to possible risk factors of food insecurity (e.g., agriculture, economic, sociopolitical, environmental). Default is `False`. |
| `transf_model`       | The name of a HuggingFace model trained for Named Entity Recognition (NER) tasks. Default for French language is "ac0hik/Sentiment_Analysis_French".                                                                          |
| `Ollama_model`       | The name of an Ollama model downloaded in your local Ollama server. Default is "phi3:3.8b-mini-instruct-4k-q4_K_M".                                                                                                           |
| `run_heideltime`     | Set to `True` to run NER on time expressions using the HeidelTime library (requires Perl and Java), or `False` (default) to use the Timexy library (less accurate). Both extract time entities using the Timex3 international standard. |
| `reference_date`     | If `run_heideltime = True`, the reference date will be used to better identify relative time expressions. For example, if the reference is "2021-08-24", the entity "hier" (yesterday) will have the time value = "2021-08-23". If `reference_date = None` (default), "hier" will be extracted but the time_value will not be useful (XXXX-XX-XX). Please try to provide the reference date in the format "YYYY-MM-DD" to avoid errors.                                    |


**Output**
str -> dict:
```
{'polarizedEntities': [
{'start_char': 3, 'end_char': 9, 'label': 'VIOLENCE', 'theme': 'SOCIOPOLITIQUE', 'text': 'guerre'}
{'start_char': 223, 'end_char': 230, 'label': 'FOOD', 'theme': 'FOOD', 'text': 'légumes'}
{'start_char': 284, 'end_char': 288, 'label': 'LOC', 'theme': 'LOC', 'text': 'Inde'}
{'start_char': 354, 'end_char': 362, 'label': 'DURATION', 'time_value': 'P6M', 'text': 'six mois'}
],
'polarizedSentences' : [
{'cited_factors': ["monte le coût de l'engrais"],
 'concepts': ['augmentation des prix des intrants agricoles'],
 'end_char': 58,
 'polarity_label': 'negative',
 'score': -1,
 'sentence': "La guerre est mauvaise car il monte le coût de l'engrais.",
 'start_char': 0,
 'themes': ['economique']}
{'cited_factors': [''],
 'concepts': [],
 'end_char': 219,
 'polarity_label': 'neutral',
 'score': 0,
 'sentence': 'Si les prix du coton ont chuté de 25% au mois de septembre pour '
             'clôturer le 30 septembre à 85,34 cents la livre, le moral des '
             'participants n’était pas en berne.',
 'start_char': 58,
 'themes': None}]
 ```

# 3. We can also visualize the output

* `visualize_entities(input_text: str, output_from_main_function: dict)` for display only the entities, not sentences, from a single document.
* `visualize_entities_overlapping(input_text: str, output_from_main_function: dict, export=False)` for displaying entities and analyzed sentences from a single document, it avoid erros with overlapping. Set `export=True` to create an HTML document. 
* `apply_visualize_entities_overlapping(df: DataFrame)` for displaying and exporting a single HTML with all the output of `apply_food_sentiment_analysis(df)` (used for a dataset containing many documents).
