# Fine_Tuned_Llama_2
Steps to Run the Model:

1. Make Sure you installed upgraded versions of all the necessary libraries and packages mention in Step 1 .
2. Create wandb account and create your login token , needed during training step.
3. Before coding, connect colab with GPU.
4. Now, simply copy paste the code in colab Notebook .
5. Run the Code.

Workflow:
1. Install Required Packages:
I need accelerate, peft, transformers, datasets and TRL to leverage the recent SFTTrainer. I used bitsandbytes to quantize the base model into 4bit. Also install einops as it is a requirement to load Falcon models.

2. Load Dataset::
I used ‘AlexanderDoria/novel17_test' dataset whic is extracted from a few hunder French novels published from 1600 to 1700
Dataset format: 
["### Human: Écrire un texte dans un style baroque, utilisant le langage et la syntaxe du 17ème siècle, mettant en scène un échange entre un prêtre et un jeune homme confus au sujet de ses péchés.### Assistant: Si j'en luis éton. né ou empêché ce n'eſt pas ſans cauſe vů que ſouvent les liommes ne ſçaventque dire non plus que celui de tantôt qui ne ſçavoit rien faire que des civiéresVALDEN: Jefusbien einpêché confeſſant un jour un jeune Breton Vallonqui enfin de confeſſion me dit qu'il avoit beſongné une civiere . Quoilui dis je mon amice peché n'eſt point écrit au livre Angeli que d'enfernommé la ſommedes pechez ,qui eſt le livre le plus déteſtable qui fut jamais fait& le plus blafphematoire d'autant qu'il eſt dédié à la plus femme de bien je ne ſçai quelle penitence te donner ; mais non mon amiquel goûty prenois-tu ? Mon fieur bon & delectable. Quoi!"]

3. Initialize Model and Tokenizer  
4. Configure LoRA (Low-Rank Adaptation)
5. Configure Training Arguments:
6. Initialize SFT Trainer:
7. Convert Model Norm Layers to Float32:
8. Model Training
9. Model Testing
10. Push Model to Hugging Face Model Hub
