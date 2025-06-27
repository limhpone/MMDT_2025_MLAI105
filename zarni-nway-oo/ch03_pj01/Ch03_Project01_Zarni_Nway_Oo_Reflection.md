# Reflections on this project

For this project, I had the chance to work with one of the most knowledgeable and hardworking classmates in our class, Nu Wai. It was both exciting and overwhelming trying to keep up with the fast pace of development. While I was still trying to understand the concepts, she had already spotted flaws in the code and started working on the presentation slides and reports. It made me realize my own gaps, but at the same time, I was really impressed by how quickly things were moving.

We discussed some of the concepts in our Telegram group, but to be honest, I feel like I learned more from those discussions than I actually contributed. During the testing phase, we shared different ideas, and most of my work came from a suggestion by Phyoe Myat Oo, who mentioned testing with images related to Myanmar. That idea really stuck with me, so I started collecting Myanmar-specific images and ran different tests. I also used Claude AI to help speed up the process by generating graphs and reports that could support our model evaluations.

## So, what did you actually do?

- I started by trying to understand the concepts through Dr. Myo Thida's videos and Andrew Ng’s lectures.

- I also read a few articles and original research paper, though I still feel the concepts didn’t fully click.

- Despite that, I was eager to experiment. I played around with different datasets, including Kaggle’s mammals dataset, where I selected one image from each category for testing.

- Inspired by our discussions, I began testing with Myanmar-specific images by collecting them from Google and running them through the models.

- I developed functions to generate charts and reports, using Claude AI to make the process faster and more efficient.

- Through these tests, I observed how model performance varied across different datasets, especially with out-of-distribution cultural images.

Still, I wasn’t completely satisfied with only looking at confidence scores and inference speed. I wanted to dig deeper, so I tried creating fuzzy evaluation functions, thinking it would help capture model performance more accurately. But I quickly realized that because of different labelling systems or the lack of culturally specific labels in datasets like ImageNet, accuracy would still look low, even when the model made somewhat reasonable guesses.

To get around this, I switched to more manual analysis, using Google Image Search to look up the model's predicted keywords and see if they made visual or cultural sense.

At this point, I know my understanding of neural networks is still very basic, and my problem-solving approach to these kinds of challenges is far from mature. But just like the neural networks we work with, I think I need to go deeper.
