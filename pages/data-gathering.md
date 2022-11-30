# Data Gathering

First, we prepared the dataset. This task was carried out as a group in the [overview repository](https://git.scc.kit.edu/nlp-scientific-paper/overview). Using scraping scripts we collected the abstracts of software engineering papers from EASE, ESEC_FSE, ICSE, TOSEM and TSE venues in 2020. Then, we manually labeled the dataset and reviewed the labels. Each sentence of an abstract is assigned one or more labels from the target classes. The class hierarchy is provided as a DrawIO file [here](https://git.scc.kit.edu/nlp-scientific-paper/overview/-/tree/main/Classification). 

We have 6 target classes and each sentence is labeled as one or more of these target classes: 

1. Background and motivation (BM)
   - Non-original information, informs on the relevance of the paper topic
   - Example: *"Data synthesis is one of the most significant tasks in Systematic Literature Review (SLR)"*
   - Typical phrases include 
     - *in recent years*, 
     - *... is a challenging task*, 
     - *... has become more relevant*, 
     - and also quite typically a definition of a concept/technology which is at the center of the paper 
2. Aim and contribution (AC)
   - What is this paper aiming to achieve? What is the scientific contribution? 
   - Example: *"This paper presents an automated machine learning-based approach for classifying the role-stereotype of classes in Java."*
   - Typical phrases include
     - *The aim of this paper/study*
     - *We aim at*
     - *We present*
   - Issues:
     - Sentences following an AC sentence sometimes describe the details of the contribution or side contributions (e.g. a new algorithm to arrive at the actually intended contribution). These sentences are hard to separate from RO.  
3. Research object (RO)
   - Which specific scientific field is under investigation? 
   - Example: *"This research investigates the evolution of object-oriented inheritance hierarchies in open source, Java systems."*
   - Typical phrases include
     - *we investigate*
     - *we examine*
     - *to study how*
     - *we hypothesize*
   - Issues:
     - Technical details of the research such as the algorithms used, the data that has been collected etc. are hard to separate from AC
4. Research method (RM)
   - Used method for validation, evaluation of own work. Empirical analysis, surveys, theoretical proofs etc. 
   - Example: *"We carried out an empirical inquiry by integrating SLR and confirmatory email survey."*
   - Typical phrases include
     - *empirical*
     - *survey*
     - *we compare our method to state-of-the-art*
     - *tested on XYZ datasets* 
5. Results and findings (RF)
   - Concrete results with numbers (quantitative), qualitative interpretation of the results
   - Example: *"This analysis shows that the Random Forest algorithm yields the best classification performance."* 
   - Typical phrases include
     - *results show that ...*
     - *our experiment results*
     - *our method outperforms other state-of-the-art methods*
     - *we find that ...*
     - *this study confirms ...*
6. Summary (SU)
   - Wrap-up, hint at future work, limitations of the study etc. 
   - Example: *"Based on these perceptions, we conclude that automatic code inspection is considerably relevant in an end-user development context such as VBA."*
   - Typical phrases include
     - *we conclude*
   - Issues:
     - It is hard to differentiate between the summary of results/findings and the whole paper. Prior to the labeling efforts we didn't clear this distinction which resulted in a blurry line between RF and SU. During our reviews we then decided that the summary must be more than just the summary of the results and findings. E.g. *"The experiment results show that the prototyped system with the proposed learning approach not only yields significant knowledge gain compared to the conventional learning approach but also gains better learning satisfaction of students."* should not be labeled as SU but RF. I argue for a need of reconsideration since this distinction is hard to make even by humans and the first classifier prototypes have very poor test results for SU.  

## Issues 

Labeling sentences with the target classes above hasn't been trivial. Quickly we've seen that the chosen classes aren't easily separable from each other. Below are difficult class pairs with examples: 

   - RF <-> SU
     - In some cases the summary includes concrete results/findings. 
     - E.g. *"Findings: Whilst the trend of posted questions is sharply increasing, the questions are becoming more specific to technologies and more difficult to attract answers."* 
       - This sentence can be classified both as a conclusion (falls under SU) and a qualitative result (falls under RF). RF label seems obvious (also classified as such by the author) and needs no justification. SU label also makes sense since this sentence concludes the research with an outlook into the future. 
     - Note: To achieve better separation we have decided to limit summary strictly to summaries of the **whole** paper, not just the final results. This results in many papers not having a summary sentence. 
   - AC <-> RO
     - The research object is sometimes also a contribution. Therefore it has been hard to separate both classes clearly. 
     - E.g. *"Method: This paper extends the graph matching technology based on the past and proposes a new method that combines network analysis and structural matching for detection."
       - The research object is graph matching. But it is also a contribution since they propose a new method. 

Comparing the labels given by Vladimir with mine, the confusion becomes more clear. Below are two graphs showing the correlation between target classes in my labels (left) and Vladimir's (right). Source code to reproduce the correlation matrices and visualizations can be found [here](label_correlations.ipynb)

![](img/corr_utku.png "Label correlations (Utku)") 
![](img/corr_vladimir.png "Label correlations (Vladimir)")

We can see that the two datasets differ significantly between the distinction of AC and RO. This conflict can be solved by the reviewer's final rule but having such a major difference of understanding, this might indicate a poor choice and definition of target classes. Therefore it should be considered to merge the two classes into one. The distinction between AC and RO can still be made in a deeper classification level (see the hierarchical classification diagram [here](https://git.scc.kit.edu/nlp-scientific-paper/overview/-/tree/main/Classification). 

Label distribution of the complete dataset is visualized as a histogram below: 

![](img/label_distribution_all.png) 

It is easy to recognize that the dataset is highly imbalanced, especially for RO and SU classes. 
To mitigate the imbalancedness of the dataset an alternative dataset has been created with 4 labels where AC/RO and RF/SU class pairs have been merged. The resulting dataset has a more balanced label distribution: 

![](img/label_distribution_all_merged_categories.png)
 
Also we see less correlation between classes: 

![](img/corr_all_merged_categories.png)

## Datasets

In the end we have 2 different datasets one of which has 2 pairs of classes merged which results in 2 less target classes as described in [issues](#issues):

- [Complete dataset with 6 target classes](data/sentences_all.csv)
- [Complete dataset with 4 target classes](data/sentences_all_merged_categories.csv)

Both datasets consist of paper abstracts from various venues to represent variability in style. 

The table below summarizes the nature of our datasets:

![](img/dataset_insights.png)