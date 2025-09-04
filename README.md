# AliShredder Web | a software to measure phylogenetic quality

The AliShredder web interface facilitates the analysis of biological sequence data through a user-friendly web interface. It allows users to upload sequence files, which are then processed using the AliShredder software [2] for analysis. The application is built using Flask [3], a Python web framework, and incorporates various technologies for file handling, data processing, and visualization.

As a primary objective, computed statistics from AliShredder are used to evaluate the quality of sequence data. The analyses provide valuable insight into the accuracy and reliability of the data and identify which regions should be excluded from further phylogenetic analysis. Therefore, the AliShredder web application ensures the reliability of subsequent phylogenetic analyses by serving as a crucial quality control.

## Features
- Upload multiple sequence alignment (MSA) files in supported formats (e.g., FASTA).  
- Automated **sliding window analysis** to detect informative regions.  
- Visualization of **parsimony-informative sites**.  
- Interactive completeness scoring of alignment windows.  
- Export results for downstream phylogenetic workflows.

## Access

The AliShredder Web Interface is available online — no installation is required.  
Simply visit the following link to start using the application:  [AliShredder Web Interface](http://alishredder.cibiv.univie.ac.at/)  

## User Manual
All usage details are provided in the User Manual included in this repository (see `user_manual.pdf`).  
The manual contains step-by-step web usage, supported input formats, troubleshooting tips — everything a user or reviewer needs to run the web interface and interpret results.

## Biological Background 
In phylogenetic analysis, the goal is to understand the evolutionary relationships among biological entities. Phylogenetic trees represent the evolutionary relationships between sequences, whereas the branches show how closely or distantly related these sequences are. 

However, phylogenetic analysis poses challenges, as certain segments of genetic sequences may not be optimal for tree reconstruction. Factors like missing data, rapid evolution without selective pressure, or limited substitutions due to strong selective pressures can impact the reliability of results [2].

To address this challenge, AliShredder is utilized as a tool to ensure a dependable source for further phylogenetic analysis. Its function is to automate the process of identifying informative sites in genetic sequences by utilizing the sliding window approach. This method involves dividing the genetic sequence into either overlapping or non-overlapping windows of a predetermined size. Each window functions as a localized segment for detailed alignment analysis. 
Afterward, these windows are evaluated to determine which areas are better or less suitable for analysis. The process illustrated in the figure involves the scoring of windows that have been analysed, which in turn provides their respective completeness scores.

<img width="602" height="241" alt="windows" src="https://github.com/user-attachments/assets/ac1efbe9-959e-4b37-a934-49ab1ac163be" /> 


Visualizing the phylogenetic quality of alignment regions can be done in a compelling way by counting the number of parsimony-informative sites. This helps to obtain a more accurate estimation of the tree's structure. By distinguishing between different types of sites, we can gain a better understanding of the information they provide [2]:

- **Fully Informative Site**: Contains at least two different characters, each occurring at least twice and no ambiguous characters.  
- **Partly Informative Site**: May contain ambiguous characters.  
- **Uninformative, but Variable Site**: Doesn't fit the other categories.  
- **Completely Constant Site**: Contains only one character.  
- **Gapped Constant Site**: Contains one character and other ambiguous characters.  

## Architecture Overview
Upload MSA file → AliShredder Core Analysis → Visualization & Export

The **AliShredder Web Interface** is hosted on the CIBIV server, where an Apache web server communicates with the Flask application through a WSGI layer. User-uploaded multiple sequence alignments are processed by the AliShredder software, which integrates IQ-Tree data structures and AliStat metrics to evaluate sequence quality. The system architecture combines backend processing, a user-friendly HTML/JavaScript interface, static assets, and automated cronjobs for cleanup, ensuring seamless data upload, analysis, and visualization.

---
### Acknowledgments

This project was developed as part of a university software project.
I was responsible for data visualization and frontend development.
A colleague contributed to the backend implementation.

### References  

[2] Chirila, B. L. (2019). *Extension and Parallelization of the AliShredder Software for Extracting Characteristics from Multiple Sequence Alignments.* Bachelorarbeit, Universität Wien, Fachrichtung Informatik - Bioinformatik, p.5ff.  

[3] Flask Documentation: [https://flask.palletsprojects.com/en/3.0.x/](https://flask.palletsprojects.com/en/3.0.x/)  
